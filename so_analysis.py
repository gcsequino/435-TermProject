from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import MapType, StringType
from sentiment_analysis import CommentSentimentAnalyer
import argparse
import os


def get_answered_posts(data_dir, limit=None):
    df = spark.read.parquet(os.path.join(data_dir, 'Posts.parquet'))
    # Rename _AcceptedAnswerId to TmpId so that there won't be a duplicate column in following join
    accepted_answer_id_df = df.select('_AcceptedAnswerId').distinct().withColumnRenamed('_AcceptedAnswerId', 'TmpId')

    # create DF with only posts that are "Accepted" answers (since both posts and answers are in the posts table)
    if limit:
        accepted_answer_posts_df = df.join(accepted_answer_id_df, df['_Id'] == accepted_answer_id_df['TmpId']).limit(limit)
    else:
        accepted_answer_posts_df = df.join(accepted_answer_id_df, df['_Id'] == accepted_answer_id_df['TmpId'])

    return accepted_answer_posts_df




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--spark", type=str, required=False,
                   help="Address of the Spark master.")
    p.add_argument("-d", "--data_dir", type=str, required=True,
                   help="Data directory in HDFS.", default='/TermProject/data')
    p.add_argument("-l", "--limit", type=int, default=None,
                   help="Limit number of samples to process.")
    args = p.parse_args()
    if args.spark:
        context = SparkContext(appName="Stackoverflow Analysis", master=args.spark).getOrCreate()
    else:
        context = SparkContext(appName="Stackoverflow Analysis").getOrCreate()
    spark = SparkSession(context)
    context.setLogLevel("ERROR")
    so_users = spark.read.parquet(os.path.join(args.data_dir, "Users.parquet"))
    so_users.printSchema()
    print("User count", so_users.count())
    so_posts = spark.read.parquet(os.path.join(args.data_dir, "Posts.parquet"))
    so_posts.printSchema()
    print("Post count", so_posts.count())


    print("Getting accepted answers")
    accepted_answer_posts_df = get_answered_posts(args.data_dir, limit=args.limit)
    print(f"Got {accepted_answer_posts_df.count()} accepted answer posts")

    print("Running Sentiment Analysis")
    context.addPyFile("sentiment_analysis.py")
    out_type = MapType(StringType(), StringType(), False)
    analyze_udf = udf(lambda b: CommentSentimentAnalyer().get_sentiment_polarity(b), out_type)
    accepted_answer_posts_df = accepted_answer_posts_df.withColumn("sentiments", analyze_udf("_Body"))
    accepted_answer_posts_df = accepted_answer_posts_df.withColumn('negative', accepted_answer_posts_df["sentiments"].getItem('neg'))\
        .withColumn('positive', accepted_answer_posts_df["sentiments"].getItem('pos'))\
        .withColumn('neutral', accepted_answer_posts_df["sentiments"].getItem('neu'))\
        .withColumn('compound', accepted_answer_posts_df["sentiments"].getItem('compound'))

    accepted_answer_posts_df.write.parquet(os.path.join(args.data_dir, 'PostsWithSentiment.parquet'), 'overwrite')


