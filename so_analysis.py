from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, count, col, avg
from pyspark.sql.types import MapType, StringType, ArrayType
from CommentSentimentAnalyzer import CommentSentimentAnalyzer
import argparse
import os
import re


def get_answered_posts(spark, data_dir, limit=None):
    df = spark.read.parquet(os.path.join(data_dir, 'Posts.parquet'))
    # Rename _AcceptedAnswerId to TmpId so that there won't be a duplicate column in following join
    accepted_answer_id_df = df.select('_AcceptedAnswerId').distinct().withColumnRenamed('_AcceptedAnswerId', 'TmpId')

    # create DF with only posts that are "Accepted" answers (since both posts and answers are in the posts table)
    if limit:
        accepted_answer_posts_df = df.join(accepted_answer_id_df, df['_Id'] == accepted_answer_id_df['TmpId']).limit(limit)
    else:
        accepted_answer_posts_df = df.join(accepted_answer_id_df, df['_Id'] == accepted_answer_id_df['TmpId'])

    return accepted_answer_posts_df


def get_top_tags(spark: SparkSession, data_dir, limit=None):
    posts = spark.read.parquet(os.path.join(data_dir, 'Posts.parquet'))
    post_tags = posts.select("_Tags").dropna().withColumnRenamed("_Tags", "tags")
    split_tag = udf(lambda tag: re.sub("(^<|>$)", "", tag).split("><"), ArrayType(StringType()))
    split_tags = post_tags.select(split_tag(post_tags.tags).alias("tags"))
    all_tags = split_tags.select(explode("tags").alias("tags"))
    tags = all_tags.groupBy("tags").agg(count("tags").alias("counts")).orderBy("counts", ascending=False)
    if limit:
        tags = tags.limit(limit)
    return tags


def get_experts_by_reputation(users_df, posts_df):
    top_n = 50
    experts_by_rep_df = users_df.orderBy(col('_Reputation').desc()).limit(top_n).withColumnRenamed('_Id', 'UserId')
    #for row in experts_by_rep_df.take(experts_by_rep_df):
    #    print(f"{row['_DisplayName']}, Rep: {row['_Reputation']}, U: {row['_UpVotes']}, D: {row['_DownVotes']}, V: {row['_Views']}, U/D: {row['_UpVotes']/row['_DownVotes'] if row['_DownVotes'] else row['_UpVotes']}")
    expert_posts_df = posts_df.join(experts_by_rep_df, posts_df['_OwnerUserId'] == experts_by_rep_df['UserId'])
    expert_posts_df_with_sentiment = add_sentiment(expert_posts_df, '_Body')
    experts_sentiment_df = expert_posts_df_with_sentiment.groupBy(['UserId', '_DisplayName']).agg(avg('positive').alias('average_positive'),
                                                                                                  avg('negative').alias('average_negative'),
                                                                                                  avg('neutral').alias('average_neutral'),
                                                                                                  avg('compound').alias('average_compound'),
                                                                                                  avg(col('_UpVotes') / col('_DownVotes')).alias('UpVotesToDownVotes'),
                                                                                                  count('_Id'))

    experts_sentiment_df.orderBy(col('average_negative').desc()).show(top_n)


def add_sentiment(df, column_to_analyze):
    # Adds sentiment columns to dataframe
    print("Running Sentiment Analysis")
    out_type = MapType(StringType(), StringType(), False)
    analyze_udf = udf(lambda b: CommentSentimentAnalyzer().get_sentiment_polarity(b), out_type)
    df_with_sentiment = df.withColumn("sentiments", analyze_udf(column_to_analyze))
    df_with_sentiment = df_with_sentiment.withColumn('negative', df_with_sentiment["sentiments"].getItem('neg'))\
        .withColumn('positive', df_with_sentiment["sentiments"].getItem('pos'))\
        .withColumn('neutral', df_with_sentiment["sentiments"].getItem('neu'))\
        .withColumn('compound', df_with_sentiment["sentiments"].getItem('compound'))
    return df_with_sentiment


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--spark", type=str, required=False,
                   help="Address of the Spark master.")
    p.add_argument("-d", "--data_dir", type=str, required=True,
                   help="Data directory in HDFS.", default='/TermProject/data')
    p.add_argument("-l", "--limit", type=int, default=None,
                   help="Limit number of samples to process.")
    p.add_argument("-q", "--question", type=int, default=None,
                   help="Question scenario number (1-5)")
    args = p.parse_args()
    if args.spark:
        context = SparkContext(appName="Stackoverflow Analysis", master=args.spark).getOrCreate()
    else:
        context = SparkContext(appName="Stackoverflow Analysis").getOrCreate()
    spark = SparkSession(context)
    context.addPyFile("CommentSentimentAnalyzer.py")
    context.setLogLevel("ERROR")
    so_users = spark.read.parquet(os.path.join(args.data_dir, "Users.parquet"))
    so_users.printSchema()
    print("User count", so_users.count())
    so_posts = spark.read.parquet(os.path.join(args.data_dir, "Posts.parquet"))
    so_posts.printSchema()
    print("Post count", so_posts.count())

    if args.question:
        if args.question == 1:
            print(get_top_tags(spark, args.data_dir))
        elif args.question == 2:
            pass
        elif args.question == 3:
            pass
        elif args.question == 4:
            pass
        elif args.question == 5:
            experts_by_rep_df = get_experts_by_reputation(so_users, so_posts)

    else:
        # Run all?
        pass

    # accepted_answer_posts_df = get_answered_posts(spark, args.data_dir, limit=args.limit)
    # accepted_answer_posts_df = add_sentiment(accepted_answer_posts_df, "_Body")
    # accepted_answer_posts_df.write.parquet(os.path.join(args.data_dir, 'PostsWithSentiment.parquet'), 'overwrite')


