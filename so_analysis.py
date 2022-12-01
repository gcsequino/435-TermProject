from enum import Enum
from pyspark import SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, explode, count, col, avg, broadcast
from pyspark.sql.types import MapType, StringType, ArrayType
from CommentSentimentAnalyzer import CommentSentimentAnalyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re

class PostType(Enum):
    QUESTION = 1
    ANSWER = 2

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

def split_post_tags(df, tagId="_Tags"):
    split_tag = udf(lambda tag: re.sub("(^<|>$)", "", tag).split("><") if tag else [], ArrayType(StringType()))
    split_tags = df.select("*", split_tag(df[tagId]).alias(f"{tagId}_split"))
    split_tags = split_tags.drop(tagId)
    split_tags = split_tags.select("*", explode(f"{tagId}_split").alias(f"{tagId}_exploded"))
    split_tags = split_tags.drop(f"{tagId}_split")
    split_tags = split_tags.withColumnRenamed(f"{tagId}_exploded", tagId)
    return split_tags

def get_top_tags(posts, limit=None) -> DataFrame:
    post_tags = posts.select("_Tags").dropna()
    all_tags = split_post_tags(post_tags, "_Tags")
    tags = all_tags.groupBy("_Tags").agg(count("_Tags").alias("counts")).orderBy("counts", ascending=False)
    if limit:
        tags = tags.limit(limit)
    return tags

def get_questions_with_tags(posts, tags, limit=None):
    question_posts = posts.select("_Id", "_Body", "_Tags").where(f"_PostTypeId == 1")
    question_posts = split_post_tags(question_posts).alias("questions")
    tags = tags.alias("tags")
    questions_with_tag = question_posts.join(broadcast(tags), col("questions._Tags") == col("tags._Tags"), "left_semi")
    if limit:
        questions_with_tag = questions_with_tag.limit(limit)
    return questions_with_tag

def get_answers_for_tags(posts, tags, limit=None):
    question_posts = get_questions_with_tags(posts, tags).alias("questions")
    answer_posts = posts.select("_Id", "_Body", "_ParentId").where("_PostTypeId == 2").alias("answers")
    questions_with_answers = question_posts.join(answer_posts, col("questions._Id") == col("answers._ParentId"))
    answers = questions_with_answers.select(col("answers._Id"), col("answers._Body"), col("questions._Tags"))
    if limit:
        answers = answers.limit(limit)
    return answers

def get_post_sentiments(posts, postType: PostType, tagLimit=None, postLimit=None) -> DataFrame:
    top_tags = get_top_tags(posts, tagLimit)
    if postType == PostType.QUESTION:
        posts_with_tags = get_questions_with_tags(posts, top_tags, postLimit)
    elif postType == PostType.ANSWER:
        posts_with_tags = get_answers_for_tags(posts, top_tags, postLimit)
    else:
        print("Uknown post type", postType)
        return
    posts_with_tags = add_sentiment(posts_with_tags, "_Body")
    avg_sentiments = posts_with_tags.groupBy("_Tags").agg(avg('positive').alias('average_positive'),
                                                                                                avg('negative').alias('average_negative'),
                                                                                                avg('neutral').alias('average_neutral'),
                                                                                                avg('compound').alias('average_compound'))
    return avg_sentiments

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

def get_comment_sentiments(posts, comments, tagLimit):
    top_tags = get_top_tags(posts, tagLimit)
    questions = get_questions_with_tags(posts, top_tags)
    #print('Top questions: ', questions.count())
    answers = get_answers_for_tags(posts, top_tags)
    #print('Top answers: ', answers.count())

    q_comments = comments.join(questions, comments._PostId == questions._Id, 'leftsemi')
    q_comments = q_comments.join(questions, q_comments._PostId == questions._Id, 'leftouter').select(q_comments._Id, '_PostId', '_Text', '_Tags')

    a_comments = comments.join(answers, comments._PostId == answers._Id, 'leftsemi')
    a_comments = a_comments.join(answers, a_comments._PostId == answers._Id, 'leftouter').select(a_comments._Id, '_PostId', '_Text', '_Tags')

    # print('Total comments: ', comments.count())
    # print('Comments on top questions: ', q_comments.count())
    # print('Comments on top answers: ', a_comments.count())

    return add_sentiment(q_comments, '_Text'), add_sentiment(a_comments, '_Text')

def generate_Q1_figures(top_tags: DataFrame):
    top_tags = dict(top_tags.collect())
    fig, ax = plt.subplots()
    ax.bar(top_tags.keys(), list(map(lambda val: val/1E6, top_tags.values())))
    ax.tick_params(axis="x", rotation=75)
    ax.set_ylabel("number of posts in millions")
    ax.set_ylim(0, 3)
    ax.autoscale(enable=False, axis="y")
    ax.set_title("Top 20 tag count")
    fig.tight_layout()
    fig.savefig("top_tags.png")

def generate_Q2_figures(post_sentiments: DataFrame, name_prefix):
    post_sentiments = dict(post_sentiments.select("_Tags", "average_compound").orderBy("average_compound", ascending=False).collect())
    fig, axes = plt.subplots()
    axes.set_title(f"Polarity of {name_prefix}")
    axes.bar(post_sentiments.keys(), post_sentiments.values())
    axes.tick_params(axis="x", rotation=75)
    axes.set_ylabel("Average compound polarity")
    fig.tight_layout()
    fig.savefig(f"{name_prefix}_polarity.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--spark", type=str, required=False,
                   help="Address of the Spark master.")
    p.add_argument("-d", "--data_dir", type=str, required=False,
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
    print("Starting ...")

    so_users = spark.read.parquet(os.path.join(args.data_dir, "Users.parquet"))
    print("Users schema")
    so_users.printSchema()
    print("User count", so_users.count(), end="\n\n")

    so_posts = spark.read.parquet(os.path.join(args.data_dir, "Posts.parquet"))
    print("Posts schema")
    so_posts.printSchema()
    print("Post count", so_posts.count(), end="\n\n")
    
    so_comments = spark.read.parquet(os.path.join(args.data_dir, "Comments.parquet"))
    print("Comments schema")
    so_comments.printSchema()

    if args.question:
        if args.question == 1:
            top_tag_df = get_top_tags(so_posts, limit=args.limit)
            generate_Q1_figures(top_tag_df)
        elif args.question == 2:
            print("Sentiment analysis over 'Question' posts")
            question_sentiments = get_post_sentiments(so_posts, postType=PostType.QUESTION, tagLimit=20, postLimit=args.limit)
            generate_Q2_figures(question_sentiments, "Questions")
            print("Sentiment analysis over 'Answer' posts")
            answer_sentiments = get_post_sentiments(so_posts, postType=PostType.ANSWER, tagLimit=20, postLimit=args.limit)
            generate_Q2_figures(answer_sentiments, "Answers")
        elif args.question == 3:
            # What is the sentiment of "Comments" linked to posts in the top N subjects?
            question_comment_sentiment, answer_comment_sentiment = get_comment_sentiments(so_posts,
                                                                                          so_comments,
                                                                                          tagLimit=50)
            avg_q_comment_sentiments = question_comment_sentiment.groupBy("_Tags").agg(avg('positive').alias('average_positive'),
                                                                                                avg('negative').alias('average_negative'),
                                                                                                avg('neutral').alias('average_neutral'),
                                                                                                avg('compound').alias('average_compound'))
            avg_a_comment_sentiments = answer_comment_sentiment.groupBy("_Tags").agg(avg('positive').alias('average_positive'),
                                                                                                avg('negative').alias('average_negative'),
                                                                                                avg('neutral').alias('average_neutral'),
                                                                                                avg('compound').alias('average_compound'))
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


