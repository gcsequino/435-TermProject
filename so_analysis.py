from enum import Enum
from pyspark import SparkContext

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, explode, count, col, avg, broadcast, when, mean
import pyspark.sql.functions as psf

from pyspark.sql.types import MapType, StringType, ArrayType
from CommentSentimentAnalyzer import CommentSentimentAnalyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re
import matplotlib as plt

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
    answer_posts = posts.select("_Id", "_Body", "_ParentId", "_OwnerUserId").where("_PostTypeId == 2").alias("answers")
    questions_with_answers = question_posts.join(answer_posts, col("questions._Id") == col("answers._ParentId"))
    answers = questions_with_answers.select(col("answers._Id"), col("answers._Body"), col("questions._Tags"), col("answers._OwnerUserId"))
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
    top_n = 100
    experts_by_rep_df = users_df.orderBy(col('_Reputation').desc()).limit(top_n).withColumnRenamed('_Id', 'UserId')
    #for row in experts_by_rep_df.take(experts_by_rep_df):
    #    print(f"{row['_DisplayName']}, Rep: {row['_Reputation']}, U: {row['_UpVotes']}, D: {row['_DownVotes']}, V: {row['_Views']}, U/D: {row['_UpVotes']/row['_DownVotes'] if row['_DownVotes'] else row['_UpVotes']}")
    expert_posts_df = posts_df.join(experts_by_rep_df, posts_df['_OwnerUserId'] == experts_by_rep_df['UserId'])
    expert_posts_df_with_sentiment = add_sentiment(expert_posts_df, '_Body')
    expert_posts_df_with_sentiment.show(5)
    experts_sentiment_df = expert_posts_df_with_sentiment.groupBy(['UserId', '_DisplayName']).agg(avg('positive').alias('average_positive'),
                                                                                                  avg('negative').alias('average_negative'),
                                                                                                  avg('neutral').alias('average_neutral'),
                                                                                                  avg('compound').alias('average_compound'),
                                                                                                  psf.sum('overall_is_positive').alias('total_positive'),
                                                                                                  psf.sum('overall_is_negative').alias('total_negative'),
                                                                                                  avg(col('_UpVotes') / col('_DownVotes')).alias('UpVotesToDownVotes'),
                                                                                                  count('_Id'))

    experts_sentiment_df.orderBy(col('average_negative').desc()).repartition(1).write.mode('overwrite').options(header='True', delimiter=',').csv('/tmp/expert_sentiment.csv')


def get_average_sentiment(posts_df):
    print("Getting overall sentiment metrics")
    posts_df_with_sentiment = add_sentiment(posts_df, '_Body')
    posts_df_with_sentiment = posts_df_with_sentiment.select(avg('positive').alias('average_positive'), 
                                                             avg('negative').alias('average_negative'),
                                                             avg('neutral').alias('average_neutral'),
                                                             avg('compound').alias('average_compound'),
                                                             psf.sum('overall_is_positive').alias('total_positive'),
                                                             psf.sum('overall_is_negative').alias('total_negative'),
                                                             count('positive').alias('total_count'))
    posts_df_with_sentiment.repartition(1).write.mode('overwrite').options(header='True', delimiter=',').csv('/tmp/overall_sentiment.csv')

def get_experts_by_subject(users, posts, m_users=50, n_subjects=20):
    top_tags_df = get_top_tags(posts, n_subjects)
    top_tags_df.show()

    answers_with_tags = get_answers_for_tags(posts, top_tags_df)

    users_per_tag = {} # <tag, df of users>

    for tag in top_tags_df.select("_Tags").collect():
        tag_answers_df = answers_with_tags.filter(answers_with_tags._Tags == tag._Tags) # get all answers for specific tag

        # get sentiments for each accepted answer
        tag_answers_df_with_sentiment = add_sentiment(tag_answers_df, "_Body")

        experts_sentiment_df = tag_answers_df_with_sentiment.groupBy(['_OwnerUserId']).agg(avg('positive').alias('average_positive'),
                                                                                                     avg('negative').alias('average_negative'),
                                                                                                     avg('neutral').alias('average_neutral'),
                                                                                                     avg('compound').alias('average_compound'),
                                                                                                     count('_OwnerUserId').alias('total_accepted_answers'))

        # order tag_answers_df by the count of user id
        top_users_sorted = experts_sentiment_df.orderBy("total_accepted_answers", ascending=False).limit(m_users).dropna()
        
        # users_with_accepted_answers = users.join(answers # get users with accepted answers
        top_m_users = users.join(top_users_sorted, users._Id == top_users_sorted._OwnerUserId, "inner").orderBy("total_accepted_answers", ascending=False)

        # upvotes / downvotes
        top_m_users = top_m_users.withColumn("UpVotesToDownVotes", (top_m_users._UpVotes / top_m_users._DownVotes))

        users_per_tag[tag._Tags] = top_m_users.select(top_m_users._DisplayName, top_m_users._Id, top_m_users.total_accepted_answers, top_m_users.average_compound, top_m_users.average_negative, top_m_users.average_neutral, top_m_users.average_positive, top_m_users.UpVotesToDownVotes)

        average_sentiments = top_m_users.select(mean(top_m_users.total_accepted_answers), mean(top_m_users.average_compound), mean(top_m_users.average_negative), mean(top_m_users.average_neutral), mean(top_m_users.average_positive), mean(top_m_users.UpVotesToDownVotes))

        users_per_tag[tag._Tags + "_AVGS"] = average_sentiments

    return users_per_tag

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
    df_with_sentiment = df_with_sentiment.withColumn('overall_is_positive', when(col('compound') > 0.05, 1).otherwise(0))
    df_with_sentiment = df_with_sentiment.withColumn('overall_is_negative', when(col('compound') < -0.05, 1).otherwise(0))
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
                                                                                          tagLimit=20)
            avg_q_comment_sentiments = question_comment_sentiment.groupBy("_Tags").agg(avg('positive').alias('average_positive'),
                                                                                                avg('negative').alias('average_negative'),
                                                                                                avg('neutral').alias('average_neutral'),
                                                                                                avg('compound').alias('average_compound'))
            avg_q_comment_sentiments.printSchema()
            avg_a_comment_sentiments = answer_comment_sentiment.groupBy("_Tags").agg(avg('positive').alias('average_positive'),
                                                                                                avg('negative').alias('average_negative'),
                                                                                                avg('neutral').alias('average_neutral'),
                                                                                                avg('compound').alias('average_compound'))
        elif args.question == 4:
            print("Top M 'experts' per top N subjects.")
            experts_by_subject_df = get_experts_by_subject(so_users, so_posts, m_users=50, n_subjects=args.limit)                
            
        elif args.question == 5:
            experts_by_rep_df = get_experts_by_reputation(so_users, so_posts)
            get_average_sentiment(so_posts)

    else:
        # Run all?
        pass

    # accepted_answer_posts_df = get_answered_posts(spark, args.data_dir, limit=args.limit)
    # accepted_answer_posts_df = add_sentiment(accepted_answer_posts_df, "_Body")
    # accepted_answer_posts_df.write.parquet(os.path.join(args.data_dir, 'PostsWithSentiment.parquet'), 'overwrite')


