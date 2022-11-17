from pyspark import SparkContext
from pyspark.sql import SparkSession

if __name__ == "__main__":
    context = SparkContext(appName="Stackoverflow Analysis").getOrCreate()
    spark = SparkSession(context)
    context.setLogLevel("ERROR")
    so_users = spark.read.parquet("/TermProject/data/Users.parquet")
    so_users.printSchema()
    print("User count", so_users.count())
    so_posts = spark.read.parquet("/TermProject/data/Posts.parquet")
    so_posts.printSchema()
    print("Post count", so_posts.count())