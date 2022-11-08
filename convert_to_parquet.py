from pyspark import SparkContext
from pyspark.sql import SparkSession

file_prefix = "/TermProject/data/"
xml_files = ["Badges.xml", "Comments.xml", "Posts.xml", "Tags.xml", "Users.xml"]

if __name__ == "__main__":
    context = SparkContext(appName="Stackoverflow convert xml files to parquet").getOrCreate()
    spark = SparkSession(context)
    for file in xml_files:
        file_name = file.split(".")[0]
        root_tag = file_name.lower()
        xml = spark.read.format("xml")\
                            .options(rowTag="row", rootTag=root_tag)\
                            .load(file_prefix + file)
        xml.write.parquet(file_prefix + file_name + ".parquet")