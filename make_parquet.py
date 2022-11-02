import argparse
from pyspark.sql import SparkSession


def make_parquet(spark, xml_file, parquet_file):
    df = spark.read.format('xml').option('rowTag', 'row').load(xml_file)
    df.write.parquet(parquet_file)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--spark", type=str, required=True, default="spark://salem:30333",
                       help="Address of the Spark instance.")
    p.add_argument("-x", "--xml_file", type=str, required=True,
                       help="Source XML file in HDFS.", default='/SO/Posts.xml')
    p.add_argument("-p", "--parquet_file", type=str, required=True,
                       help="Destination Parquet file in HDFS.", default='/SO/posts.parquet')
    args = p.parse_args()
    spark = SparkSession.builder.master(args.spark).getOrCreate()
    make_parquet(spark, args.xml_file, args.parquet_file)
