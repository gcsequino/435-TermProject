# 435-TermProject
Collecting post data from Stack Overflow to identify the technologies that generate the most posts


This will take 20-30 min
```
hadoop fs -mkdir /SO
hadoop fs -put /s/salem/a/tmp/jborg/Posts.xml /SO
```

Example make parquet - Posts may take ~10 min
```
$SPARK_HOME/bin/spark-submit --packages com.databricks:spark-xml_2.12:0.15.0  make_parquet.py -s spark://salem:30333 -x /SO/Posts.xml -p /SO/posts.parquet
```

Afterward, you can confirm parquet exists and is readable, then remove the XML from HDFS. You don't need the XML plugin and can do this from any Python with Pyspark now:

```
source ~/py38_venv/bin/activate
python
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.master('spark://salem:30333').getOrCreate()
>>> df = spark.read.parquet('/SO/posts.parquet')
>>> df.columns
['_AcceptedAnswerId', '_AnswerCount', '_Body', '_ClosedDate', '_CommentCount', '_CommunityOwnedDate', '_ContentLicense', '_CreationDate', '_FavoriteCount', '_Id', '_LastActivityDate', '_LastEditDate', '_LastEditorDisplayName', '_LastEditorUserId', '_OwnerDisplayName', '_OwnerUserId', '_ParentId', '_PostTypeId'
, '_Score', '_Tags', '_Title', '_ViewCount']
>>> df.select('_AcceptedAnswerId').distinct().count()
11755277

```

Running sentiment analyzer (limit 1200 just for example):

```
python so_analysis.py --spark spark://salem:30333 -d /SO -l 1200
hadoop fs -ls /SO/PostsWithSentiment.parquet
```
