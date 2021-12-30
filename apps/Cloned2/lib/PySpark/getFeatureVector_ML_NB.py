from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import os
class Spark_():
    def __init__(self, df, i):
        self.df = df
        self.i = i
    def models(self):
        try:
            spark = SparkSession.builder \
                    .master("local[*]") \
                    .config("spark.executor.memory", "70g") \
                    .config("spark.driver.memory", "50g") \
                    .config("spark.memory.offHeap.enabled",True) \
                    .config("spark.memory.offHeap.size","16g") \
                    .appName("genome_and_phenome") \
                    .getOrCreate()
            self.dfd = spark.read.option("maxcolumns",1800000).csv(self.df,header  = True,inferSchema=True)
            return self.dfd
        except:
            print("Pyspark Failed to Read the csv file")
    def nodes(self):
        try:
            output_nodes = self.dfd.select(self.i).distinct().count()
            self.input_nodes = self.dfd.drop('_c0')
            print(self.input_nodes)
            i_n = len(self.input_nodes.columns[:-1])
            return output_nodes, i_n, self.input_nodes
        except:
            print("Failed to Retrun MLP Nodes")
    def data_nodes(self):
        try:
            vectorAssembler = VectorAssembler(inputCols=self.input_nodes.columns[:-1], outputCol='features')
            df = vectorAssembler.transform(self.input_nodes)
            indexer = StringIndexer(inputCol=self.i, outputCol='label')
            df1 = indexer.fit(df).transform(df)
            return df1,vectorAssembler, indexer
        except:
            print("Failed to return vectorsa and indexer")
    