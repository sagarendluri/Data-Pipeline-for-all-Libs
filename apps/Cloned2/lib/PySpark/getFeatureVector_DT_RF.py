import findspark
findspark.init()
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import sys
import os
class spark():
    def __init__(self, df, i,drop_col ):
        self.df = df
        self.i = i
        self.drop_col=drop_col 
    def models(self):
        try:
            spark = SparkSession.builder \
                    .master("local[*]") \
                    .config("spark.executor.memory", "70g") \
                    .config("spark.driver.memory", "50g") \
                    .config("spark.memory.offHeap.enabled",True) \
                    .config("spark.memory.offHeap.size","16g") \
                    .appName("gp") \
                    .getOrCreate()
            self.dfd = spark.read.option("maxcolumns",1800000).csv(self.df,header  = True,inferSchema=True)
            self.dfd = self.dfd.drop("index","ID")#*self.drop_col)
            return self.dfd
        except Exception as Ex:
            print("Read_csv exited with the error : ")
            print(Ex)
    def cleanDataFrame(self):
        try:
#             indexers = self.i+"_index"#[StringIndexer(inputCol=column, outputCol=column + "_index").fit(self.dfd) for column in [self.i]]
#             pipeline = Pipeline(stages=indexers)
#             ddf = pipeline.fit(self.dfd).transform(self.dfd)
#             final = self.dfd
            feature_list = []
            for col in self.dfd.columns:
                if col == self.i:
                    continue
                else:
                    feature_list.append(col)

            assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
            final = self.dfd
            return assembler,final
        except Exception as Ex:
            print("VectorAssembler exited with the error : ")
            print(Ex)
