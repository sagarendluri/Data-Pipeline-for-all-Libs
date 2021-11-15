import findspark
findspark.init()
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from sklearn import metrics
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
class NBClassifier():
    def __init__(self,input_nodes,vectorAssembler,indexer):
        self.input_nodes = input_nodes
        self.vectorAssembler = vectorAssembler
        self.indexer = indexer
    def building(self):
        try:
            self.input_nodes = self.input_nodes.replace(-1,0)
            dfd = self.input_nodes.replace(-2,0)
            (trainingData, testData) = dfd.randomSplit([0.8, 0.3], seed = 100)
            nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
            pipeline = Pipeline(stages=[ self.indexer,self.vectorAssembler, nb])
            model = pipeline.fit(trainingData)
            predictions = model.transform(testData)
            predictions.select("label", "probability", "prediction").show()
            evaluator =MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            predictionAndLabel = predictions.select("prediction", "label").rdd
            from pyspark.sql.types import Row

            #here you are going to create a function
            def f(x):
                d = {}
                for i in range(len(x)):
                    d[str(i)] = x[i]
                return d
            #Now populate that
            df = predictionAndLabel.map(lambda x: Row(**f(x))).toDF()
            print(df)

            
#             y_test = accuracy.select(['1st_Layer_Clusters_index']).collect()
#             y_pred = accuracy.select(['prediction']).collect()
#             cm = classification_report(y_test, y_pred)
            print("CM",cm)
            print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            importance_list = pd.Series(model.featureImportances.values)
            sorted_imp = importance_list.sort_values(ascending= False)
            print("importance",sorted_imp)
            kept = list((sorted_imp[sorted_imp > 0.03]).index)

        except:
            print("Failed to build NBClassifier")

