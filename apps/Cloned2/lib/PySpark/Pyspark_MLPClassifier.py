import findspark
findspark.init()
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn import metrics
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
class MLPClassifier():
    def __init__(self,feature_vector,o_n,i_n):
        self.feature_vector = feature_vector
        self.o_n = o_n
        self.i_n = i_n
    def building(self):
        try:
            splits = self.feature_vector.randomSplit([0.8,0.2],1)
            train_df = splits[0]
            test_df = splits[1]
            layers = [self.i_n,100,100,self.o_n]
            mlp = MultilayerPerceptronClassifier(layers = layers, seed = 1)
            mlp_model = mlp.fit(train_df)
            pred_df = mlp_model.transform(test_df)
            evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'accuracy')
            mlpacc = evaluator.evaluate(pred_df)
            
            schema = StructType([StructField(str(i), StringType(), True) for i in range(32)])

            df = sqlContext.createDataFrame(predictionAndLabel, schema)

#             y_test = mlpacc.select(['1st_Layer_Clusters_index']).collect()
#             y_pred = mlpacc.select(['prediction']).collect()
#             cm = classification_report(y_test, y_pred)
#             print("CM",cm)
            print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            importance_list = pd.Series(dt_model.featureImportances.values)
            sorted_imp = importance_list.sort_values(ascending= False)
            print("importance",sorted_imp)
            kept = list((sorted_imp[sorted_imp > 0.03]).index)
        except:
            print("Failed to build MLP")