import findspark
findspark.init()
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report
from pyspark.ml.evaluation import RegressionEvaluator
from sklearn import metrics
import numpy as np
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_DB import DB_upload
import apps.Cloned2.lib.PBS.Upload_Data_Database.DB_details
import pandas as pd
from sklearn.model_selection import train_test_split
from smart_open import smart_open
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from configparser import ConfigParser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
class DTreeClassifier():
    def __init__(self,feature_vector,df,dname,i,classifiation_or_Regression,config_object_user,
                         user_defined_terminology,sample_type ,description ,uom_type,d_col,analysis_id):
        self.feature_vector = feature_vector
        self.df =df
        self.dname =dname
        self.i = i
        self.classifiation_or_Regression = classifiation_or_Regression
        self.config_object_user = config_object_user
        self.user_defined_terminology = user_defined_terminology
        self.sample_type =sample_type
        self.description = description
        self.uom_type = uom_type
        self.d_col = d_col
        self.analysis_id= analysis_id
    def building(self):
        try:
            if self.classifiation_or_Regression =="Classification":
                rf = DecisionTreeClassifier(labelCol="label", featuresCol="features")
                from pyspark.ml import Pipeline
                pipeline = Pipeline(stages=[self.feature_vector, rf])
                from pyspark.ml.tuning import ParamGridBuilder
                import numpy as np
                paramGrid = ParamGridBuilder() \
                        .addGrid(rf.maxDepth, [4, 6, 8]) \
                        .addGrid(rf.maxBins, [5, 10, 20, 40]) \
                        .build()
                from pyspark.ml.tuning import CrossValidator
                from pyspark.ml.evaluation import MulticlassClassificationEvaluator
                crossval = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=MulticlassClassificationEvaluator(),
                        numFolds=3)
                df = self.df.withColumnRenamed(self.i, 'label')
                (trainingData, testData) =df.randomSplit([0.8, 0.2])
                cvModel = crossval.fit(trainingData)
                print("T_finished")
                rf_prediction = cvModel.transform(testData)
                self.i ,target = ["label",self.i]
                preds_and_labels =rf_prediction.select("prediction", self.i).withColumn('label', F.col(self.i).cast(FloatType())).orderBy('prediction')
                #select only prediction and label columns
                preds_and_labels = preds_and_labels.select(['prediction','label'])
                metric= MulticlassMetrics(preds_and_labels.rdd.map(tuple))
                cm2 = metric.confusionMatrix().toArray()
                print(cm2)
                y_test = preds_and_labels.select([self.i]).collect()
                y_pred = preds_and_labels.select(['prediction']).collect()
                cm = confusion_matrix(y_test, y_pred)
                print("CM",cm)
                Accuracy = metrics.accuracy_score(y_test, y_pred)
                print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#                 print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#                 print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#                 print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                evaluator = MulticlassClassificationEvaluator(labelCol=self.i, predictionCol="prediction",
                        metricName="accuracy")
                rf_accuracy = evaluator.evaluate(rf_prediction)
                print(rf_accuracy)
                config_object_user =self.config_object_user
                user_defined_terminology =self.user_defined_terminology
                sample_type = self.sample_type
                description = self.description
                uom_type =self.uom_type
                bestPipeline = cvModel.bestModel
                bestModel = bestPipeline.stages[1]
                importances = bestModel.featureImportances
                bestModel = bestPipeline.stages[1]
                estimator = bestModel.extractParamMap()
                cvModel.bestModel.extractParamMap()
                x_values = list(range(len(importances)))
                config_object = ConfigParser()
                ini = config_object.read(r'/dse/apps/Cloned2/config.ini')
                config_object.read(ini)
                config_object_project = ConfigParser()
                userinfo = config_object[config_object_user]
                access_key_id = userinfo["access_key_id"]
                secret_access_key = userinfo["secret_access_key"]
                bucket_name = userinfo["bucket"]
                path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, self.dname)
                df = pd.read_csv(smart_open(path))
                df1=df.select_dtypes(exclude=['object'])
#                 def drop(df):
#                     try:
#                         if self.d_col==None:
#                             return df
#                         else:
#                             df = df.drop(self.d_col,axis=1)
#                             return df
#                     except:
#                         print("Give valid columns which columns dataset has")
#                 df1 = drop(df)
                print(df1.columns)
                dname = self.dname
                X= df1.drop(target,axis=1)
                y=df1[target]
                X_train, X_test, y_train, y_test_p = train_test_split(X,y, test_size = 0.2,random_state = 42)
                l=1
                model=user_defined_terminology+ target+".model"
#                 None = model
                cvModel.save(model)
                print("model_finished")
                analysis_id= self.analysis_id
                DB_upload(Accuracy,X_train,X_test,y_test,y_pred, 
                              importances,grid,estimator,l,cm,target,None,None,dname,config_object_user,
                         user_defined_terminology,sample_type ,description ,uom_type,analysis_id)
            else:
                rf = DecisionTreeRegressor(labelCol="label", featuresCol="features")
                from pyspark.ml import Pipeline
                pipeline = Pipeline(stages=[self.feature_vector, rf])
                from pyspark.ml.tuning import ParamGridBuilder
                import numpy as np
                paramGrid = ParamGridBuilder() \
                        .addGrid(rf.maxDepth, [4, 6, 8]) \
                        .addGrid(rf.maxBins, [5, 10, 20, 40]) \
                        .build()
                from pyspark.ml.tuning import CrossValidator
                from pyspark.ml.evaluation import MulticlassClassificationEvaluator
                crossval = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(),
                        numFolds=3)
                df = self.df.withColumnRenamed(self.i, 'label')
                (trainingData, testData) =df.randomSplit([0.8, 0.2])
                cvModel = crossval.fit(trainingData)
                self.i,target = ["label" , self.i]
                rf_prediction = cvModel.transform(testData)
                preds_and_labels =rf_prediction.select("prediction", self.i).withColumn('label', F.col(self.i).cast(FloatType())).orderBy('prediction')
                preds_and_labels = preds_and_labels.select(['prediction','label'])
                y_test = preds_and_labels.select([self.i]).collect()
                y_pred = preds_and_labels.select(['prediction']).collect()
                dt_evaluator = RegressionEvaluator(
                labelCol="label", predictionCol="prediction", metricName="r2")
                rmse = dt_evaluator.evaluate(rf_prediction)
                print("Root Mean Squared Error (r2) on test data = %g" % rmse)
                config_object_user =self.config_object_user
                user_defined_terminology =self.user_defined_terminology
                sample_type = self.sample_type
                description = self.description
                uom_type =self.uom_type
                bestPipeline = cvModel.bestModel
                bestModel = bestPipeline.stages[1]
                importances = bestModel.featureImportances
                bestModel = bestPipeline.stages[1]
                estimator = bestModel.extractParamMap()
                cvModel.bestModel.extractParamMap()
                x_values = list(range(len(importances)))
                config_object = ConfigParser()
                ini = config_object.read(r'/dse/apps/Cloned2/config.ini')
                config_object.read(ini)
                config_object_project = ConfigParser()
                userinfo = config_object[config_object_user]
                access_key_id = userinfo["access_key_id"]
                secret_access_key = userinfo["secret_access_key"]
                bucket_name = userinfo["bucket"]
                path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, self.dname)
                df = pd.read_csv(smart_open(path))
#                 def drop(df):
#                     try:
#                         if self.d_col==None:
#                             return df
#                         else:
#                             df = df.drop(self.d_col,axis=1)
#                             return df
#                     except:
#                         print("Give valid columns which columns dataset has")
#                 df1 = drop(df)
                
                df1=df.select_dtypes(exclude=['object'])
                print(df1.columns)
                dname = self.dname
                target=target
                print(target)
                model_file_name = user_defined_terminology
                X= df1.drop(target,axis=1)
                y=df1[target]
                X_train, X_test, y_train, y_test_p = train_test_split(X,y, test_size = 0.2,random_state = 42)
                l=1
                model=user_defined_terminology+ target+"."+"model"
#                 None = model
                cvModel.save(model)
                analysis_id= self.analysis_id
#                 cm = classification_report(y_test, y_pred)
#                 print("CM",cm)
#                 print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#                 print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#                 print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#                 print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                DB_upload(None,X_train,X_test,y_test,y_pred, 
                              importances,None,estimator,l,None,target,None,None,dname,config_object_user,
                         user_defined_terminology,sample_type ,description,uom_type,analysis_id)
        except Exception as Ex:
            print("DTClassifier exited with the error : ")
            print(Ex)
        

