import joblib
import pymysql
import json
import sqlalchemy
import pandas as pd
from smart_open import smart_open 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from DB import localdb
from QC import train_test
from os.path import splitext
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.classification import DecisionTreeClassifierModel
from pyspark.ml.regression import DecisionTreeRegressorModel
from Fill_DB import DB_upload 
import numpy as np
import boto3
from Fill_h2o_DB import h2o_DB_upload
class DataPrediction():
    def __init__(self,data,i,path,model_file,user_defined_terminology,sample_type ,description ,uom_type,cut_off,config_object_user,
                aws_access_key_id ,aws_secret_access_key ,bucket_name):
        self.data = data
        self.i = i
        self.path = path
        self.model_file = model_file
        self.user_defined_terminology =user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type =uom_type
        self.cut_off = cut_off
        self.config_object_user =config_object_user
        self.aws_access_key_id=aws_access_key_id
        self.aws_secret_access_key  =aws_secret_access_key
        self.bucket_name =bucket_name
    def split(self):
        cut_off = self.cut_off
        config_object_user = self.config_object_user
        user_defined_terminology =self.user_defined_terminology
        sample_type = self.sample_type
        description = self.description
        uom_type =self.uom_type
        model_file=self.model_file
        aws_access_key_id= self.aws_access_key_id
        aws_secret_access_key  = self.aws_secret_access_key 
        bucket_name =self.bucket_name
        X = self.data.drop([self.i],axis=1)
        y = self.data[self.i]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 42)
        X_train, X_test = train_test(X_train, X_test,cut_off)
        file_name, extension = splitext(self.model_file)
        if y.dtypes == 'int64' or y.dtypes =='int32':
            if extension == '.pkl':
                joblib_file = (smart_open(self.path))
                joblib_LR_model = joblib.load(joblib_file)
                score = joblib_LR_model.score(X_train,y_train)
                print("Test score: {0:.2f} %".format(100 * score))
                y_pred = joblib_LR_model.predict(X_test)
                cm1 =confusion_matrix(y_test, y_pred)
                Accuracy = metrics.accuracy_score(y_test, y_pred)
                print("Accuracy",Accuracy)
                cm = {'confusion_metrics':cm1.tolist()}
                print("CM",cm)
                print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
                print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                estimator, grid = "prediction" ,None
                l = 1
                target = self.i
                model_file_name,model ,dname= self.path ,self.path,self.path
                DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
                                   None,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                             user_defined_terminology,sample_type ,description ,uom_type)
            elif extension == '.h5':
                tf_keras_file = (smart_open(self.path))
                with h5py.File(tf_keras_file, 'r') as f:
                    data = f
                    model = tf.keras.models.load_model(data)
                    y_pred = model.predict(X_test)
                    print(y_pred)
                    cm1 =confusion_matrix(y_test, y_pred)
                    Accuracy = metrics.accuracy_score(y_test, y_pred)
                    print("Accuracy",Accuracy)
                    cm = {'confusion_metrics':cm1.tolist()}
                    print("CM",cm)
                    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
                    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                    estimator, grid = "prediction" , None
                    l = 1
                    target = self.i
                    model_file_name,model ,dname= self.path ,self.path,self.path
                    DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
                                       None,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                                 user_defined_terminology,sample_type ,description ,uom_type)
            elif extension == '.zip':
                import h2o
                from h2o.automl import H2OAutoML
                h2o.init()
                cols = list(X_train.columns)
                cols.append(self.i)
                selected_cols = self.data[cols]
                x=list(cols)
                y = self.i
                df = h2o.H2OFrame(selected_cols)
                train,  test = df.split_frame(ratios=[.8])
                session = boto3.Session(
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                 )
                s3 = session.resource('s3')
                s3.meta.client.download_file(Filename=model_file, Bucket=bucket_name, Key=model_file)
                mdl = h2o.import_mojo(model_file)
                predictions = mdl.predict(test)
                rmse = mdl.model_performance().rmse()
                print(rmse)
                cm = mdl.model_performance().confusion_matrix().as_data_frame()
                cm2 = dict(cm)
                print(cm2)
                grid = mdl.get_params()
                print(grid)
                mse = mdl.model_performance().mse()
                print(mse)
                r = mdl.model_performance().r2()
                print(r)
                Accuracy = mdl.model_performance().auc()
                print(Accuracy)
                importanc = mdl.varimp(use_pandas=True)
                importancs =pd.DataFrame(importanc,columns =['percentage' , 'variable'] )
                print(importancs)
                importances = importancs['percentage']
                model_file_name,model ,dname= self.path ,self.path,self.path
                h2o_DB_upload(Accuracy,X_train,X_test,rmse,mse,r,importances,
                          grid,None,cm2,y,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type
                          ,description ,uom_type)
                
                    
        else:
            cm = None
            if extension == '.pkl':
                joblib_file = (smart_open(self.path))
                joblib_LR_model = joblib.load(joblib_file)
                score = joblib_LR_model.score(X_train,y_train)
                print("Test score: {0:.2f} %".format(100 * score))
                y_pred = joblib_LR_model.predict(X_test)
                errors = abs(y_pred - y_test)
                mape = np.mean(100 * (errors / y_test))
                Accuracy = 100 - mape 
                print("Accuracy",Accuracy)
                print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                estimator, grid = "prediction" ,None
                l = 1
                target = self.i
                model_file_name,model ,dname= self.path ,self.path,self.path
                DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
                                   None,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                             user_defined_terminology,sample_type ,description ,uom_type)
                
            elif extension == '.h5':
                tf_keras_file = (smart_open(self.path))
                with h5py.File(tf_keras_file, 'r') as f:
                    data = f
                    model = tf.keras.models.load_model(data)
                    y_pred = model.predict(X_test)
                    print(y_pred)
                    errors = abs(y_pred - y_test)
                    mape = np.mean(100 * (errors / y_test))
                    Accuracy = 100 - mape 
                    print("Accuracy",Accuracy)
                    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                    estimator, grid = "prediction" ,None
                    l = 1
                    target = self.i
                    model_file_name,model ,dname= self.path ,self.path,self.path
                    DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
                                       None,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                                 user_defined_terminology,sample_type ,description ,uom_type)
            elif extension == '.zip':
                import h2o
                from h2o.automl import H2OAutoML
                h2o.init()
                cols = list(X_train.columns)
                cols.append(self.i)
                selected_cols = self.data[cols]
                x=list(cols)
                y = self.i
                df = h2o.H2OFrame(selected_cols)
                train,  test = df.split_frame(ratios=[.8])
                session = boto3.Session(
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                 )
                s3 = session.resource('s3')
                s3.meta.client.download_file(Filename=model_file, Bucket=bucket_name, Key=model_file)
                mdl = h2o.import_mojo(model_file)
                predictions = mdl.predict(test)
                rmse = mdl.model_performance().rmse()
                print(rmse)
                mse = mdl.model_performance().mse()
                print(mse)
                r = mdl.model_performance().rmsle()
                print(r)
                grid = mdl.get_params()
                print(grid)
                Accuracy = mdl.model_performance().mae()
                print(Accuracy)
                importanc = mdl.varimp(use_pandas=True)
                importancs =pd.DataFrame(importanc,columns =['percentage' , 'variable'] )
                print(importancs)
                importances = importancs['percentage']
                model_file_name,model ,dname= self.path ,self.path,self.path
                h2o_DB_upload(Accuracy,X_train,X_test,rmse,mse,r,importances,
                          None,None,cm,y,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type
                          ,description ,uom_type)
    print('prediction_Results_Loaded')
