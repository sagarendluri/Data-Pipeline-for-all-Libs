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
# from apps.Cloned2.lib.PBS.DB import
from apps.Cloned2.lib.PBS.QC import train_test
from os.path import splitext
# import tensorflow as tf
import h5py
from apps.Cloned2.lib.PBS.Fill_DB import DB_upload
import numpy as np
import boto3
# from Fill_h2o_DB import h2o_DB_upload
class DataPrediction():
    def __init__(self,data,i,path,model_file,user_defined_terminology,sample_type ,description ,uom_type,cut_off,config_object_user,
            aws_access_key_id ,aws_secret_access_key ,bucket_name,analysis_id,sample_col):#,model_file):
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
        self.analysis_id = analysis_id
        self.sample_col = sample_col
        #self.model_file = model_file
    def split(self):
       # model_file=self.model_file
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
        #print(self.data)
        num =  len(self.data)//2
        X_train =self.data[:num]
        X_test = self.data[num:]
        #print(X_test)
        #print(X_train)
        #X = self.data.drop([self.i],axis=1)
        y = self.data[self.i]
        #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 42)
       # X_train, X_test = train_test(X_train, X_test,cut_off)
        #X_train["y"] =y
        print(self.data.columns)
        model_file = self.model_file
        analysis_id = self.analysis_id
        print("splited",model_file)
        file_name, extension = splitext(self.model_file)
        if y.dtypes == 'float' or y.dtypes =='float':
            if extension == '.pkl':
                print("joblib")
                joblib_file = (smart_open(self.path))
                joblib_LR_model = joblib.load(joblib_file)
                #score = joblib_LR_model.score(X_train,y_train)
                #print("Test score: {0:.2f} %".format(100 * score))
                y_pred = joblib_LR_model.predict(self.data)
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
                estimator, grid = "prediction" ,"None"
                l = 1
                target = self.i
                model_file_name,model ,dname= self.path ,self.path,self.path
                DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
                                   None,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                             user_defined_terminology,sample_type ,description ,uom_type,analysis_id)
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
                                 user_defined_terminology,sample_type ,description ,uom_type,analysis_id)
            elif extension == '.zip':
                # import h2o
                # from h2o.automl import H2OAutoML
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
                          ,description ,uom_type,analysis_id)


        else:
            cm ="None"
            if extension == '.pkl':
                print("regression")
                a =["X","K"]
                for i in a:
                    if i.startswith("X") or i.startswith("K"):
                        joblib_file = (smart_open(self.path))
                        joblib_LR_model = joblib.load(joblib_file)
                        #f_names = joblib_LR_model.feature_names
                        X_train_pred =joblib_LR_model.predict(X_train[['Year ', 'Hybrid CODE ', 'Alt Plot#', 'DAYP', 'DAYM', 'EAHT', 'PLHT', 'STND']])
                        X_test_pred = joblib_LR_model.predict(X_test[['Year ', 'Hybrid CODE ', 'Alt Plot#', 'DAYP', 'DAYM', 'EAHT', 'PLHT', 'STND']])
                        X_train_pred = list(X_train_pred)
                        X_test_pred = list(X_test_pred)
                        X_train_pred.extend(X_test_pred)
                        df = pd.DataFrame({"Prediction":X_train_pred})
                        df.to_csv("Prediction_file.csv")
                    else:
                        joblib_file = (smart_open(self.path))
                        joblib_LR_model = joblib.load(joblib_file)
                        #score = joblib_LR_model.score(X_train,y_train)
                        #print("Test score: {0:.2f} %".format(100 * score))
                        X_train_pred = joblib_LR_model.predict(self.data[['Year ', 'Hybrid CODE', 'Alt Plot#', 'DAYP', 'DAYM', 'EAHT', 'PLHT', 'STND']])
                        X_test_pred = joblib_LR_model.predict(self.data[['Year ', 'Hybrid CODE ', 'Alt Plot#', 'DAYP', 'DAYM', 'EAHT', 'PLHT', 'STND']])
                        #print(pred)
                        X_train_pred = list(X_train_pred)
                        #X_test_pred = list(X_test_pred)
                        print(X_test_pred)
                        #X_train_pred.extend(X_test_pred)
                        df = pd.DataFrame({"prediction":X_train_pred})
                       # df.to_csv("Prediction_file.csv")

                #errors = abs(y_pred - y_test)
                #mape = np.mean(100 * (errors / y_test))
                #Accuracy = 100 - mape
                #print("Accuracy",Accuracy)
                #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                #print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                #estimator, grid = "prediction" ,None
                #l = 1
                #target = self.i
                #model_file_name,model ,dname= self.path ,self.path,self.path
                #DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
                 #                  None,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                  #           user_defined_terminology,sample_type ,description ,uom_type,analysis_id)

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
                                 user_defined_terminology,sample_type ,description ,uom_type,analysis_id)
            elif extension == '.zip':
                print("h2o")
                import h2o
                #from h2o.automl import H2OAutoML
                h2o.init()
                cols = list(X_train.columns)
               # cols.append(self.i)
                selected_cols = self.data[cols]
                x=list(cols)
                #y = self.i
                #print("models",X_train.columns)
                X_train_pred = h2o.H2OFrame(X_train[['Year ', 'Hybrid CODE ', 'Alt Plot#', 'DAYP', 'DAYM', 'EAHT', 'PLHT', 'STND']])
                X_test_pred = h2o.H2OFrame(X_test[['Year ', 'Hybrid CODE ', 'Alt Plot#', 'DAYP', 'DAYM', 'EAHT', 'PLHT', 'STND']])
                session = boto3.Session(
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                 )
                s3 = session.resource('s3')
                s3.meta.client.download_file(Filename=model_file, Bucket=bucket_name, Key=model_file)
                mdl = h2o.import_mojo(model_file)
                X_train_pred = mdl.predict(X_train_pred).as_data_frame()
                X_test_pred = mdl.predict(X_test_pred).as_data_frame()
                X_train_pred = list(X_train_pred.values)
                print(X_train_pred)

                