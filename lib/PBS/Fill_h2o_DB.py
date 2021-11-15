import matplotlib.pyplot as plt
from configparser import ConfigParser
import pandas as pd
import json
import boto3
from sklearn import metrics
import numpy as np
import joblib
from DB_details import DB_Credentials
def h2o_DB_upload(Accuracy,X_train,X_test,rmse,mse,rmsle,importances,grid,estimator,cm,target,model_file_name,model,dname,
                  config_object_user,user_defined_terminology,sample_type ,description ,uom_type):
#     try:
        DB = DB_Credentials(config_object_user)
        cursor ,conn = DB.DB_details()
        config_object = ConfigParser()
        ini = config_object.read(r'config.ini')
        config_object.read(ini)
        config_object_project = ConfigParser()
        userinfo = config_object[config_object_user]
        access_key_id = userinfo["access_key_id"]
        secret_access_key = userinfo["secret_access_key"]
        bucket_name = userinfo["bucket"]
        x_train_means = X_train.mean(axis=0)
        request_payload_data = {}
        data = X_train[:].values.T
        data1 = pd.DataFrame(data)
        print("importance",importances)
        columnsNameslist = list(X_train.columns)
        for col_index in range(0,len(list(X_train.columns))):
            data = data1.values[col_index]
            data2 = (data - data.min()) / (data.max() - data.min())
            hist = plt.hist(data2)
            request_payload_data[columnsNameslist[col_index]] = {
                'mean': x_train_means[col_index],
                'importance': importances[col_index],
                'normalised_data_distribution': list(hist[0])
            }
        performance_matrix = {
                'Samples in training set': len(X_train),
                'Samples in test set': len(X_test),
            }
        esti ={}
        performance_matrix['Mean Squared Error (MSE)'] = mse
        performance_matrix['Root Mean Squared Error (RMSE)'] = rmse
        performance_matrix['RMSLE/r^2'] = rmsle
        print("MAE/Accuracy",Accuracy)
        cr = {"performance":{"Algorithm": model,
                              "performance_matrix": performance_matrix,
                              },
              "All_Metrics":[{"MAE/Accuracy":Accuracy,
                             "confusion matrix": cm,
                             "grid": grid,
                             "Estimator": esti,
                             "model_file_name": model_file_name,
                             "request_payload_data": request_payload_data,
                             "dataset_name":dname}]}
        
        insert_query = """INSERT INTO Phenotype(
                                        user_defined_terminology, sample_type, phenotype,
                                        description, uom_type)
                                        VALUES ( %s, %s, %s, %s, %s) """

        cursor.execute(insert_query ,(str(user_defined_terminology)
                                          ,str(sample_type),str(target),str(description),str(uom_type)))
        conn.commit()
        phe_id_id = cursor.lastrowid

        print(cursor.lastrowid,"latest_record")
        mySql_insert_query = """INSERT INTO Phenotype_model_info(model_file_path,
                                            confusion_matrix, hyperparameter_grid, best_estimator,
                                            dataset_name, request_payload,performance,Phenotypes_id_id)
                                            VALUES (%s, %s, %s, %s, %s, %s, %s,%s) """
        request_payload_data= request_payload_data
       # Upload the file
        s3_client = boto3.client('s3', aws_access_key_id=access_key_id,
                                 aws_secret_access_key=secret_access_key)
        s3_client.upload_file(model_file_name,bucket_name , model_file_name)
        print('Model_loaded_in_s3_sucsess')
        recordTuple = (
                        model, json.dumps(cm),
                        json.dumps(grid),
                        json.dumps(esti),
                        dname, json.dumps(request_payload_data),
                        json.dumps(cr),phe_id_id
        )
        cursor.execute(mySql_insert_query, recordTuple)
        conn.commit()
#     except:
#         print("pleas give phenotype table details ")

