import matplotlib.pyplot as plt
from configparser import ConfigParser
import pandas as pd
import json
import boto3
from sklearn import metrics
import numpy as np
import joblib
from apps.Cloned2.lib.PBS.Upload_Data_Database.DB_details import DB_Credentials


def DB_upload(train_accuracy, test_accuracy, X_train, X_test, y_test, y_pred, importances, grid, estimator, l, cm,
              target, model_file_name,
              model, dname, config_object_user,
              user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name, result, dataset_type,
              df,mbe):
    # try:
    DB = DB_Credentials(config_object_user, db_name)
    cursor, conn = DB.DB_details()
    config_object = ConfigParser()
    ini = config_object.read(r'/dse/apps/Cloned2/config.ini')
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
    columnsNameslist = list(X_train.columns)
    for col_index in range(0, len(list(X_train.columns))):
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
    gt = y_test
    esti = {}

    def accuracy(test_accuracy):
        if "-inf" == str(test_accuracy):
            Accuracy = " "
            return Accuracy
        else:
            print("int", test_accuracy)
            return test_accuracy

    test_accuracy = accuracy(test_accuracy)
    performance_matrix['Mean Absolute Error (MAE)'] = metrics.mean_absolute_error(gt, y_pred)
    performance_matrix['Mean Squared[Variance] Error (MSE)'] = metrics.mean_squared_error(gt, y_pred)
    performance_matrix['Root Mean Squared Error (RMSE)'] = np.sqrt(metrics.mean_squared_error(gt, y_pred))
    performance_matrix['Mean Bias Error (MBE)'] = mbe
    cr = {"performance": {"Algorithm": model,
                          "performance_matrix": performance_matrix,
                          },
          "All_Metrics": [{"Accuracy": test_accuracy,
                           "Train_Accuracy": train_accuracy,
                           "confusion matrix": cm,
                           "grid": grid,
                           "Estimator": esti,
                           "request_payload_data": request_payload_data,
                           "dropped_correlated_features": result,
                           "dataset_name": dname}],
          "Predictions": {"Test_data": df.to_dict('list')}}
    s3_client = boto3.client('s3', aws_access_key_id=access_key_id,
                             aws_secret_access_key=secret_access_key)
    s3_client.upload_file(model_file_name, bucket_name, model_file_name)
    insert_query = """INSERT INTO Phenotype(
                                        user_defined_terminology, sample_type, phenotype,
                                        description, uom_type)
                                        VALUES ( %s, %s, %s, %s, %s) """

    mySql_insert_query = """INSERT INTO Phenotype_model_info(model_file_path,
                                            confusion_matrix, hyperparameter_grid, best_estimator,
                                            dataset_name, request_payload,performance,Phenotypes_id_id ,analysis_name,dataset_type)
                                            VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s,%s) """
    request_payload_data = request_payload_data
    cursor.execute("SELECT id,user_defined_terminology FROM Phenotype")
    c = cursor.fetchall()
    lst = list(c)
    res = [item for tp in lst for item in tp]
    if str(user_defined_terminology) in res:
        id_ = res.index(str(user_defined_terminology))
        d = id_ - 1
        recordTuple = (model_file_name, json.dumps(cm),
                       json.dumps(grid),
                       json.dumps(esti),
                       dname, json.dumps(request_payload_data),
                       json.dumps(cr), res[d], analysis_id, dataset_type
                       )
        cursor.execute(mySql_insert_query, recordTuple)
        conn.commit()
    else:
        cursor.execute(insert_query, (str(user_defined_terminology)
                                      , str(sample_type), str(target), str(description), str(uom_type)))
        conn.commit()
        phe_id_id = cursor.lastrowid
        recordTuple = (model_file_name, json.dumps(cm),
                       json.dumps(grid),
                       json.dumps(esti),
                       dname, json.dumps(request_payload_data),
                       json.dumps(cr), phe_id_id, analysis_id, dataset_type
                       )
        cursor.execute(mySql_insert_query, recordTuple)
        conn.commit()

# except Exception as Ex:
#     print("Fill_DB exited with the error : ")  # $%" % (Ex))
#     print(Ex)
