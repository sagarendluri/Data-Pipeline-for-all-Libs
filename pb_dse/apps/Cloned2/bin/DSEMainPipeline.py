# try:
from django.http.response import StreamingHttpResponse
from configparser import ConfigParser
from os.path import splitext
import pandas as pd
from smart_open import smart_open
import sys
# from apps.Cloned2.lib.PBS.VCF2CSV_converter import DC_PD
from apps.Cloned2.lib.PBS.Dimensionality_reduction_F_Selection.PCA_F_Selection import PCA
from apps.Cloned2.lib.PBS.Run_Single_Model.Run_single_Class_Reg_Selection import Run_single_Class_Reg_Model
from apps.Cloned2.lib.PBS.Run_All_Models.Run_all_Class_Reg_Selection import Run_R_and_C_Models
from apps.Cloned2.lib.PBS.Wrapper_Methods_F_Selection.RFECV_Selection import RFECV_Models
#     print("successfully installed all modules DSEMainPipeline bin file")
# except Exception as E:
#       print("Pipeline exited with the error : ")#$%" % (Ex))
#       print(E)
        # print("Not successfully installed all modules in DSEMainPipeline bin file")
def Decision_M(dname,i,sklearn,algorithm,
                config_object_user,ai ,model_file,predict,phenome_data,user_defined_terminology,sample_type,description,uom_type,all_M,d_col,index,to_csv_name,min_depth,max_depth,min_samples_split,n_estimators_start,n_estimators_stop,n_neighbors,xgb_objective,xgb_learning_rate,
                xgb_max_depth,xgb_min_child_weight,svm_C,svm_gamma,svm_kernel,default,analysis_id,db_name,N_features,pca,
                learning_rate_init ,max_iter, hidden_layer_sizes, activation,alpha,early_stopping,cut_off = 0.8):
#     try:
        i = i
        dname = dname
        model_file = model_file
        predict = predict
        phenome = phenome_data
        algos = algorithm
        sklearn = sklearn
        config_object_user = config_object_user
        ai = ai
        user_defined_terminology =user_defined_terminology
        sample_type = sample_type
        description = description
        uom_type = uom_type
        all_M = all_M
        d_col = d_col
        index = index
        to_csv_name = to_csv_name
        min_depth =min_depth
        max_depth = max_depth
        min_samples_split = min_samples_split
        n_estimators_start = n_estimators_start
        n_estimators_stop = n_estimators_stop
        n_neighbors = n_neighbors
        xgb_objective = xgb_objective
        xgb_learning_rate = xgb_learning_rate
        xgb_max_depth = xgb_max_depth
        xgb_min_child_weight = xgb_min_child_weight
        svm_C = svm_C
        svm_gamma = svm_gamma
        svm_kernel = svm_kernel
        default = default
        analysis_id=analysis_id
        db_name =db_name
        N_features  =N_features
        pca= pca
        learning_rate_init = learning_rate_init
        max_iter =max_iter
        hidden_layer_sizes = hidden_layer_sizes
        activation =activation
        alpha = alpha
        early_stopping =early_stopping
        cut_off = cut_off
        config_object = ConfigParser()
        ini = config_object.read(r'/dse/apps/Cloned2/config.ini')#C:\Users\sagar\dse_sagar\config.ini')#)
        config_object.read(ini)
        config_object_project = ConfigParser()
        userinfo = config_object[config_object_user]
        access_key_id = userinfo["access_key_id"]
        secret_access_key = userinfo["secret_access_key"]
        bucket_name = userinfo["bucket"]
        file_name, extension = splitext(dname)
        path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, dname)
        pre_D ='s3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, model_file)
        pheN = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, phenome)
        if "RFECV" == N_features():
            results = RFECV_Models(path,i,ini,dname,config_object_user,
                                user_defined_terminology,sample_type ,description ,uom_type,d_col,cut_off,access_key_id,
                                secret_access_key, bucket_name,analysis_id,min_depth,max_depth,min_samples_split,
                                n_estimators_start,n_estimators_stop,n_neighbors,xgb_objective,xgb_learning_rate,
                                xgb_max_depth,xgb_min_child_weight,svm_C,svm_gamma,svm_kernel,default,db_name,N_features,learning_rate_init ,max_iter, hidden_layer_sizes, activation,alpha,early_stopping)
            results.csv_Predicts()

        elif "PCA"==pca():
            results = PCA(path, i, ini, dname,config_object_user,
                                user_defined_terminology,sample_type ,description ,uom_type,d_col,cut_off,access_key_id,secret_access_key, bucket_name,analysis_id,min_depth,max_depth,min_samples_split,n_estimators_start,n_estimators_stop,n_neighbors,xgb_objective,xgb_learning_rate,xgb_max_depth,xgb_min_child_weight,svm_C,svm_gamma,svm_kernel,default,db_name,learning_rate_init ,max_iter, hidden_layer_sizes, activation,alpha,early_stopping)
            results.csv_Predicts()
        elif "Run_all_Models"==all_M:
            results = Run_R_and_C_Models(path,i, ini, dname, config_object_user,
                                user_defined_terminology,sample_type ,description ,uom_type,all_M,d_col,cut_off,access_key_id,secret_access_key, bucket_name,analysis_id,min_depth,max_depth,min_samples_split,n_estimators_start,n_estimators_stop,n_neighbors,xgb_objective,xgb_learning_rate,xgb_max_depth,xgb_min_child_weight,svm_C,svm_gamma,svm_kernel,default,db_name,learning_rate_init ,max_iter, hidden_layer_sizes, activation,alpha,early_stopping)
            results.csv_Predicts()
        else:
            results = Run_single_Class_Reg_Model(path,  i, ini, algos, sklearn, ai, dname, config_object_user,
                                user_defined_terminology,sample_type ,description ,uom_type,d_col,cut_off,access_key_id,secret_access_key, bucket_name,analysis_id,min_depth,max_depth,min_samples_split,n_estimators_start,n_estimators_stop,n_neighbors,xgb_objective,xgb_learning_rate,xgb_max_depth,xgb_min_child_weight,svm_C,svm_gamma,svm_kernel,default,db_name)
            results.csv_Predicts()
#     except Exception as Ex:
#       print("Pipeline exited with the error : ")
#       print(Ex)

#         elif extension == '.vcf':
#             model_instance = DC_PD(path, pheN, i,config_object_user,access_key_id,secret_access_key,bucket_name,d_col,index,to_csv_name,predict,all_M ,algos, sklearn, ai,user_defined_terminology,sample_type ,description ,uom_type,cut_off)
#             model_instance.total_QC()


