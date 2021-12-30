# try:
import argparse
import os
import boto3
from configparser import ConfigParser
from smart_open import smart_open 
from os.path import splitext
import sys
from apps.Cloned2.lib.PBS.VCFto_csv import DC_PD
from apps.Cloned2.lib.PySpark.ModuleSelector import modularization
#     print("successfully installed all modules PySparkMainPipeline bin file")
# except:
#     print("Not successfully installed all modules in PySparkMainPipeline bin file")
def Decision_M_Pyspark(dname,target,DTClassifier,RFClassifier,config_object_user,model_file,predict,
                       phenome_data,
                       user_defined_terminology,
                       sample_type,description,uom_type,all_M,d_col,index,to_csv_name,Classification_or_Regression,analysis_id,output):
    try:
        dname = dname
        print(dname)
        i = target
        print(i)
        DTC = DTClassifier
        print(DTC)
        RFC = RFClassifier
        config_object_user = config_object_user
        model_file =model_file
        predict =predict
        pname =phenome_data
        user_defined_terminology = user_defined_terminology
        sample_type = sample_type
        description = description
        uom_type = uom_type
        all_M=all_M
        d_col = d_col
        index = index
        to_csv_name = to_csv_name
        Classification_or_Regression = Classification_or_Regression
        output=output
        analysis_id =analysis_id
        NBC = "NBC"
        MLPC = "MLPC"
        ALL = "ALL"
        config_object = ConfigParser()
        ini = config_object.read(r'/dse/apps/Cloned2/config.ini')
        config_object.read(ini)
        userinfo = config_object[config_object_user]
        access_key_id = userinfo["access_key_id"]
        secret_access_key = userinfo["secret_access_key"]
        bucket = userinfo["bucket"]
        bucket_name = bucket
        file_name, extension = splitext(dname)
        path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, dname)
        pheN = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, pname)
        if extension=='.vcf':
            print("vcf")
            model_instance = DC_PD(path, pheN, i, DTC, NBC, RFC, MLPC,ALL,config_object_user,access_key_id,secret_access_key,bucket_name,d_col,index,to_csv_name)
            model_instance.total_QC()
        else:
            session = boto3.Session(
                 aws_access_key_id=access_key_id,
                 aws_secret_access_key=secret_access_key,
             )
            s3 = session.resource('s3')
            s3.meta.client.download_file(Filename=dname, Bucket=bucket, Key=dname)
            pipeL = modularization(dname, i,d_col, Classification_or_Regression,DTC, NBC, RFC, MLPC,ALL,config_object_user,
                         user_defined_terminology,sample_type ,description ,uom_type,analysis_id)
            pipeL.all_modules()
    except Exception as Ex:
    	print("Pipeline exited with the error : ")
    	print(Ex)
