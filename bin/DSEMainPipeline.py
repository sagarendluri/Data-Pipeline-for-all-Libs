# try:
import argparse
import pytest
import unittest
from configparser import ConfigParser
from os.path import splitext
import pandas as pd
from smart_open import smart_open
import sys
from VCF2CSV_converter import DC_PD
from datacleaning import DataCleaning
from prediction import DataPrediction
from csv_predicts import csv_predicts

# print("successfully installed all modules DSEMainPipeline bin file")
# except:
# print("Not successfully installed all modules in DSEMainPipeline bin file")

parser = argparse.ArgumentParser(
    description='''DSEMainpipeline.py Description. This is all about complete AI and you just see which arguments you want to run as an input string. ''',
    epilog="""All is well that ends well.""")
parser.add_argument("--target", help="Enter target feature ", type=str, required=True)
parser.add_argument("--dataset", help="Enter dataset name", type=str, required=True)
parser.add_argument("--predict", help="Enter dataset name", type=str)
parser.add_argument("--algorithm", help="Algorithm name", type=int)
parser.add_argument("--sklearn", help="sklearn", type=str)
parser.add_argument("--model_file", help="Enter dataset name", type=str)
parser.add_argument("--phenome_data", help="Enter dataset name", type=str)
parser.add_argument("--config_object_user", help="Config Object or the user primary name", type=str, required=True)
parser.add_argument("--ai", help="h2o", type=str)
parser.add_argument("--user_defined_terminology", help="user_defined_terminology", type=str )
parser.add_argument("--sample_type", help="sample_type", type=str)
parser.add_argument("--description", help="description", type=str)
parser.add_argument("--uom_type", help="uom_type", type=str)
parser.add_argument("--Run_all_models", help="Run all sklearn models and DNN", type=str)
# parser.add_argument("--help", help=" Show help on running the pipeline", type=bool)
parser.add_argument("--cut_off", help="enter cut_off correllation value ", type=float ,default=0.8 )
parser.add_argument("--drop_columns", help="drop unwanted columns", type=str,nargs='+')
parser.add_argument("--index", help="set a phenotype column as index ,it's must be the same in vcf smaple column", type=str)
parser.add_argument("--to_csv_name", help="what name you want to write with genome and phenome file", type=str)
parser.add_argument("--lazypredict", help="lazypredict", type=str)


args = parser.parse_args()
#
# if args.help:
# 	print("provide some help ...... ")
# try:
i = args.target
dname = args.dataset
model_file = args.model_file
predict = args.predict
phenome = args.phenome_data
algos = args.algorithm
sklearn = args.sklearn
config_object_user = args.config_object_user
ai = args.ai
user_defined_terminology = args.user_defined_terminology
sample_type = args.sample_type
description = args.description
uom_type = args.uom_type
all_M = args.Run_all_models
d_col = args.drop_columns
cut_off = args.cut_off
index = args.index
to_csv_name = args.to_csv_name
lazypredict = args.lazypredict
config_object = ConfigParser()
ini = config_object.read(r'config.ini')
config_object.read(ini)
config_object_project = ConfigParser()
userinfo = config_object[config_object_user]
access_key_id = userinfo["access_key_id"]
secret_access_key = userinfo["secret_access_key"]
bucket_name = userinfo["bucket"]
file_name, extension = splitext(dname)
path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, dname)
pre_D = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, model_file)
pheN = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, phenome)
if extension == '.csv':
    results = csv_predicts(path, pre_D, i, ini, algos, sklearn, ai, dname, predict,model_file,config_object_user,
                          user_defined_terminology,sample_type ,description ,uom_type,all_M,d_col,cut_off,access_key_id,secret_access_key,
                          bucket_name,lazypredict)
    results.csv_Predicts()
elif extension == '.vcf':
    print('vcf')
    model_instance = DC_PD(path, pheN, i,config_object_user,access_key_id,secret_access_key,bucket_name,d_col,index,to_csv_name,predict,all_M ,algos, sklearn, ai,user_defined_terminology,sample_type ,description ,uom_type,cut_off)
    model_instance.total_QC()
# except Exception as Ex:
# 	print("Pipeline exited with the error : ")#$%" % (Ex))
# 	print(Ex)
