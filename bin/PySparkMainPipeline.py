try:
    import argparse
    import os
    import boto3
    from configparser import ConfigParser
    from smart_open import smart_open 
    from os.path import splitext
    import sys
    from VCFto_csv import DC_PD
    from ModuleSelector import modularization
    print("successfully installed all modules PySparkMainPipeline bin file")
except:
    print("Not successfully installed all modules in PySparkMainPipeline bin file")
parser = argparse.ArgumentParser(
    description='''PySparkMainpipeline.py Description. This is all about complete AI and you just see which arguments you want to run as an input string. ''',
    epilog="""All is well that ends well."""
)
parser.add_argument("--target", help="enter target feature", type=str)
parser.add_argument("--dataset", help="enter dataset name", type=str)
parser.add_argument("--phenome_data", help="enter dataset name", type=str)
parser.add_argument("--DTClassifier", help="enter a model name", type=str)
parser.add_argument("--NBClassifier", help="enter a model name", type=str)
parser.add_argument("--RFClassifier", help="enter a model name", type=str)
parser.add_argument("--MLPClassifier", help="enter a model name", type=str)
parser.add_argument("--ALL_Modules", help="entered a all module", type=str)
parser.add_argument("--config_object_user", help="Config Object or the user primary name", type=str, required=True)
parser.add_argument("--drop_columns", help="drop unwanted columns", type=str,nargs='+')
parser.add_argument("--index", help="set a phenotype column as index ,it's must be the same in vcf smaple column", type=str,required=True)
parser.add_argument("--to_csv_name", help="what name you want to write with genome and phenome file", type=str,required=True)
parser.add_argument("--Classification_or_Regression", help="what name you want to write with genome and phenome file", type=str,required=True)
parser.add_argument("--user_defined_terminology", help="user_defined_terminology", type=str )
parser.add_argument("--sample_type", help="sample_type", type=str)
parser.add_argument("--description", help="description", type=str)
parser.add_argument("--uom_type", help="uom_type", type=str)
args = parser.parse_args()
i = args.target
dname = args.dataset
pname = args.phenome_data
DTC = args.DTClassifier
NBC = args.NBClassifier
RFC = args.RFClassifier
MLPC = args.MLPClassifier
ALL = args.ALL_Modules
config_object_user = args.config_object_user
d_col = args.drop_columns
index = args.index
to_csv_name = args.to_csv_name
user_defined_terminology = args.user_defined_terminology
sample_type = args.sample_type
description = args.description
uom_type = args.uom_type
Classification_or_Regression = args.Classification_or_Regression
config_object = ConfigParser()
ini = config_object.read('config.ini')
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
                         user_defined_terminology,sample_type ,description ,uom_type)
    pipeL.all_modules()

