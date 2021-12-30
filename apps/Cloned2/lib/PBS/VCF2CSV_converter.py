import allel
import numpy as np
import pandas as pd
from smart_open import smart_open
from apps.Cloned2.lib.PBS.datacleaning import DataCleaning
from apps.Cloned2.lib.PBS.prediction import DataPrediction
from apps.Cloned2.lib.PBS.Classification_Modles import CLF_modeling
from apps.Cloned2.lib.PBS.Regression_Models import RGN_Modeling
from apps.Cloned2.lib.PBS.ALL_Classification_Models import sk_Models
from apps.Cloned2.lib.PBS.ALL_Regression_models import ALL_RGN_Modeling
import boto3
class DC_PD():
    def __init__(self, path, pheN, i,config_object_user,aws_key,aws_secret,bucket,d_col,index,to_csv_name,predict,all_M ,algos, sklearn, ai,user_defined_terminology,sample_type ,description ,uom_type,cut_off):
        self.path = path
        self.pheN = pheN
        self.i = i
        self.config_object_user = config_object_user
        self.aws_key=aws_key
        self.aws_secret=aws_secret
        self.bucket=bucket
        self.d_col = d_col
        self.index =index
        self.to_csv_name = to_csv_name
        self.predict = predict
        self.all_M = all_M
        self.algos = algos
        self.sklearn = sklearn
        self.ai = ai
        self.user_defined_terminology =user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type =uom_type
        self.cut_off = cut_off
    def total_QC(self):
#         try:
            to_csv_name=self.to_csv_name
            config_object_user = self.config_object_user
            algos = self.algos
            sklearn = self.sklearn
            ai = self.ai
            cut_off= self.cut_off
            user_defined_terminology =self.user_defined_terminology
            sample_type = self.sample_type
            description = self.description
            uom_type =self.uom_type
            d_col= self.d_col
            print("file type is VCF")
            callset = allel.read_vcf(smart_open(self.path))
            snps = callset['variants/ID']
            da = callset['calldata/GT']
            data = da.transpose([2, 0, 1]).reshape(-1, da.shape[1])
            df = pd.DataFrame(data)
            a = len(df)
            h = int(a / 2)
            sm = callset['samples']
            def split(df):
                hd = df.head(h)
                tl = df.tail(len(df) - h)
                return hd, tl
            # Split dataframe into top 3 rows (first) and the rest (second)
            heads, tails = split(df)
            df1 = pd.DataFrame(heads)
            df2 = pd.DataFrame(tails)
            df1.columns = sm
            df2.columns = sm
            df1['snps'] = snps
            df2['snps'] = snps
            sum_df = df1.set_index('snps').add(df2.set_index('snps'), fill_value=0).reset_index()
            sum1_df = sum_df.set_index('snps')
            sum_df = sum1_df.T
            sum_df = sum_df.reset_index()
            phe = pd.read_csv(smart_open(self.pheN))
            def drop(phe):
                if self.d_col==None:
                    return phe
                else:
                    phe = phe.drop(self.d_col,axis=1)
                    return phe
            phe = drop(phe)
            phe = phe.fillna(phe.mean())
            new_phe = phe.set_index(self.index)
            sum1 = sum_df.set_index('index')
            df3 = pd.merge(sum1, new_phe, left_index=True, right_index=True)
            final = df3.reset_index()
            final = final.drop('index',axis=1)
            print("writing into csv")
            to_csv_name=self.to_csv_name 
            final.to_csv(to_csv_name)
            config_object_user = self.config_object_user
            bucket=self.bucket
            session = boto3.Session(
                 aws_access_key_id=self.aws_key,
                 aws_secret_access_key=self.aws_secret,
             )
            s3 = session.resource('s3')
            # Filename - File to upload
            # # Bucket - Bucket to upload to (the top level directory under AWS S3)
            # # Key - S3 object name (can contain subdirectories). If not specified then file_name is used
            s3.meta.client.upload_file(Filename=to_csv_name, Bucket=bucket, Key=to_csv_name)
            i = self.i
            df = pd.read_csv(to_csv_name)
            print(df)
            model_instance = DataCleaning(df, i)
            model_instance.dtypes_handliing()
            print(model_instance.handiling_categorical())
            model_instance.handiling_int_col()
            cleaned_Data_frm = model_instance.concat_cat()
            cleaned_Data_frm1 = model_instance.concat_int()
            y, label_types = model_instance.encoder()
            if "predict" == self.predict:
                data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1, y)
                model_instance = DataPrediction(data, i, pre_D,model_file,user_defined_terminology,sample_type ,description ,uom_type,cut_off,config_object_user ,aws_access_key_id,aws_secret_access_key,bucket_name )
                resutls = model_instance.split()
            elif "Run_all_models"==self.all_M:
                if y.dtypes == 'int64' or y.dtypes =='int32':
                    data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1, y)
                    models = sk_Models(data, y, i, label_types, dname,config_object_user,
                                         user_defined_terminology,sample_type ,description ,uom_type,cut_off)
                    models.Building_Models()
                else:
                    data = model_instance.QC(cleaned_Data_frm, cleaned_Data_frm1, y)
                    models = ALL_RGN_Modeling(data, y, i, label_types, dname,config_object_user,
                                                 user_defined_terminology,sample_type ,description ,uom_type,cut_off)
                    models.Building_Models_Reg()
            elif y.dtypes == 'int64' or y.dtypes =='int32':
                data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1, y)
                models = CLF_modeling(data, y, i, label_types, algos, sklearn, ai, dname,config_object_user,
                                     user_defined_terminology,sample_type ,description ,uom_type,cut_off)
                models.Classification()
            else:
                data = model_instance.QC(cleaned_Data_frm, cleaned_Data_frm1, y)
                models = RGN_Modeling(data, y, i, label_types, algos, sklearn, ai, dname,config_object_user,
                                     user_defined_terminology,sample_type ,description ,uom_type,cut_off)
                models.Regression()
                return print("models built")

            return print("vcf_converted_into_csv")
#         except:
#             print("Failed to convert vcf to csv")
