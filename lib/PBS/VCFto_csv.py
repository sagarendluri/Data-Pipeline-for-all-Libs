try:
    import allel
    import pandas as pd
    import numpy as np
    import os
    from smart_open import smart_open
    from io import StringIO
    import boto3
    from ModuleSelector import modularization
    print("successfully installed all modules VCFto_csv lib file")
except:
    print("Not successfully installed all modules in VCFto_csv lib file")
path = os.path.join(os.path.expanduser('~'), 'dse_sagar', 'Data' )
class DC_PD():
    def __init__(self, path, pheN, i,DTC, NBC, RFC, MLPC,ALL,config_object_user,aws_key,aws_secret,bucket,d_col,index,to_csv_name):
        self.path = path
        self.pheN = pheN
        self.config_object_user =config_object_user
        self.i = i
        self.DTC = DTC
        self.NBC = NBC
        self.RFC = RFC
        self.MLPC = MLPC
        self.ALL = ALL
        self.aws_key=aws_key
        self.aws_secret=aws_secret
        self.bucket=bucket
        self.d_col = d_col
        self.index =index
        self.to_csv_name = to_csv_name
    def total_QC(self):
        try:
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
            i = self.i
            DTC = self.DTC
            NBC = self.NBC
            RFC = self.RFC
            MLPC = self.MLPC
            ALL = self.ALL
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
            pipeL = modularization(to_csv_name, i, DTC, NBC, RFC, MLPC,ALL)
            pipeL.all_modules()
            return print("vcf_converted_into_csv")
        except:
            print("Failed to convert vcf to csv")
            
            
            
            
            
            