import pandas as pd
from smart_open import smart_open
from Classification_Modles import CLF_modeling
from datacleaning import DataCleaning
from prediction import DataPrediction
from Regression_Models import RGN_Modeling
from ALL_Classification_Models import sk_Models
from ALL_Regression_models import ALL_RGN_Modeling
from lazypredict_Regression import lazypredict_Regression
from lazypredict_classifier import lazypredict_classifier
class csv_predicts():
    def __init__(self, path, pre_D, i, ini, algos, sklearn, ai, dname, predict,model_file,config_object_user,
                user_defined_terminology,sample_type ,description ,uom_type,all_M,d_col,cut_off,aws_access_key_id,aws_secret_access_key,bucket_name,lazypredict ):
        self.path = path
        self.pre_D = pre_D
        self.i = i
        self.ini = ini
        self.algos = algos
        self.sklearn = sklearn
        self.ai = ai
        self.dname = dname
        self.predict = predict
        self.model_file = model_file
        self.config_object_user=config_object_user
        self.user_defined_terminology =user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type =uom_type
        self.all_M = all_M
        self.d_col= d_col
        self.cut_off = cut_off
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key=aws_secret_access_key
        self.bucket_name =bucket_name
        self.lazypredict=lazypredict
    def csv_Predicts(self):
        print('file formate csv')
        df = pd.read_csv(smart_open(self.path))
        def drop(df):
            try:
                if self.d_col==None:
                    return df
                else:
                    df = df.drop(self.d_col,axis=1)
                    return df
            except:
                print("Give valid columns which columns dataset has")
        df1 = drop(df)
#         df1 = df2.replace({'0':np.nan, 0:np.nan,'-2':np.nan, -2:np.nan,'-1':np.nan, -1:np.nan})
#         df1 = df1.dropna(thresh=df1.shape[0]*0.4,how='all',axis=1)
#         for col in df1.select_dtypes(include=np.number):
#             df1[col] = df1[col].fillna(df1[col].median())
        i = self.i
        ini = self.ini
        dname = self.dname
        algos = self.algos
        sklearn = self.sklearn
        ai =self.ai
        pre_D = self.pre_D
        model_file = self.model_file
        cut_off= self.cut_off
        config_object_user =self.config_object_user
        user_defined_terminology =self.user_defined_terminology
        sample_type = self.sample_type
        description = self.description
        uom_type =self.uom_type
        aws_access_key_id = self.aws_access_key_id
        aws_secret_access_key= self.aws_secret_access_key
        bucket_name = self.bucket_name
        model_instance = DataCleaning(df1, i)
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
                data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1, y)
                models = ALL_RGN_Modeling(data, y, i, label_types, dname,config_object_user,
                                             user_defined_terminology,sample_type ,description ,uom_type,cut_off)
                models.Building_Models_Reg()
        
        elif "lazypredict" ==self.lazypredict:
            if y.dtypes == 'int64' or y.dtypes =='int32':
                data = model_instance.QC(cleaned_Data_frm, cleaned_Data_frm1, y)
                models = lazypredict_classifier(data, y,i, dname,config_object_user,
                                     user_defined_terminology,sample_type ,description ,uom_type,cut_off)
            else:
                data = model_instance.QC(cleaned_Data_frm, cleaned_Data_frm1, y)
                models = lazypredict_Regression(data, y,i, dname,config_object_user,
                                     user_defined_terminology,sample_type ,description ,uom_type,cut_off)
            
            
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

