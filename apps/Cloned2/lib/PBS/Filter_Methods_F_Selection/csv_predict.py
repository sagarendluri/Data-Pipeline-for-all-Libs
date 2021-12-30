import pandas as pd
from smart_open import smart_open
from apps.Cloned2.lib.PBS.datacleaning import DataCleaning
from apps.Cloned2.lib.PBS.Feature_Selection_Model.RFE_Regression import ALL_RGN_Modeling
from apps.Cloned2.lib.PBS.Feature_Selection_Model.PCA import PCA_Unsupervised
from apps.Cloned2.lib.PBS.Filter_Methods_F_Selection.chi2_test_for_Categorical import x2
class csv_predict():
    def __init__(self, path, pre_D, i, ini, algos, sklearn, ai, dname, predict,model_file,config_object_user,user_defined_terminology,sample_type,description,uom_type,all_M,d_col,cut_off,aws_access_key_id,aws_secret_access_key,bucket_name,analysis_id,min_depth,max_depth,min_samples_split,n_estimators_start,n_estimators_stop,n_neighbors,xgb_objective,xgb_learning_rate,xgb_max_depth,xgb_min_child_weight,svm_c,svm_gamma,svm_kernel,default,db_name,N_features):
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
        self.analysis_id=analysis_id
        self.min_depth =min_depth
        self.max_depth =max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators_start = n_estimators_start
        self.n_estimators_stop = n_estimators_stop
        self.n_neighbors = n_neighbors
        self.xgb_objective =xgb_objective
        self.xgb_learning_rate =xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_min_child_weight =xgb_min_child_weight
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.svm_kernel = svm_kernel
        self.default = default
        self.db_name = db_name
        self.N_features = N_features
    def csv_Predicts(self):
        # try:
            df = pd.read_csv(smart_open(self.path))
            def drop(df):
                if self.d_col==None:
                    return df
                else:
                    df = df.drop(self.d_col,axis=1)
                    return df
            df1 = drop(df)
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
            analysis_id =self.analysis_id
            min_depth = self.min_depth
            max_depth = self.max_depth
            min_samples_split = self.min_samples_split
            n_estimators_start = self.n_estimators_start
            n_estimators_stop = self.n_estimators_stop
            n_neighbors = self.n_neighbors
            xgb_objective =  self.xgb_objective
            xgb_learning_rate = self.xgb_learning_rate
            xgb_max_depth = self.xgb_max_depth
            xgb_min_child_weight = self.xgb_min_child_weight
            svm_c = self.svm_c
            svm_gamma = self.svm_gamma
            svm_kernel = self.svm_kernel
            default = self.default
            db_name = self.db_name
            N_features = self.N_features
            model_instance = DataCleaning(df1, i)
            model_instance.dtypes_handliing()
            model_instance.handiling_categorical()
            model_instance.handiling_int_col()
            cleaned_Data_frm = model_instance.concat_cat()
            cleaned_Data_frm1 = model_instance.concat_int()
            y, label_types = model_instance.encoder()
            if "X2"==self.X2:
                if y.dtypes == 'int64' or y.dtypes =='int32':
                    data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1, y)
                    models = X2(data,y)
                    # models.parameter_tuning()
                    # models.Building_Models()
                else:
                    data = model_instance.QC(cleaned_Data_frm, cleaned_Data_frm1, y)
                    models = X2(data,y)
                    # models.parameter_tuning()
                    # models.Building_Models_Reg()