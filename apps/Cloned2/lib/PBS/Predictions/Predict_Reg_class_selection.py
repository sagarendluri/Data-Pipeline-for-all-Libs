import pandas as pd
from smart_open import smart_open
from apps.Cloned2.lib.PBS.Data_preprocessing.datacleaning import DataCleaning
from apps.Cloned2.lib.PBS.Predictions.Pediction_for_Regression import DataPrediction


class Csv_predicts:
    def __init__(self, path, pre_D, i, ini, dname, predict, model_file, config_object_user, user_defined_terminology,
                 sample_type, description, uom_type, d_col, aws_access_key_id, aws_secret_access_key,
                 bucket_name, analysis_id, db_name):
        self.path = path
        self.pre_D = pre_D
        self.i = i
        self.ini = ini
        self.dname = dname
        self.predict = predict
        self.model_file = model_file
        self.config_object_user = config_object_user
        self.user_defined_terminology = user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type = uom_type
        self.d_col = d_col
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
        self.analysis_id = analysis_id
        self.db_name = db_name

    def csv_Predicts(self):
        # try:
        df = pd.read_csv(smart_open(self.path))

        def drop(df):
            if self.d_col == None:
                return df
            else:
                df = df.drop(self.d_col, axis=1)
                return df

        df1 = drop(df)
        i = self.i
        ini = self.ini
        dname = self.dname
        pre_D = self.pre_D
        path = self.path
        model_file = self.model_file
        config_object_user = self.config_object_user
        user_defined_terminology = self.user_defined_terminology
        sample_type = self.sample_type
        description = self.description
        uom_type = self.uom_type
        aws_access_key_id = self.aws_access_key_id
        aws_secret_access_key = self.aws_secret_access_key
        bucket_name = self.bucket_name
        analysis_id = self.analysis_id
        db_name = self.db_name
        model_instance = DataCleaning(df1, i)
        model_instance.dtypes_handliing()
        model_instance.handiling_categorical()
        model_instance.handiling_int_col()
        cleaned_Data_frm = model_instance.concat_cat()
        cleaned_Data_frm1 = model_instance.concat_int()
        y, label_types = model_instance.encoder()
        if y.dtypes == 'float' or y.dtypes == 'float':
            data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1, y)
            model_instance = DataPrediction(data, i, path, model_file, user_defined_terminology, sample_type,
                                            description, uom_type, config_object_user, aws_access_key_id,
                                            aws_secret_access_key, bucket_name, analysis_id)
            resutls = model_instance.split()
