import pandas as pd
from smart_open import smart_open
from apps.Cloned2.lib.PBS.Data_preprocessing.datacleaning import DataCleaning
from apps.Cloned2.lib.PBS.Run_All_Models.ALL_Classification_Models import SK_models
from apps.Cloned2.lib.PBS.Run_All_Models.ALL_Regression_models import ALL_RGN_Modeling
from apps.Cloned2.lib.PBS.Outliers.Find_Outliers import iqr


class Run_R_and_C_Models:
    def __init__(self, path, i, ini, dname, config_object_user, user_defined_terminology, sample_type, description,
                 uom_type, all_M, d_col, cut_off, aws_access_key_id, aws_secret_access_key, bucket_name, analysis_id,
                 min_depth, max_depth, min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors,
                 xgb_objective, xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_c, svm_gamma, svm_kernel,
                 default, db_name, learning_rate_init, max_iter, hidden_layer_sizes, activation, alpha, early_stopping,
                 dataset_type, test_size, samples):
        self.path = path
        self.i = i
        self.ini = ini
        self.dname = dname
        self.config_object_user = config_object_user
        self.user_defined_terminology = user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type = uom_type
        self.all_M = all_M
        self.d_col = d_col
        self.cut_off = cut_off
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
        self.analysis_id = analysis_id
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators_start = n_estimators_start
        self.n_estimators_stop = n_estimators_stop
        self.n_neighbors = n_neighbors
        self.xgb_objective = xgb_objective
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_min_child_weight = xgb_min_child_weight
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.svm_kernel = svm_kernel
        self.default = default
        self.db_name = db_name
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.dataset_type = dataset_type
        self.test_size = test_size
        self.samples = samples

    def csv_Predicts(self):
        # try:
        df = pd.read_csv(smart_open(self.path))
        print(df)

        def drop(df):
            if self.d_col == None:
                return df
            else:
                df = df.drop(self.d_col, axis=1)
                return df

        df1 = drop(df)
        samples_ = df1[[self.samples]]
        print(samples_)
        i = self.i
        ini = self.ini
        dname = self.dname
        cut_off = self.cut_off
        config_object_user = self.config_object_user
        user_defined_terminology = self.user_defined_terminology
        sample_type = self.sample_type
        description = self.description
        uom_type = self.uom_type
        aws_access_key_id = self.aws_access_key_id
        aws_secret_access_key = self.aws_secret_access_key
        bucket_name = self.bucket_name
        analysis_id = self.analysis_id
        min_depth = self.min_depth
        max_depth = self.max_depth
        min_samples_split = self.min_samples_split
        n_estimators_start = self.n_estimators_start
        n_estimators_stop = self.n_estimators_stop
        n_neighbors = self.n_neighbors
        xgb_objective = self.xgb_objective
        xgb_learning_rate = self.xgb_learning_rate
        xgb_max_depth = self.xgb_max_depth
        xgb_min_child_weight = self.xgb_min_child_weight
        svm_c = self.svm_c
        svm_gamma = self.svm_gamma
        svm_kernel = self.svm_kernel
        default = self.default
        db_name = self.db_name
        learning_rate_init = self.learning_rate_init
        max_iter = self.max_iter
        hidden_layer_sizes = self.hidden_layer_sizes
        activation = self.activation
        alpha = self.alpha
        early_stopping = self.early_stopping
        dataset_type = self.dataset_type
        test_size = self.test_size
        samples = self.samples
        model_instance = DataCleaning(df1, i)
        model_instance.dtypes_handliing()
        model_instance.handiling_categorical()
        model_instance.handiling_int_col()
        cleaned_Data_frm = model_instance.concat_cat()
        cleaned_Data_frm1 = model_instance.concat_int()
        y, label_types = model_instance.encoder()
        if y.dtypes == 'int64' or y.dtypes == 'int32':
            data = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1, y)
            result = pd.concat([data, samples_], axis=1)
            models = SK_models(filled_median, y, i, label_types, dname, config_object_user, user_defined_terminology,
                               sample_type,
                               description, uom_type, cut_off, analysis_id, min_depth, max_depth, min_samples_split,
                               n_estimators_start, n_estimators_stop, n_neighbors, xgb_objective, xgb_learning_rate,
                               xgb_max_depth, xgb_min_child_weight, svm_c, svm_gamma, svm_kernel, default, db_name,
                               learning_rate_init, max_iter, hidden_layer_sizes, activation, alpha, early_stopping,
                               dataset_type, test_size, samples)
            models.Building_Models()

        else:
            data = model_instance.QC(cleaned_Data_frm, cleaned_Data_frm1, y)
            result = pd.concat([data, samples_], axis=1)
            models = ALL_RGN_Modeling(filled_median, y, i, label_types, dname, config_object_user, user_defined_terminology,
                                      sample_type, description, uom_type, cut_off, analysis_id, min_depth, max_depth,
                                      min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors,
                                      xgb_objective,
                                      xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_c, svm_gamma,
                                      svm_kernel,
                                      default, db_name, learning_rate_init, max_iter, hidden_layer_sizes, activation,
                                      alpha,
                                      early_stopping, dataset_type, test_size, samples)
            models.Building_Models_Reg()
    # except Exception as Ex:
    #     print("Pipeline exited with the error : #$%" % (Ex))
    #     print(Ex)
