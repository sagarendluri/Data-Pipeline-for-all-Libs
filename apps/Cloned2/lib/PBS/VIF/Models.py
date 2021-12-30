from apps.Cloned2.lib.PBS.VIF.Remove_Multi_Co_Linearity import VIF_info
from apps.Cloned2.lib.PBS.Data_preprocessing.datacleaning import DataCleaning


class VIF_Models:
    def __init__(self, path, i, ini, dname, config_object_user, user_defined_terminology, sample_type, description,
                 uom_type, d_col, cut_off, analysis_id, min_depth, max_depth, min_samples_split, n_estimators_start,
                 n_estimators_stop, n_neighbors, xgb_objective, xgb_learning_rate, xgb_max_depth, xgb_min_child_weight,
                 svm_c, svm_gamma, svm_kernel, default, db_name, N_features, learning_rate_init, max_iter,
                 hidden_layer_sizes, activation, alpha, early_stopping, dataset_type, test_size,vifNo):
        self.path = path
        self.i = i
        self.ini = ini
        self.dname = dname
        self.config_object_user = config_object_user
        self.user_defined_terminology = user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type = uom_type
        self.d_col = d_col
        self.cut_off = cut_off
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
        self.N_features = N_features
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.dataset_type = dataset_type
        self.test_size = test_size
        self.vifNo = vifNo

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
        cut_off = self.cut_off
        config_object_user = self.config_object_user
        user_defined_terminology = self.user_defined_terminology
        sample_type = self.sample_type
        description = self.description
        uom_type = self.uom_type
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
        N_features = self.N_features
        learning_rate_init = self.learning_rate_init
        max_iter = self.max_iter
        hidden_layer_sizes = self.hidden_layer_sizes
        activation = self.activation
        alpha = self.alpha
        early_stopping = self.early_stopping
        dataset_type = self.dataset_type
        test_size = self.test_size
        vifNo = self.vifNo
        model_instance = DataCleaning(df1, i)
        model_instance.dtypes_handliing()
        model_instance.handiling_categorical()
        model_instance.handiling_int_col()
        cleaned_Data_frm = model_instance.concat_cat()
        cleaned_Data_frm1 = model_instance.concat_int()
        y, label_types = model_instance.encoder()
        if y.dtypes == 'int64' or y.dtypes == 'int32':
            df = model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1, y)
            model = VIF_info()
            df = model.fit(df, vifNo)
            data = data[[df]]
            models = ALL_CLS_Modeling(data, y, i, label_types, dname, config_object_user, user_defined_terminology,
                                      sample_type, description, uom_type, cut_off, analysis_id, min_depth, max_depth,
                                      min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors,
                                      xgb_objective,
                                      xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_c, svm_gamma,
                                      svm_kernel,
                                      default, db_name, N_features, learning_rate_init, max_iter, hidden_layer_sizes,
                                      activation,
                                      alpha,
                                      early_stopping, dataset_type, test_size)
            # models.parameter_tuning()
            models.Building_Models_Cls()
        else:
            data = model_instance.QC(cleaned_Data_frm, cleaned_Data_frm1, y)
            model = VIF_info()
            df = model.fit(data, vifNo)
            data = data[[df]]
            models = ALL_RGN_Modeling(data, y, i, label_types, dname, config_object_user, user_defined_terminology,
                                      sample_type, description, uom_type, cut_off, analysis_id, min_depth, max_depth,
                                      min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors,
                                      xgb_objective, xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_c,
                                      svm_gamma, svm_kernel, default, db_name, N_features, learning_rate_init, max_iter,
                                      hidden_layer_sizes, activation, alpha, early_stopping, dataset_type, test_size)
            #                 models.parameter_tuning()
            models.Building_Models_Reg()
