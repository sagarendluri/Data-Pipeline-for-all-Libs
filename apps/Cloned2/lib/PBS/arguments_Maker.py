import sys


def args(pd_df, df):
    print(df)
    print("pd_frame", pd_df)

    def dname():
        dname = "dname"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    d_name = dname()

    def target():
        target = "target"
        if target in pd_df.columns:
            target = df[0].get(target)
            return target
        else:
            return None

    target_ = target()

    def sklearn():
        sklearn = "sklearn"
        if sklearn in pd_df.columns:
            sklearn = df[0].get(sklearn)
            return sklearn
        else:
            return None

    sklearn_ = sklearn()

    def algorithm():
        algorithm = "algorithm"
        if algorithm in pd_df.columns:
            algorithm = df[0].get(algorithm)
            return algorithm
        else:
            return None

    algorithm_ = algorithm()

    def config_object_user():
        config_object_user = "config_object_user"
        if config_object_user in pd_df.columns:
            config_object_user = df[0].get(config_object_user)
            return config_object_user
        else:
            return None

    config_object_user_ = config_object_user()

    def ai():
        ai = "ai"
        if ai in pd_df.columns:
            ai = df[0].get(ai)
            return ai
        else:
            return None

    ai_ = ai()

    # print(ai)
    def model_file():
        model_file = "model_file"
        if model_file in pd_df.columns:
            model_file = df[0].get(model_file)
            return model_file
        else:
            return None

    model_file_ = model_file()

    def predict():
        predict = "predict"
        if predict in pd_df.columns:
            predict = df[0].get(predict)
            return predict
        else:
            return None

    predict_ = predict()

    def phenome_data():
        phenome_data = "phenome_data"
        if phenome_data in pd_df.columns:
            phenome_data = df[0].get(phenome_data)
            return phenome_data
        else:
            return None

    phenome_data_ = phenome_data()

    def user_defined_terminology():
        user_defined_terminology = "user_defined_terminology"
        if user_defined_terminology in pd_df.columns:
            user_defined_terminology = df[0].get(user_defined_terminology)
            return user_defined_terminology
        else:
            return None

    user_defined_terminology_ = user_defined_terminology()

    def sample_type():
        sample_type = "sample_type"
        if sample_type in pd_df.columns:
            sample_type = df[0].get(sample_type)
            return sample_type
        else:
            return None

    sample_type_ = sample_type()

    def description():
        description = "description"
        if description in pd_df.columns:
            description = df[0].get(description)
            return description
        else:
            return None

    description_ = description()

    def uom_type():
        uom_type = "uom_type"
        if uom_type in pd_df.columns:
            uom_type = df[0].get(uom_type)
            return uom_type
        else:
            return None

    uom_type_ = uom_type()

    def all_M():
        all_M = "all_M"
        if all_M in pd_df.columns:
            all_M = df[0].get(all_M)
            return all_M
        else:
            return None

    all_M_ = all_M()

    def d_col():
        d_col = "d_col"
        if d_col in pd_df.columns:
            d_col = df[0].get(d_col)
            return d_col
        else:
            return None

    d_col_ = d_col()

    def index():
        index = "index"
        if index in pd_df.columns:
            index = df[0].get(index)
            return index
        else:
            return None

    index_ = index()

    def to_csv_name():
        to_csv_name = "to_csv_name"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    to_csv_name_ = to_csv_name()

    def min_depth():
        to_csv_name = "min_depth"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    min_depth = min_depth()

    def max_depth():
        to_csv_name = "max_depth"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    max_depth = max_depth()

    def min_samples_split():
        to_csv_name = "min_samples_split"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    min_samples_split = min_samples_split()

    def n_estimators_start():
        to_csv_name = "n_estimators_start"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    n_estimators_start = n_estimators_start()

    def n_estimators_stop():
        to_csv_name = "n_estimators_stop"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    n_estimators_stop = n_estimators_stop()

    def n_neighbors():
        to_csv_name = "n_neighbors"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    n_neighbors = n_neighbors()

    def xgb_objective():
        to_csv_name = "xgb_objective"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    xgb_objective = xgb_objective()

    def xgb_learning_rate():
        to_csv_name = "xgb_learning_rate"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    xgb_learning_rate = xgb_learning_rate()

    def xgb_max_depth():
        to_csv_name = "xgb_max_depth"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    xgb_max_depth = xgb_max_depth()

    def xgb_min_child_weight():
        to_csv_name = "xgb_min_child_weight"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    xgb_min_child_weight = xgb_min_child_weight()

    def svm_c():
        to_csv_name = "svm_c"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    svm_c = svm_c()

    def svm_gamma():
        to_csv_name = "svm_gamma"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    svm_gamma = svm_gamma()

    def svm_kernel():
        to_csv_name = "svm_kernel"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    svm_kernel = svm_kernel()

    def default():
        to_csv_name = "default"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    default = default()

    def cut_off():
        cut_off = "cut_off"
        if cut_off in pd_df.columns:
            cut_off = df[0].get(cut_off)
            return cut_off
        else:
            return 0.8

    cut_off_ = cut_off()

    def output():
        output = "output"
        if output in pd_df.columns:
            output = df[0].get(output)
            return output
        else:
            return "test_"

    output = output()

    def analysis_id():
        analysis_id = "analysis_id"
        if analysis_id in pd_df.columns:
            analysis_id = df[0].get(analysis_id)
            return analysis_id
        else:
            print("analysis_id IS not given")
            sys.exit()
            return print("analysis_id IS not given")

    analysis_id = analysis_id()

    def db_name():
        to_csv_name = "db_name"
        if to_csv_name in pd_df.columns:
            to_csv_name = df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None

    db_name = db_name()

    def N_features():
        dname = "N_features"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    def pca():
        dname = "PCA"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    def learning_rate_init():
        dname = "learning_rate_init"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    learning_rate_init = learning_rate_init()

    def max_iter():
        dname = "max_iter"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    max_iter = max_iter()

    def hidden_layer_sizes():
        dname = "hidden_layer_sizes"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    def activation():
        dname = "activation"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    activation = activation()

    def alpha():
        dname = "alpha"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    alpha = alpha()

    def early_stopping():
        dname = "early_stopping"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    early_stopping = early_stopping()

    def dataset_type():
        dname = "dataset_type"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    dataset_type = dataset_type()

    def test_size():
        dname = "test_size"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    test_size = test_size()

    def samples_column():
        dname = "sample_column"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    samples = samples_column()

    def filter_M():
        dname = "Filter_Methods"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    filter_M = filter_M()

    def vifNo():
        dname = "vifNo"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    vifNo = vifNo()

    def VIF():
        dname = "VIF"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    def Outliers():
        dname = "Outliers"
        if dname in pd_df.columns:
            dname = df[0].get(dname)
            return dname
        else:
            return None

    return d_name, target_, sklearn_, algorithm_, config_object_user_, ai_, model_file_, predict_, phenome_data_, user_defined_terminology_, sample_type_, description_, uom_type_, all_M_, d_col_, index_, to_csv_name_, min_depth, max_depth, min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors, xgb_objective, xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_c, svm_gamma, svm_kernel, default, cut_off_, output, analysis_id, db_name, N_features, pca, learning_rate_init, max_iter, hidden_layer_sizes, activation, alpha, early_stopping, dataset_type, test_size, samples, filter_M, vifNo, VIF, Outliers
