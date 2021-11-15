import ast
import numpy  as np
def  User_RandomForest_Classifier_grids(min_depth,max_depth,min_samples_split,n_estimators_start,
                                        n_estimators_stop,RandomForest_Classifier_grids,default):
    if "default" == default:

        return RandomForest_Classifier_grids
    else:
        RandomForest_Classifier_grids = {
            'max_depth': [int(x) for x in np.linspace([ast.literal_eval(x) for x in min_depth][0],
                                                      [ast.literal_eval(x) for x in max_depth][0], num=5)],
            'max_features': ['auto', 'sqrt'],
            'min_samples_split': [ast.literal_eval(x) for x in min_samples_split],
            'n_estimators': [int(x) for x in
                             np.linspace(start=[ast.literal_eval(x) for x in n_estimators_start][0],
                                         stop=[ast.literal_eval(x) for x in n_estimators_stop][0], num=5)]}
        return RandomForest_Classifier_grids
def User_XGBoost_Classifier_grids(xgb_objective,xgb_learning_rate,xgb_max_depth,xgb_min_child_weight,n_estimators_start,n_estimators_stop,XGBoost_Classifier_grids,default):
    if "default" == default:
        return XGBoost_Classifier_grids
    else:
        XGBoost_Classifier_grids =    {
                'learning_rate': [ast.literal_eval(x) for x in xgb_learning_rate],
                'max_depth': [ast.literal_eval(x) for x in xgb_max_depth],
                'min_child_weight': [ast.literal_eval(x) for x in xgb_min_child_weight],
                'n_estimators': [int(x) for x in np.linspace(start=
                                                             [ast.literal_eval(x) for x in n_estimators_start][
                                                                 0],
                                                             stop=
                                                             [ast.literal_eval(x) for x in n_estimators_stop][
                                                                 0],
                                                             num=5)]}
        return XGBoost_Classifier_grids
def User_KNeighbors_Classifier_grids(n_neighbors,KNeighbors_Classifier_grids,default):
    if "default" == default:
        return KNeighbors_Classifier_grids
    else:
        KNeighbors_Classifier_grids=    {'n_neighbors': np.arange(1, [ast.literal_eval(x) for x in n_neighbors][0])}
        return KNeighbors_Classifier_grids
def User_SVC_Classifier_grids(svm_c,svm_gamma,svm_kernel,SVC_Classifier_grids,default):
    if "default" == default:
        return SVC_Classifier_grids
    else:
        SVC_Classifier_grids =     {'C': [ast.literal_eval(x) for x in svm_c],
                                     'gamma': svm_gamma,
                                     'kernel': svm_kernel}
        return SVC_Classifier_grids
def User_Multilayer_Perceptron_Classifier_grids(learning_rate_init,max_iter,hidden_layer_sizes,activation,alpha,
                                                                    early_stopping,Multilayer_Perceptron_Classifier_grids,default):
    if "default" == default:
        return Multilayer_Perceptron_Classifier_grids

    else:
        "converting string to tuple to list"
        func = hidden_layer_sizes()
        tple = ast.literal_eval(func[0])
        hidden_layer_sizes = list(tple)
        Multilayer_Perceptron_Classifier_grids = {'learning_rate_init': [ast.literal_eval(x) for x in learning_rate_init],#ast.literal_eval(learning_rate_init()[0]),
         'max_iter':[ast.literal_eval(x) for x in max_iter] ,
         'hidden_layer_sizes': hidden_layer_sizes,
         'activation':activation ,
         'alpha':[ast.literal_eval(x) for x in alpha],
         'early_stopping':[ast.literal_eval(x) for x in early_stopping]}
        print(Multilayer_Perceptron_Classifier_grids)
    return Multilayer_Perceptron_Classifier_grids