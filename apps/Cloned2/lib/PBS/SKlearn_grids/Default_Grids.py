import numpy as np
def RandomForest_Classifier_grids():
    RandomForest_Classifier_grids ={
                                    'max_depth': [int(x) for x in np.linspace(1, 45, num=3)],
                                    'max_features': ['auto', 'sqrt'],
                                    'min_samples_split': [5, 10],
                                    'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=5)]}
    return RandomForest_Classifier_grids
def XGBoost_Classifier_grids():
    XGBoost_Classifier_grids =  {
                                'learning_rate': [0.045],
                                'max_depth': [3, 4],
                                'min_child_weight': [2],
                                'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=5)]}
    return XGBoost_Classifier_grids
def KNeighbors_Classifier_grids():
    KNeighbors_Classifier_grids ={'n_neighbors': np.arange(1, 25)}
    return KNeighbors_Classifier_grids
def SVC_Classifier_grids():
    SVC_Classifier_grids = {'C': [0.001, 0.01, 0.1, 1, 10],
                                 'gamma': [0.001, 0.01, 0.1, 1],
                                 'kernel': ['linear']}
    return SVC_Classifier_grids
def Multilayer_Perceptron_Classifier_grids():
    Multilayer_Perceptron_Classifier_grids ={'learning_rate_init': [0.0001],
                                             'max_iter': [300],
                                             'hidden_layer_sizes': [(500, 400, 300, 200, 100),
                                                                    (400, 400, 400, 400, 400),
                                                                    (300, 300, 300, 300, 300),
                                                                    (200, 200, 200, 200, 200)],
                                             'activation': ['logistic', 'relu', 'tanh'],
                                             'alpha': [0.0001, 0.001, 0.005],
                                             'early_stopping': [True, False]}
    return Multilayer_Perceptron_Classifier_grids