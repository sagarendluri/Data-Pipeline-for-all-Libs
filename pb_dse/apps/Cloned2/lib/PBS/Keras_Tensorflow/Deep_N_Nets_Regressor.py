import keras
from keras.callbacks import EarlyStopping
from keras import backend
import eli5
from eli5.sklearn import PermutationImportance
from dateutil.parser import parse
from scipy.stats import norm
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.constraints import maxnorm
def Deep_Neural_Nets(X_train , y_train,X_test, y_test, target):
    def Reg_model():
        model = Sequential()
        model.add(Dense(500, input_dim=X_train.shape[1], activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mean_squared_error'])
        return model
    model = KerasRegressor(build_fn=Reg_model, verbose=0)
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    random_best = grid.best_estimator_.predict(X_test)
    errors = abs(random_best - y_test)
    mape = np.mean(100 * (errors / y_test))
    Accuracy = 100 - mape
    grids = grid.best_params_
    estimator = grid.best_estimator_
    model_name = "Keras_Reg_Deep_Neural_Nets"
    perm = PermutationImportance(grid, random_state=1).fit(X_train, y_train)
    importances = perm.feature_importances_
    model_file_name = "Keras_Reg_Deep_Neural_Nets_"+target+".h5"
    estimator.model.save(model_file_name)
    cm = "None"
    return Accuracy,random_best,importances,grids,estimator,model_file_name,model_name,cm
#     DB_upload(Accuracy, X_train, X_test, y_test, random_best, importances,
#               grids, estimator, l, "None", target, model_file_name, model_name, dname, config_object_user,
#               user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name)
#                 print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#         except Exception as Ex:
#             print("ALL_Regression_Models exited with the error : ")#$%" % (Ex))
#             print(Ex)


