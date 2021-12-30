from tensorflow import keras
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend
import eli5
from eli5.sklearn import PermutationImportance
from dateutil.parser import parse
from scipy.stats import norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.constraints import maxnorm
from tensorflow.keras.layers import LSTM
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True
import shap
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


def Deep_Neural_Nets(X_train, y_train, X_test, y_test, target, model_):
    def reg_model():
        model = Sequential()
        model.add(Dense(500, input_dim=X_train.shape[1], activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mean_squared_error'])
        return model

    model = KerasRegressor(build_fn=reg_model, verbose=0)
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid.fit(X_train, y_train)
    random_best = grid.best_estimator_.predict(X_test)
    train_accuracy = grid.score(X_train, y_train) * 100
    test_accuracy = grid.score(X_test, y_test) * 100
    y_pred = grid.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    grids = grid.best_params_
    estimator = grid.best_estimator_
    model_name = "Keras_Reg_Deep_Neural_Nets"
    perm = PermutationImportance(grid, random_state=1).fit(X_train, y_train)
    importances = perm.feature_importances_
    print(importances)
    model_file_name = model_ + target + ".h5"
    estimator.model.save(model_file_name)
    cm = "None"
    return train_accuracy, test_accuracy, random_best, importances, grids, estimator, model_file_name, model_name, cm, mbe


#     DB_upload(Accuracy, X_train, X_test, y_test, random_best, importances,
#               grids, estimator, l, "None", target, model_file_name, model_name, dname, config_object_user,
#               user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name)
#                 print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#         except Exception as Ex:
#             print("ALL_Regression_Models exited with the error : ")#$%" % (Ex))
#             print(Ex)
def lstm(x_train, y_train, x_test, y_test, target, model_):
    num_steps = 1
    print(x_train.shape)
    x_train_shaped = np.reshape(np.array(x_train), newshape=(-1, num_steps, x_train.shape[1],))
    x_test_shaped = np.reshape(np.array(x_test), newshape=(-1, num_steps, x_test.shape[1],))
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(1, x_train.shape[1]), return_sequences=False),
        tf.keras.layers.Dense(26, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse')
    history = model.fit(x_train_shaped, y_train, validation_split=0.2, epochs = 20)
    DE = shap.DeepExplainer(model, x_train_shaped)
    print(DE)
    shap_values = DE.shap_values(x_train_shaped, check_additivity=False)
    vals = np.abs(shap_values).mean(0)
    importances = sum(vals).flatten()#pd.DataFrame(sum(vals).flatten(), columns=['Importance'])
    print("IMP_len",len(importances))
    train_y_pred = model.predict(x_train_shaped)
    print(train_y_pred)
    errors = abs(train_y_pred.flatten() - y_train)
    mape = np.mean(100 * (errors / y_train))
    train_accuracy = 100 - mape
    print(train_accuracy)
    test_y_pred = model.predict(x_test_shaped)
    errors = abs(test_y_pred.flatten() - y_test)
    mape = np.mean(100 * (errors / y_test))
    test_accuracy = 100 - mape
    print("test_len",len(test_y_pred))
    y_test = np.array(y_test)
    y_pred = np.array(test_y_pred.flatten())
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    grids = "None"
    estimator = "None" # grid.best_estimator_
    model_name = "LSTM_Deep_Neural_Nets"
    model_file_name = model_ + target + ".h5"
    model.save(model_file_name)
    return train_accuracy, test_accuracy, test_y_pred.flatten(), list(importances), grids, estimator, model_file_name, model_name, "None", mbe

