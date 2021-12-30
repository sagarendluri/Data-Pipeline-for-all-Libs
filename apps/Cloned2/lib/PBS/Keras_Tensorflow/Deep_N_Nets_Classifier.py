import keras
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.constraints import maxnorm
from tensorflow.keras.layers import Dropout
import numpy as np
import joblib
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import shap

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

def Deep_Neural_Nets(X_train, y_train, X_test, y_test, target, model_):
    model_file_name = "Keras_DNN_" + model_ + "_" + target + '.h5'
    model_name = "Keras_DNN_Classifier"
    if 2 == len(y_test):
        def DNN():
            model = Sequential()
            model.add(Dense(50, input_dim=X_train.shape[1], init='normal', activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(32, init='normal', activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(1, init='normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
            return model

        classifier = KerasClassifier(build_fn=DNN, verbose=1)
        batch_size = [5]
        epochs = [2]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        y_test = y_test.reshape(len(y_test), 1)
        y_pred = y_pred.reshape(len(y_pred), 1)
        diff = (y_test - y_pred)
        mbe = diff.mean()
        estimator = grid.best_estimator_
        y_pred_keras = grid.predict(X_test)
        cm1 = confusion_matrix(y_test, y_pred_keras)
        Accuracy = accuracy_score(y_test, y_pred_keras)
        cm = {'confusion_metrics': cm1.tolist()}
        perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train, y_train)
        importance = perm.feature_importances_
        return Accuracy, y_pred_keras, importance, grids, estimator, model_file_name, model_name, cm, mbe

    else:
        a = np.unique(y_train)
        a.sort()
        b = a[-1]
        b += 1

        def DNN(dropout_rate=0.0, weight_constraint=0):
            # create model
            model = Sequential()
            model.add(Dense(42, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu',
                            kernel_constraint=maxnorm(weight_constraint)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
            model.add(Dense(b, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        classifier = KerasClassifier(build_fn=DNN, epochs=10, batch_size=10, verbose=1)
        weight_constraint = [1, 2, 3, 4, 5]
        dropout_rate = [0.0, 0.1, 0.2, 0.3]
        param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
        grid.fit(X_train, y_train)
        estimator = grid.best_estimator_
        y_pred_ = grid.best_estimator_.predict(X_test)
        y_pred = y_pred_
        train_accuracy = grid.score(X_train, y_train) * 100
        test_accuracy = grid.score(X_test, y_test) * 100
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        y_test = y_test.reshape(len(y_test), 1)
        y_pred = y_pred.reshape(len(y_pred), 1)
        diff = (y_test - y_pred)
        mbe = diff.mean()
        cm1 = confusion_matrix(y_test, y_pred)
        cm = {'confusion_metrics': cm1.tolist()}
        grid.best_estimator_.model.save(model_file_name)
        grids = grid.best_params_
        perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train, y_train)
        importance = perm.feature_importances_
        return train_accuracy, test_accuracy, y_pred_, importance, grids, estimator, model_file_name, model_name, cm, mbe


def lstm(x_train, y_train, x_test, y_test, target, model_):
    print("classification")
    num_steps = 1
    print(x_train.shape)

    def actfunc(y_test):
        if 2 == len(y_test):
            return "sigmoid"
        else:
            return "softmax"

    a = np.unique(y_train)
    a.sort()
    b = a[-1]
    b += 1
    function = actfunc(y_test)
    x_train_shaped = np.reshape(np.array(x_train), newshape=(-1, num_steps, x_train.shape[1],))
    x_test_shaped = np.reshape(np.array(x_test), newshape=(-1, num_steps, x_test.shape[1],))
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(1, x_train.shape[1]), return_sequences=False),
        tf.keras.layers.Dense(26,kernel_initializer='uniform', activation='relu'),
        tf.keras.layers.Dense(1, activation=function)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse')
    history = model.fit(x_train_shaped, y_train, validation_split=0.2, epochs=20)
    DE = shap.DeepExplainer(model, x_train_shaped)
    print(DE)
    shap_values = DE.shap_values(x_train_shaped, check_additivity=False)
    vals = np.abs(shap_values).mean(0)
    importances = sum(vals).flatten()  # pd.DataFrame(sum(vals).flatten(), columns=['Importance'])
    train_y_pred = model.predict(x_train_shaped)
    train_accuracy = accuracy_score(y_train, train_y_pred.flatten())
    test_y_pred_ = model.predict(x_test_shaped)
    test_y_pred = test_y_pred_
    test_accuracy = accuracy_score(y_test, test_y_pred.flatten())
    cm1 = confusion_matrix(y_test, test_y_pred)
    cm = {'confusion_metrics': cm1.tolist()}
    y_test = np.array(y_test)
    y_pred = np.array(test_y_pred.flatten())
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    grids = "None"
    estimator = "None"  # grid.best_estimator_
    model_name = "LSTM_Deep_Neural_Nets"
    model_file_name = model_ + target + ".h5"
    model.save(model_file_name)
    return train_accuracy, test_accuracy, test_y_pred_.flatten(), importances, grids, estimator, model_file_name, model_name, cm, mbe
