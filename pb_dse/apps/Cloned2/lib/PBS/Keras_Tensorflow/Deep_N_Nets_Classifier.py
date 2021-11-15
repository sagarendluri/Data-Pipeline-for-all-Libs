import keras
from keras.callbacks import EarlyStopping
from keras import backend
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import norm
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import GridSearchCV
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.constraints import maxnorm
from keras.layers import Dropout
import numpy as np
import joblib
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import accuracy_score,confusion_matrix,mean_absolute_error
def Deep_Neural_Nets(X_train , y_train,X_test, y_test, target):
    model_file_name = "Keras_DNN_" + target + '.h5'
    model_name = "Keras_DNN_Classifier"
    if 2 == len(y_test):
        def DNN():
            model = Sequential()
            model.add(Dense(512, input_dim=X_train.shape[1], init='normal', activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
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
        grid_result = grid.fit(X_train, y_train)
        estimator = grid.best_estimator_
        y_pred_keras=grid.predict(X_test)
        cm1 =confusion_matrix(y_test, y_pred_keras)
        Accuracy = accuracy_score(y_test, y_pred_keras)
        cm = {'confusion_metrics':cm1.tolist()}
        perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train,y_train)
        importance= perm.feature_importances_
        return  Accuracy,y_pred_keras,importance,grids,estimator,model_file_name,model_name,cm

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
        classifier = KerasClassifier(build_fn=DNN, epochs=50, batch_size=10, verbose=1)
        weight_constraint = [1, 2, 3, 4, 5]
        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(X_train, y_train)
        estimator = grid.best_estimator_
        y_pred_keras = grid.predict(X_test)
        cm1 = confusion_matrix(y_test, y_pred_keras)
        Accuracy = accuracy_score(y_test, y_pred_keras)
        cm = {'confusion_metrics': cm1.tolist()}
        grid.best_estimator_.model.save(model_file_name)
        grids = grid.best_params_
        perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train, y_train)
        importance = perm.feature_importances_
        return Accuracy,y_pred_keras,importance,grids,estimator,model_file_name,model_name,cm
