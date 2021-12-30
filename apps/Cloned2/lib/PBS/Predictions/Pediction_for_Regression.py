import joblib
import pymysql
import json
import sqlalchemy
import pandas as pd
from smart_open import smart_open
from os.path import splitext
import boto3

class DataPrediction:
    def __init__(self, data, i, path, model_file, user_defined_terminology, sample_type, description, uom_type,
                 config_object_user,
                 aws_access_key_id, aws_secret_access_key, bucket_name, analysis_id):
        self.data = data
        self.i = i
        self.path = path
        self.model_file = model_file
        self.user_defined_terminology = user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type = uom_type
        self.config_object_user = config_object_user
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
        self.analysis_id = analysis_id
        # self.sample_col = sample_col

    def split(self):
        config_object_user = self.config_object_user
        user_defined_terminology = self.user_defined_terminology
        sample_type = self.sample_type
        description = self.description
        uom_type = self.uom_type
        aws_access_key_id = self.aws_access_key_id
        aws_secret_access_key = self.aws_secret_access_key
        bucket_name = self.bucket_name
        num = len(self.data) // 2
        y = self.data[self.i]
        self.data = self.data.drop([self.i], axis=1)
        X_train = self.data[:num]
        X_test = self.data[num:]
        model_file = self.model_file
        analysis_id = self.analysis_id
        print("splited", model_file)
        file_name, extension = splitext(self.model_file)
        if extension == '.pkl':
            print("joblib")
            joblib_file = (smart_open(self.path))
            joblib_LR_model = joblib.load(joblib_file)
            y_pred = joblib_LR_model.predict(X_train)
            print(y_pred)
            cm1 = confusion_matrix(y_test, y_pred)
            Accuracy = metrics.accuracy_score(y_test, y_pred)
            print("Accuracy", Accuracy)
            cm = {'confusion_metrics': cm1.tolist()}
            print("CM", cm)
            print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            estimator, grid = "prediction", "None"
            l = 1
            target = self.i
            model_file_name, model, dname = self.path, self.path, self.path
            DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
                      None, grid, estimator, l, cm, target, model_file_name, model, dname, config_object_user,
                      user_defined_terminology, sample_type, description, uom_type, analysis_id)
        elif extension == '.h5':
            tf_keras_file = (smart_open(self.path))
            with h5py.File(tf_keras_file, 'r') as f:
                data = f
                model = tf.keras.models.load_model(data)
                y_pred = model.predict(X_test)
                print(y_pred)
                cm1 = confusion_matrix(y_test, y_pred)
                Accuracy = metrics.accuracy_score(y_test, y_pred)
                print("Accuracy", Accuracy)
                cm = {'confusion_metrics': cm1.tolist()}
                print("CM", cm)
                print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
                print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                estimator, grid = "prediction", None
                l = 1
                target = self.i
                model_file_name, model, dname = self.path, self.path, self.path
                DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
                          None, grid, estimator, l, cm, target, model_file_name, model, dname,
                          config_object_user,
                          user_defined_terminology, sample_type, description, uom_type, analysis_id)
        elif extension == '.zip':
            import h2o
            h2o.init()
            from h2o.estimators import H2ORandomForestEstimator
            from h2o.estimators import H2OGradientBoostingEstimator
            from h2o.estimators.glm import H2OGeneralizedLinearEstimator
            from h2o.estimators import H2OXGBoostEstimator
            import numpy as np
            import sklearn.metrics as sm
            cols = list(self.data.columns)
            print(cols)
            #                 cols.append(self.i)
            selected_cols = self.data[cols]
            x = list(cols)
            df = h2o.H2OFrame(selected_cols)
            #                 train,  test = df.split_frame(ratios=[.8])
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            s3 = session.resource('s3')
            s3.meta.client.download_file(Filename=model_file, Bucket=bucket_name, Key=model_file)
            mdl = h2o.import_mojo(model_file)
            predictions = mdl.predict(df).as_data_frame(use_pandas=True)
            df = pd.concat([y, predictions], axis=1)
            print(df)
            df.to_csv("h2o_prediction.csv")
            print("R2 score =", round(sm.r2_score(y, predictions), 2))
