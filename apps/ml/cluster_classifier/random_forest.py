import joblib
import pandas as pd
class RandomForestClassifier:
    def __init__(self):
        path_to_artifacts ="//"
        self.values_fill_missing =  joblib.load(r"C:\Users\sagar\cloned\train_mode.joblib")
        # self.encoders = joblib.load(r"C:\Users\sagar\cloned\encoders.joblib")
        self.model = joblib.load(r"C:\Users\sagar\cloned\random_forest.joblib")

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        self.d = len(input_data)
        input_data = pd.DataFrame(input_data, index=range(self.d))
        # fill missing values
        input_data.fillna(self.values_fill_missing)

        return input_data

    def predict(self, input_data):
        print("input_data",input_data)
        print(self.model.predict(input_data))

        return self.model.predict(input_data)

    def postprocessing(self, input_data):
        print("predict",input_data)
        return {"prediction": input_data,  "status": "Congrats"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Got_Error", "message": str(e)}

        return prediction