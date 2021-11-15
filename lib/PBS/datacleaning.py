import numpy as np
import pandas as pd
from sklearn import preprocessing
class DataCleaning(object):
    def __init__(self, df1, i):
        self.df1 = df1
        self.i = i
        self.x = df1.drop(self.i, axis=1)
        self.y = df1[self.i]

    def dtypes_handliing(self):
        try:
            print("Data analyzing")
            # Select the bool columns
            self.bool_col = self.x.select_dtypes(include='bool')
            # select the float columns
            self.float_col = self.x.select_dtypes(include=[np.float64])
            # select the int columns
            self.int_col = self.x.select_dtypes(include=[np.int64])
            #   # select non-numeric columns
            self.cat_col = self.x.select_dtypes(include=['category', object])
            if self.cat_col ==None:
                return self.cat_col
            else:
                self.cat_col = self.x.fillna(self.x.select_dtypes(include=['category', object].mode().iloc[0], inplace=True))
                return self.cat_col
            self.date_col = self.x.select_dtypes(include=['datetime64'])
            return self.bool_col, self.float_col, self.int_col, self.cat_col, self.date_col
        except:
            print('Data analyzing failed')

    def handiling_categorical(self):
        try:
            self.cat_result = []
            df_most_common_imputed = self.cat_col.apply(lambda x: x.fillna(x.value_counts().index[0]))
            print(df_most_common_imputed)
            for col in list(df_most_common_imputed):
                labels, levels = pd.factorize(self.cat_col[col].unique())
                if sum(labels) <= self.cat_col[col].shape[0]:
                    self.cat_result.append(pd.get_dummies(self.cat_col[col], prefix=col))
                print("Categorical Data analyzing")
            return self.cat_result
        except:
            print("Categorical Data analyzing failed")

    def handiling_int_col(self):
        try:
            print("int Data analyzing")
            self.int_result = []
            for col in self.int_col:
                labels, levels = pd.factorize(self.int_col[col].unique())
                if len(labels) == self.int_col[col].shape[0]:
                    re = self.int_col.drop([col], axis=1)
                else:
                    self.int_result.append(self.int_col[col])
            return self.int_result
        except:
            print("int Data analyzing failed")

    def concat_cat(self):
        result = [self.cat_result]
        for fname in result:
            if fname == []:
                print('No objects to concat')
            else:
                self.data = pd.concat([col for col in fname], axis=1)
                self.cleaned_Data_frm = pd.concat([self.data.reindex(self.y.index)], axis=1)
                print(list(self.cleaned_Data_frm.columns))
                return self.cleaned_Data_frm

    def concat_int(self):
        result2 = [self.int_result]
        for fname2 in result2:
            if fname2 == []:
                print('No int_cols to concat')
            else:
                self.data2 = pd.concat([col for col in fname2], axis=1)
                self.cleaned_Data_frm1 = pd.concat([self.data2.reindex(self.y.index)], axis=1)
                return self.cleaned_Data_frm1

    def encoder(self):
        if (self.y.dtype == object or self.y.dtype == bool):
            self.df1[self.i].fillna(self.df1[self.i].mode()[0], inplace=True)
            print(self.df1)
            self.y = self.df1[self.i]
            label_encoder = preprocessing.LabelEncoder()
            self.y = label_encoder.fit_transform(self.y.astype(str))
            self.dataset = pd.DataFrame()
            self.dataset[self.i] = self.y.tolist()
            print('Multiclass_classification')
            self.types = 'Multiclass_classification'
            return self.dataset[self.i], self.types
        elif self.y.dtypes == np.int:
            self.types = 'Classification_problem'
            print('Classification_problem')
            return self.y, self.types
        else:
            print('Regression_problem')
            self.types = 'Regression_problem'
            return self.y,self.types

    def QC(self, cleaned_Data_frm, cleaned_Data_frm1, y):
        try:
            print('Models Building')
            float_cols = self.float_col
            result = pd.concat([cleaned_Data_frm, cleaned_Data_frm1, y, float_cols], axis=1)
            self.data_sorted1 = result.sort_values(self.i)
            data_sorted = self.data_sorted1.loc[:, ~self.data_sorted1.columns.duplicated()]
            print(data_sorted.shape)
            return data_sorted
        except:
            print('data returned faild')

    def classification(self, cleaned_Data_frm1, cleaned_Data_frm, y):
        try:
            print("Model building")
            float_cols = self.float_col
            result = pd.concat([cleaned_Data_frm1, cleaned_Data_frm, y, float_cols], axis=1)
            self.data_sorted1 = result.loc[:, ~result.columns.duplicated()]
            self.data_sorted2 = self.data_sorted1.sort_values(self.i)
            self.data_sorted = self.data_sorted2.dropna(thresh=self.data_sorted2.shape[0] * 0.5, how='all', axis=1)
    # data_sorted = self.data_sorted()
            return self.data_sorted
        except:
            print('data returned faild')
