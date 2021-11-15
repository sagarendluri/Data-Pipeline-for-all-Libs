from sklearn.feature_selection import VarianceThreshold
import numpy as np
def train_test(X_train,X_test,cut_off):
#     try:
        vs_constant = VarianceThreshold(threshold=0)
        # select the numerical columns only.
        numerical_x_train = X_train[X_train.select_dtypes([np.number]).columns]
        # fit the object to our data.
        vs_constant.fit(numerical_x_train)
        # get the constant colum names.
        constant_columns = [column for column in numerical_x_train.columns
                            if column not in numerical_x_train.columns[vs_constant.get_support()]]
        # detect constant categorical variables.
        constant_cat_columns = [column for column in X_train.columns 
                                if (X_train[column].dtype == "O" and len(X_train[column].unique())  == 1 )]
        all_constant_columns = constant_cat_columns + constant_columns
        X_train.drop(labels=all_constant_columns, axis=1, inplace=True)
        X_test.drop(labels=all_constant_columns, axis=1, inplace=True)
        print(X_train.shape)
        # threshold value for quasi constant.
        ####### Quasi-Constant Features
        threshold = 0.98
        # create empty list
        quasi_constant_feature = []
        # loop over all the columns
        for feature in X_train.columns:
            # calculate the ratio.
            predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
            # append the column name if it is bigger than the threshold
            if predominant >= threshold:
                quasi_constant_feature.append(feature) 
        X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        X_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        print(X_train.shape)
        #######Duplicated Features
        # transpose the feature matrice
        train_features_T = X_train.T
        ########  Correlation Filter Methods
        # select the duplicated features columns names
        duplicated_columns = train_features_T[train_features_T.duplicated()].index.values
        # drop those columns
        X_train.drop(labels=duplicated_columns, axis=1, inplace=True)
        X_test.drop(labels=duplicated_columns, axis=1, inplace=True)
        print(X_train.shape)
        correlated_features = set()
        correlation_matrix = X_train.corr()
        for i in range(len(correlation_matrix .columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > cut_off:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        X_train.drop(labels=correlated_features, axis=1, inplace=True)
        X_test.drop(labels=correlated_features, axis=1, inplace=True)
        print(X_train.shape)
        return X_train,X_test #,correlated_features
#     except:
#         print('sucsessfully completed QC')
