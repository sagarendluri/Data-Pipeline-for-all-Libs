import numpy as np
import pandas as pd
outliers = []

dict = {}


def iqr(X):
    for col in X.columns:
        data = sorted(list(X[col]))
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        lr_bound = q1 - (1.5 * (q3 - q1))
        ur_bound = q3 + (1.5 * (q3 - q1))
        outliers.append(col)
        for i in data:
            if i < lr_bound or i > ur_bound:
                if col not in dict:
                    dict[col] = []
                dict[col].append(i)
                val[col] = val[col].replace([i], np.nan)
        X[X.columns] = X[X.columns].apply(pd.to_numeric, errors='coerce')
        df = X.fillna(X.median())
        return df
