import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


class VIF_info:
    def __init__(self, df, vifNo):
        self.vif_No = vifNo
        self.df = df

    def fit(self):
        vif_info = pd.DataFrame()
        vif_info['VIF'] = [variance_inflation_factor(self.df.values, i) for i in range(self.df.shape[1])]
        vif_info['Column'] = self.df.columns
        cols = vif_info.loc[vif_info["VIF"] < vifNo, 'Column']
        return cols

