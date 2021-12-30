import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler
import seaborn as sns
import pandas as pd
import theano.tensor as T
import arviz as az
import matplotlib.pyplot as plt
df = pd.read_csv(r'top_5k_319_samples.csv')
dff = df.iloc[:,:4994]
df1 = pd.read_csv(r'common_samples.csv')
df1 = df1[['Line','1st Layer Clusters']]
df1 =df1.fillna(df1.mean())
phe = df1.set_index('Line')
df3 = dff.set_index('index')
gp = pd.merge(df3, phe, left_index=True, right_index=True)
gp = gp.reset_index()
gp  = gp.iloc[:,4500:]
gp['1st Layer Clusters']= LabelEncoder().fit_transform(gp['1st Layer Clusters'])
y_obs = gp['1st Layer Clusters'].values
x_n = gp.columns[:-1]
x = gp[x_n].values
x = StandardScaler().fit_transform(x)
ndata = x.shape[0]
nparam = x.shape[1]
nclass = len(gp['1st Layer Clusters'].unique())
print( y_obs.shape, x.shape )
with pm.Model() as snps_model:
    X_data = pm.Data('X_data', x)
    y_obs_data = pm.Data('y_obs_data', y_obs)
    alfa = pm.Normal('alfa', mu=0, sd=1, shape=nclass)
    beta = pm.Normal('beta', mu=0, sd=1, shape=(nparam, nclass))
    mu = tt.dot(X_data, beta) + alfa
    p = tt.nnet.softmax(mu)
    yl = pm.Categorical('obs', p=p, observed=y_obs_data)
    trace = pm.sample()
    idata = az.from_pymc3(trace)
    pm.traceplot(idata)
with snps_model:
    pm.set_data({'X_data':np.random.normal(size=(4, 994))})
    pred = pm.fast_sample_posterior_predictive(trace)