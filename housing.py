import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.kernel_ridge import *
from sklearn import tree
from sklearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import *
from sklearn.linear_model import *
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from keras.layers import *
from keras.models import  *
from keras.utils import to_categorical
from keras.metrics import mse, mae
from encoder import Encoder

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(15, input_dim=15, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

from keras.wrappers.scikit_learn import *

df = pd.read_csv('/Users/bjayaram/Documents/Kaggle/housing/train.csv')

df['Storeys'] = df['MSSubClass'].map({20:1,30:1,40:1,45:1.5,50:1.5, 60:2, 70:2, 75: 2.5, 80:5, 85:5, 90:10, 120:1, 150:1.5, 160:2, 180:5,190:7.5})



data_columns = list(df.columns)
data_columns.remove('Id')
data_columns.remove('SalePrice')
print data_columns

price_columns= ['SalePrice']

y_train = df[price_columns].values.ravel()
x_train = df[data_columns]
pca = PCA(n_components=15)



d = defaultdict(LE)
scores = []
x1_train = Encoder.encode_data(x_train, [], d)
print 'X1_TRAIN = ',x1_train.shape
# x1_train = pca.fit_transform(x1_train, y_train)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)


np.random.seed(7)
estimators = []
estimators.append(('standardize', MinMaxScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
# pipeline.fit(x1_train, y_train)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, x1_train, y_train, cv=3)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#
# trainX, valX, trainY,valY = train_test_split(x1_train, y_train, test_size=0.1)
#
# alpha_list = range(10,10000,10)
# beta_list = range(10,10000,10)
# alphas = map(lambda x: 10/(1+x), alpha_list)
# betas = map(lambda x: 10/(1+x), beta_list)
# minScore = 1000000000000000
# # model = LassoLarsCV(normalize=True, cv=8, n_jobs=8)
# model = GradientBoostingRegressor(n_estimators=100, max_features=0.75)
# model.fit(x1_train, y_train)
# model = RidgeCV(normalize=True, alphas = alphas)
# model = LassoCV(normalize=True, alphas = alphas, max_iter=100, n_jobs=24, cv=10)
# model = RandomForestRegressor(n_estimators=100)
#
#
#
model = Sequential()
input_neurons = x1_train.shape[1]
model.add(Dense(input_neurons, input_shape=(input_neurons,), kernel_initializer='normal', activation='relu'))
model.add(Dense(500000, kernel_initializer='normal', activation='relu'))
# model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[mse,mae])
model.fit(x1_train, y_train, verbose=2, validation_split=0.2, shuffle=True, epochs=50)
print model.evaluate(x1_train, y_train)
# estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=5, verbose=0)
#
# kfold = KFold(n_splits=10)
# # results = cross_val_score(estimator, x_train, y_train, cv=kfold)
# # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# minScore = 1000000000000000
# # model = LassoLarsCV(normalize=True, cv=8, n_jobs=8)
# print alphas
# print betas

# model = ElasticNetCV(normalize=True, alphas = alphas, l1_ratio=betas, selection='random')
#
# model = RandomForestRegressor(n_estimators=100, n_jobs=10)
# model.fit(trainX,trainY)
# z = model.predict(valX)
# score = mean_squared_error(valY, z)

df = pd.read_csv('/Users/bjayaram/Documents/Kaggle/housing/test.csv')
df['Storeys'] = df['MSSubClass'].map({20:1,30:1,40:1,45:1.5,50:1.5, 60:2, 70:2, 75: 2.5, 80:5, 85:5, 90:10, 120:1, 150:1.5, 160:2, 180:5,190:7.5})

data_columns = list(df.columns)
data_columns.remove('Id')
print data_columns
# data_columns.remove('SalePrice')

x_test = df[data_columns]
x1_test = Encoder.encode_data(x_test, [], d)
print 'X1_TEST = ',x1_test.shape
# x1_test = pca.transform(x1_test)


y= model.predict(x1_test)
res= pd.DataFrame()
res['Id']=df['Id']
res['SalePrice']=y
res.to_csv('/Users/bjayaram/Documents/Kaggle/housing/results.csv', index=None)
