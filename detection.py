import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.feature_selection import *
from sklearn.preprocessing import MinMaxScaler as MMS,LabelEncoder as LE
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.kernel_ridge import *
from sklearn import tree
from sklearn.ensemble import *
from sklearn.ensemble import *
from sklearn.svm import LinearSVC, SVC, OneClassSVM
from sklearn.model_selection import *
from sklearn.linear_model import *
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from encoder import Encoder
from imblearn.over_sampling import SMOTE, ADASYN


df = pd.read_csv('/Users/bjayaram/Documents/Kaggle/criminal/criminal_train.csv')
label_cols = u'Criminal'
data_columns=list(df.columns)
data_columns.remove('PERID')
data_columns.remove('Criminal')
print data_columns


y_orig = df[label_cols].values
x_orig = df[data_columns].values








smote = SMOTE(ratio='minority', k_neighbors= 1500, kind='svm', n_jobs=16)
adasyn = ADASYN( n_jobs=24, n_neighbors=1500)
chi2=SelectKBest(chi2, k=20)
pca = PCA(n_components=10)
mms = MMS()
x_train, y_train = adasyn.fit_sample(x_orig, y_orig)
x_train = mms.fit_transform(x_train)
x_train = chi2.fit_transform(x_train, y_train)
x_train = pca.fit_transform(x_train, y_train)
print "SMOTE sample generated"
model1=RandomForestClassifier(n_estimators=1000, n_jobs=-1)
C_vals=map(lambda x:10**x,range(-5,3,1))
eta = map(lambda x:x/100.0,range(1,100,10))
# model = LogisticRegressionCV(Cs=range(1, 100), penalty='l2', solver='liblinear', n_jobs=-1)

# model = MLPClassifier(hidden_layer_sizes=(100,50))
# model = GridSearchCV(SVC(),n_jobs=16, scoring = 'f1',  cv = 3, param_grid={'kernel':['rbf', 'poly','sigmoid'],'C':range(1,100,10), 'class_weight':['balanced']})
# model = GridSearchCV(RandomForestClassifier(), cv = 5, n_jobs=-1, param_grid={'n_estimators':range(300,500,50),'class_weight':['balanced']})

# model = GridSearchCV(LogisticRegression(), n_jobs=-1, verbose=5,param_grid={'solver':['lbfgs'],'C':range(100,1000,100),'class_weight':['balanced']}, scoring='f1')
# model = GridSearchCV(LinearSVC(), n_jobs=-1, verbose=5,param_grid={'C':range(100,1000,100),'class_weight':['balanced']}, scoring='f1')
model = GridSearchCV(GradientBoostingClassifier(), cv = 5, n_jobs=36, verbose=5, param_grid={'n_estimators':range(1600,1700,200),'learning_rate':[0.1,0.15,0.2]})
# model = GridSearchCV(AdaBoostClassifier(), cv = 5, n_jobs=36, verbose=5, param_grid={'n_estimators':range(800,1500,200),'learning_rate':eta})
#
model.fit(x_train, y_train)
try:
    print "Grid best scores = ",model.best_score_
    print "Grid best param = ",model.best_params_
except:
    print cross_val_score(model,x_train, y_train, scoring='f1', cv=10)


#
df = pd.read_csv('/Users/bjayaram/Documents/Kaggle/criminal/criminal_test.csv')
label_cols = u'Criminal'
data_columns=list(df.columns)
data_columns.remove('PERID')
x_test = df[data_columns].values
x_test = mms.transform(x_test)
x_test = chi2.transform(x_test)
x_test = pca.transform(x_test)
y_test = model.predict(x_test)

res= pd.DataFrame()
res['PERID']=df['PERID']
res['Criminal']=y_test
res.to_csv('/Users/bjayaram/Documents/Kaggle/criminal/results.csv', index=None)