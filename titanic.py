import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.feature_selection import *
from sklearn.preprocessing import LabelEncoder as LE, MinMaxScaler as MMS
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.kernel_ridge import *
from sklearn import tree
from sklearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import *
from sklearn.linear_model import *
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN

def generate_cross_val_score(model, x, y, cv=10):
    kf = KFold(n_splits=cv)
    scores = []
    for train, test in kf.split(x,y):

        model.fit(x[train], y[train])
        y1 = map(int,map(round,model.predict(x[test])))
        score = f1_score(y[test],y1)
        scores.append(score)

    return sum(scores)/len(scores)


def encode_data(x_train, numerical_columns, d, enc_create=True):
    cat_cols = x_train.drop(numerical_cols, axis=1).fillna('NA')

    label_enc_map = {}

    if enc_create:
        enc = ()
        cat_data = cat_cols.apply(lambda x: d[x.name].fit_transform(x))
    else:
        cat_data = cat_cols.apply(lambda x: d[x.name].transform(x))

    df = pd.concat([x_train[numerical_cols].fillna(0), cat_data], axis=1)
    return df.values


d = defaultdict(LE)

df = pd.read_csv('/Users/bjayaram/Documents/Kaggle/titanic/train.csv')
data_cols = [u'Pclass', u'Sex', u"Age", 'SibSp', 'Parch', 'Fare', 'Embarked']
label_cols = u'Survived'
y_train = df[label_cols]
x_train = df[data_cols]
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
mms=MMS()
# smote = SMOTE(ratio='minority', k_neighbors= 5, kind='svm', n_jobs=16)

x1_train = mms.fit_transform(encode_data(x_train, numerical_cols, d))

pca = SelectKBest(chi2, k=5)
y1_train = y_train.values
print x1_train.shape
# x1_train = pca.fit_transform(x1_train, y1_train)
print x1_train.shape

# model1 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(1000,2000,500,100), learning_rate= 'adaptive')

C=map(lambda x:10**x,range(0,10))
max = -1


model = SVC(C=11,class_weight='balanced')
model = GridSearchCV(SVC(), cv =10, n_jobs=24, param_grid={'gamma':map(lambda x:10**x, range(-100,100,10)),'kernel':['rbf', 'poly','sigmoid'],'C':range(1,1000,10), 'class_weight':['balanced']})
# model = GridSearchCV(RandomForestClassifier(), cv =5, n_jobs=24, param_grid={'n_estimators':range(100,1000,200),'class_weight':['balanced']})

model.fit(x1_train, y1_train)
print 'Best',model.best_score_
print 'All', model.cv_results_

df = pd.read_csv('/Users/bjayaram/Documents/Kaggle/titanic/test.csv')
x_test = df[data_cols]
x1_test = mms.transform(encode_data(x_test, numerical_cols, d, enc_create=False))
# x1_test = pca.transform(x1_test)
y1_test = model.predict(x1_test)

df['Survived'] = y1_test
results = df[['PassengerId', 'Survived']]
results.to_csv('/Users/bjayaram/Documents/Kaggle/titanic/results.csv', index=None)
