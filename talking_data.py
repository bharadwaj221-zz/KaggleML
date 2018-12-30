import pandas as pd,re
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.kernel_ridge import *
from sklearn import tree
from sklearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC, SVC, OneClassSVM
from sklearn.model_selection import *
from sklearn.linear_model import *
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from encoder import Encoder

def convert_val(val):
    return (val+1)/2

def parse_dates(date):
    new_date=date.replace(' ','')
    new_date = new_date.replace(':', '')
    new_date = new_date.replace('-', '')
    return int(new_date)

d = defaultdict(LE)

sample_path='/Users/bjayaram/Documents/Kaggle/talkingdata/mnt/ssd/kaggle-talkingdata2/competition_files/train_sample.csv'
path='/Users/bjayaram/Documents/Kaggle/talkingdata/mnt 2/ssd/kaggle-talkingdata2/competition_files/train.csv'

df = pd.read_csv(sample_path)
df = df.loc[df['is_attributed']==1]
df['click_time']= df['click_time'].map(parse_dates)
data_cols = list(df.columns)
data_cols.remove('attributed_time')
data_cols.remove('is_attributed')

label_cols = u'is_attributed'

print 'Data generated'
y = df[label_cols].values
X = df[data_cols].values
# x_train, y_train = SMOTE().fit_sample(X, y)
print 'SMOTE Sample generated'
model = OneClassSVM()
model.fit(X)



df = pd.read_csv('/Users/bjayaram/Documents/Kaggle/talkingdata/test.csv')
df['click_time']= df['click_time'].map(parse_dates)
print 'Test data generated'


x_test = df[data_cols].values
y1_test=model.predict(x_test)

df['is_attributed'] = map(convert_val,y1_test)
results = df[['click_id', 'is_attributed']]
results.to_csv('/Users/bjayaram/Documents/Kaggle/talkingdata/results.csv', index=None)