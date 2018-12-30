import pandas as pd,re, pickle
import numpy as np
from sklearn.feature_extraction import *
from sklearn.feature_extraction.text import *
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
from url_utils import URLUtils
import sys
sys.setde
df = pd.read_csv('/Users/bjayaram/Documents/Kaggle/email_campaign/train_HFxi8kT/train.csv')
cdf = pd.read_csv('/Users/bjayaram/Documents/Kaggle/email_campaign/train_HFxi8kT/campaign_data.csv')
# cdf['text']=cdf['communication_type']+" "+get_text(cdf['email_url'])

cdf['url_text'] = cdf['email_url'].map(URLUtils.parse_urls)
cdf['text'] = cdf['email_body']+' '+cdf['url_text']
pickle.dump(cdf,open('/Users/bjayaram/Documents/Kaggle/email_campaign/cdf.p','w'))
vec = TfidfVectorizer(ngram_range=(1,2))
vec.fit_transform(cdf['url_text'].values).dense


def parse_dates(date):
    parts=date.split(" ")
    date_parts = parts[0].split("-")
    time_parts = parts[1].split(':')
    new_date =date_parts[2]+date_parts[1]+date_parts[0]+time_parts[0]+time_parts[1]

    return int(new_date)