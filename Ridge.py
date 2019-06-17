# coding: utf-8

# mainly forking from notebook
# https://www.kaggle.com/johnfarrell/simple-rnn-with-keras-script

# ADDED
# 5x scaled test set
# category name embedding
# some small changes like lr, decay, batch_size~

import os
import gc
import time
start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix, hstack

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 40000

def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


df_train = train = pd.read_csv('../../dat/train.tsv', sep='\t')
df_test = test = pd.read_csv('../../dat/test.tsv', sep='\t')

train['target'] = np.log1p(train['price'])


print(train.shape)
print('5 folds scaling the test_df')
print(test.shape)
test_len = test.shape[0]
def simulate_test(test):
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test
test = simulate_test(test)
print('new shape ', test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))

#HANDLE MISSING VALUES
print("Handling missing values...")
def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)

print('[{}] Finished handling missing data...'.format(time.time() - start_time))



#PROCESS CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
train.head(3)





#EXTRACT DEVELOPTMENT TEST

dtrain, dvalid = train_test_split(train, random_state=666, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)




def rmsle(y, y_pred):
    import math
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5


def eval_model(model):
    val_preds = model.predict(X_valid)
    val_preds = np.expm1(val_preds)
    
    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle


#Ridge https://www.kaggle.com/apapiu/ridge-script

nrow_train = dtrain.shape[0]
y_train = np.array(dtrain.price.values)
nrow_valid = dvalid.shape[0]
y_valid = np.array(dvalid.price.values)
merge = pd.DataFrame = pd.concat([dtrain, dvalid, df_test])

del df_train
del df_test
del dtrain
#del dvalid
del train
del test
gc.collect()

handle_missing_inplace(merge)
print('[{}] Finished to handle missing'.format(time.time() - start_time))

cutting(merge)
print('[{}] Finished to cut'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Finished to convert categorical'.format(time.time() - start_time))

cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv.fit_transform(merge['name'])
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
X_description = tv.fit_transform(merge['item_description'])
print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))


del merge
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

X_train = sparse_merge[:nrow_train]
X_valid = sparse_merge[nrow_train:nrow_train+nrow_valid]
X_test = sparse_merge[nrow_train+nrow_valid:]
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
del sparse_merge
model = Ridge(solver="sag", fit_intercept=True, random_state=205)
model.fit(X_train, y_train)
print('[{}] Finished to train ridge'.format(time.time() - start_time))

#EVLUEATE THE MODEL ON DEV TEST
v_rmsle = eval_model(model)
print('[{}] Finished predicting valid set...'.format(time.time() - start_time))

predsR = model.predict(X=X_test)
print('[{}] Finished to predict ridge'.format(time.time() - start_time))
predsR = np.expm1(predsR)
predsR = predsR
submission["price"] += predsR
submission.to_csv("submission_rnn_ridge.csv", index = False)
