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


df_train = train = pd.read_csv('../../../dat/train.tsv', sep='\t')
df_test = test = pd.read_csv('../../../dat/test.tsv', sep='\t')
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


#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
from keras.preprocessing.text import Tokenizer
raw_text = np.hstack([train.category_name.str.lower(), 
                      train.item_description.str.lower(), 
                      train.name.str.lower()])

tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
train.head(3)

print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))


# Save preprocessed frames

# Write
#store = pd.HDFStore('../../../dat/prep/GRU_dat.h5')
#store.put('train', train)
#store.put('test', test)
#print(store.keys())
train.to_pickle('../../../dat/prep/train.pkl')
train.to_pickle('../../../dat/prep/test.pkl')

