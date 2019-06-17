import pandas as pd

train = pd.DataFrame({'col_1':[1,2,3], 'col_2':[3,2,1]})
test = pd.DataFrame({'col_1':[4,5,6], 'col_2':[5,6,4]})
# Save preprocessed frames
# 1. Pickle
# Write:
#train.to_pickle('../../../dat/prep/train_GRU.pkl')
#test.to_pickle('../../../dat/prep/test_GRU.pkl')

# Read:
#train = pd.read_pickle('../../../dat/prep/train_GRU.pkl')
#test = pd.read_pickle('../../../dat/prep/train_GRU.pkl')

# 2. HDF5
# Write
store = pd.HDFStore('../../../dat/prep/GRU_dat.h5')
''' .append will add to what is already there in the h5 file
store.append('train', train)
store.append('test', test)
'''
store.put('train', train)
store.put('test', test)
print(train.head())
print(test.head())


del store, train, test

# Read
store = pd.HDFStore('../../../dat/prep/GRU_dat.h5')
train = store.get('train')
test = store.get('test')
print(train)
print(test)

# 3. Numpy arrays