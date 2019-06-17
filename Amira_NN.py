'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function
import numpy as np
from sklearn.tests.test_cross_validation import test_train_test_split
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
import random


# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
filter_length = 2
nb_filter = 10
pool_length = 2

# LSTM
lstm_output_size = 5

# Training
batch_size = 30
nb_epoch = 2
test__train_split_ratio = 0.1# 0.1 means 10% test

def corpus_to_indices(text):
	#words_map = build_words_map(text)
		
	#text_to_indices(text, words_map)

	# The vocabulary map
	words_map = {}

	# Index of words
	index = 0
	
	# Initialize the output list
	text_indices = []
	maxlen = 0
	# Loop line by line
	for line in text:
		# Split into words
		line_words = line.split()

		if len(line_words) > maxlen:
			maxlen = len(line_words) 
		# Initialize the line_indices
		line_indices = []
		# Loop word by word
		for word in line_words:
			# Store the word once in the wordMap
			if not word in words_map:
				words_map[word] = index
				# Increment the index for the next word
				index += 1

			# Add the index to the line_indices
			line_indices.append(words_map[word])

		# Add the line_indices to the output list
		text_indices.append(line_indices)


	return text_indices, len(words_map), maxlen

def load_data(data_file_name, annotation_file_name):

	# Load text
	f_data = open(data_file_name, 'r', encoding='UTF8')
	
	text = []
	for line in f_data:
		text.append(line)
	
	
	text_indices, voc_size, maxlen = corpus_to_indices(text)
	
	# Load labels
	f_labels = open(annotation_file_name, 'r')
	
	labels = []
	for line in f_labels:
		labels.append(int(line) - 1)
	X_train, y_train, X_test, y_test = split_train_test(text_indices, labels) 	
	return X_train, y_train, X_test, y_test, voc_size, maxlen

def split_train_test(corpus, labels):
	# Randomize the dataSet
	random_data = []
	random_labels = []
	
	# Sample indices from 0..len(corpus)
	size = len(corpus)
	rand_indices = random.sample(range(size), size)
	
	
	# Insert in the final dataset N=self.datasetSize random tweets from the rawData
	for index in rand_indices:
		random_data.append(corpus[index])
		random_labels.append(labels[index])
	
	# Calculate the test set size
	test_set_size = int(test__train_split_ratio * size)
	
	# The trainSet starts from the begining until before the end by the test set size 
	train_set = random_data[0 : size - test_set_size]
	test_set  = random_data[len(train_set) : size]
	train_set_labels = random_labels[0 : size - test_set_size]
	test_set_labels  = random_labels[len(train_set) : size]	
	return train_set, train_set_labels, test_set, test_set_labels


print('Loading data...')
data_file_name = "tweets.txt"
lables_file_name = "sentiment.txt"
X_train, y_train, X_test, y_test, max_features, maxlen = load_data(data_file_name, lables_file_name)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))

model.add(LSTM(lstm_output_size))

model.add(Dense(1))
model.add(Activation('sigmoid'))

import keras.optimizers
opt = keras.optimizers.adam(0.01)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))


model.save('atb_model', overwrite=True)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)




