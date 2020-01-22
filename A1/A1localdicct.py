# UCSC CSE 143 Introduction to Natural Language Processing
# Assignment 1: Language Modeling and Smoothing
# Rebecca Dorn, [other names]
# cruzid: radorn, [other cruzid]
import pandas as pd
import numpy as np

def main():
	# Get the training data
	train = [] # Initialize empty list for training data
	with open('CSE143/A1/A1-Data/1b_benchmark.train.tokens', 'r') as filehandle: # Open train data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			train.append(current_place) # Append this sentance to our list of train data
	bigram_count_vec = bigram_model(train) # Get probability distribution for unigram
	print("Generated Bigram Distribution") # Tell user where we are inj program

	# Get the dev data
	dev = [] # Initialize empty list for development
	with open('CSE143/A1/A1-Data/1b_benchmark.dev.tokens', 'r') as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			dev.append(current_place) # Append this sentance to our list of dev data
	yhat_bigram = bigram_predict(bigram_count_vec,dev)
	print(yhat_bigram)
	
	#yhat = unigram_predict(unigram_count_vec,dev) # Predict sentence likelihoods via unigram
	#print("Calculated predictions for Unigram", yhat) # Update user

# Get list of words separated by spaces
def get_tokens(sentence): # Return a list of normalized words
	normalized = [] # intialize normalized works as empty
	for word in sentence.split(" "): # Split sentence into words via regex
		normalized.append(word) # place word in list
	return normalized # return our list of normalized words

# Extract dictionary of unigram vocabulary and counts of those unigrams
def unigram_model(train):
	num_stop = 0
	unigram_corpus = {} # Initialize our dictionary as empty
	for instance in train: # For each sentance in train
		num_stop += 1
		tokens = get_tokens(instance) # split sentences into words
		for word in tokens: # For each word in the sentence
			if word not in unigram_corpus.keys(): # if this is a new unigram
				unigram_corpus[word] = 1 # Initialize it's sightings to 1
			else:
				unigram_corpus[word] += 1 # Increment our counter by 1
	# Create new dictionary with frequent unigrams, mapping rare unigrams to 'UNK'
	freq_corpus = {'UNK':0} # Initialize the number of rare words to 0
	for unigram in unigram_corpus.keys(): # Go through each word and it's count
		if unigram_corpus[unigram] < 3: # If it was not sighted enough
			freq_corpus['UNK'] += 1 # Don't add to new dictionary, increment 'UNK' counter
		else: # Else, we saw it enough to not be considered a rare word
			freq_corpus[unigram] = unigram_corpus[unigram] # Put this unigram and its count into our new dictionary
	freq_corpus['<STOP>'] = num_stop
	print("Check corpus cardinality: 26602 == ",len(freq_corpus),"?") # Professor told us this should be 26602
	return freq_corpus

def bigram_model(train):
	bigram_corpus = {} # Initialize our dictionary as empty
	#bigram_corpus[('<START>', tokens[0])] 
	for j, instance in enumerate(train): # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		# print("Iteration ",j,"out of ",len(train))
		for i in range(-1,len(tokens)): # for each bigram
			if i == -1: # if we need to include the <START> token
				bigram = ('<START>',tokens[0]) # set bigram accordingly
			elif i+1 == len(tokens): # if the second unigram is <STOP>
				bigram = (tokens[i],'<STOP>') # set bigram accordingly
			else:
				bigram = (tokens[i], tokens[i+1]) # set bigram, instead of tokens[i:i+2]
			if bigram not in bigram_corpus.keys(): # if this is a new bigram
				bigram_corpus[bigram] = 1 # initialize our bigram count to 1
			else:
				bigram_corpus[bigram] += 1 # increment bigram counter
	# go through bigram corpus, change UNKs

	freq_corpus = [['UNK','UNK',0]] # initialize our list of lists to be UNK
	for item in bigram_corpus.items():
		bigram, count = item
		bigram0, bigram1 = bigram
		if count < 3:
			freq_corpus[0][2] += 1
		else:
			freq_corpus.append([bigram0,bigram1,count])
	return freq_corpus

# Predict a sentence's probability via previously extracted vocabulary
def unigram_predict(vocab,test):
	yhat = [] # Initialize our vector of predictions as empty
	for instance in test: # for each sentence in our test data
		tokens = get_tokens(instance) # split the sentence into unigrams
		product = vocab['<STOP>']/len(vocab) # Initialize product as count of stop
		for word in tokens: # go through unigrams in this test sentence
			if word in vocab.keys(): # if this word is in our dictionary
				count = float(vocab[word])/len(vocab) # p(uni_i) = c(uni_i in train)/c(uni's in vocab)
			else: # else, map this rare word to 'UNK'
				count = float(vocab['UNK'])/len(vocab) # p('UNK') = c('UNK' in train)/c(uni's in vocab)
			product = float(product)*count # add this probability to our running product
		yhat.append(product / len(vocab)) # append this instance's probability to our predictions
	return yhat # return predicted probabilities

def bigram_predict(vocab,test):

	# Generate proabilities for sentences
	yhat = [] # initialize yhat as empty
	for i, instance in enumerate(test): # for each sentence
		print("Iteration ",i,"out of ",len(test))
		tokens = get_tokens(instance) # split instance into list by " "
		prob_sentence = 1 # initialize product variable, 1 * anything nonzero = 1
		for i in range(-1,len(tokens)):
			if i == -1: # if we need to signal start
				bigram = ('<START>', tokens[0]) # set bigram accordingly
			elif i == (len(tokens)-1): # elif were at the end
				bigram = (tokens[i],'<STOP>') # set bigram accordingly
			else:
				bigram = (tokens[i],tokens[i+1]) # set bigram
			bigram0, bigram1 = bigram
			count_similar = 0 # initialize similarity count to 0
			for instance in vocab:
				# bi0 is instance[0], bi1 is instance[1] and count is instance[2]
				if instance[0] == bigram0 and instance[1] == bigram1:
					count_match = instance[2]
				elif instance[0] == bigram0: # only will hit this clause if not full match
					count_similar += instance[2]
				if count_similar == 0: # this is a new bigram
					count_similar = len(vocab) # number of bigrams
					count_match = vocab[0][2] # number of 'UNK'
			prob_word = float(count_match) / count_similar
			prob_sentence = prob_sentence * prob_word
		yhat.append(prob_sentence)
	return yhat

def interpolate(test, unigram, bigram, trigam, lamb_1, lamb_2, lamb_3):
	yhat_interpolated = []
	for instance in test: # for each training instance
		init = 0 # signal we're at the beginning of a new sentence
		tokens = get_tokens(instance) 
		for word in tokens:
			if init == 0: # if we're on the first word, only unigrams
				theta_uni = lamb_1 * unigram_predict(unigram,instance)
				theta_bi = 0 # TODO: how do we do the first couple unigrams?
				init += 1 # Increase location for next time
			elif init == 1: # if we're on the second word, only uni and bigrams
				theta_uni = lamb_1 * unigram_predict(unigram,instance)
				theta_bi = lamb_2 * bigram_predict(bigram,instance)
				theta_tri = 0
				init += 1
			else: # else, we incorporate uni, bi and trigrams
				theta_uni = lamb_1 * unigram_predict(unigram,instance)
				theta_bi = lamb_2 * bigram_predict(bigram,instance)
				theta_tri = lamb_3 * trigram_predict(trigram,instance)
		theta_smoothed = theta_uni + theta_bi + theta_tri
		yhat_interpolated.append(theta_smoothed)
	return yhat_interpolated

if __name__ == "__main__":
    main()
