# UCSC CSE 143 Introduction to Natural Language Processing
# Assignment 1: Language Modeling and Smoothing
# Rebecca Dorn, [other names]
# cruzid: radorn, [other cruzid]

#include <string.h>

def main():
	# Get the training data
	train = [] # Initialize empty list for training data
	with open('A1-Data/1b_benchmark.train.tokens', 'r') as filehandle: # Open train data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			train.append(current_place) # Append this sentance to our list of train data
	
	print("Running ngram_model...\n")
	unigram_count_vec = ngram_model(train,1) # Get probability distribution for unigram
	bigram_count_vec = ngram_model(train,2) # Get probability distribution for bigram
	trigram_count_vec = ngram_model(train,3) # Get probability distribution for trigram
	#print("Trigram count vector sample: ", trigram_count_vec[:5])

	# Get the dev data
	dev = [] # Initialize empty list for development
	with open('A1-Data/1b_benchmark.dev.tokens', 'r') as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			dev.append(current_place) # Append this sentance to our list of dev data

	print("Running ngram_predict...")
	unigram_predictions = ngram_predict(unigram_count_vec,dev, 1) # Predict sentence likelihoods via unigram
	print("Predicted unigrams.")
	bigram_predictions = ngram_predict(bigram_count_vec, dev, 2)
	print("Predicted bigrams: ", bigram_predictions[:5])
	trigram_predictions = ngram_predict(trigram_count_vec, dev, 3)
	print("Calculated predictions for uni, bi and trigrams.") # Update user
	print("Sample trigram predictions: ",trigram_predictions[:10])

	yhat_interpolated = interpolate(dev, unigram_count_vec, bigram_count_vec, trigram_count_vec, .33, .33, .34)
	#print("Calculated interpolated data", yhat_interpolated[0:5])

# Get list of words separated by spaces
def get_tokens(sentence): # Return a list of normalized words
	normalized = [] # intialize normalized works as empty
	for word in sentence.split(" "): # Split sentence into words via regex
		normalized.append(word) # place word in list
	return normalized # return our list of normalized words

# Extract dictionary of n-gram vocabulary and counts of those n-grams
def ngram_model(train, n):
	ngram_corpus = {} # Initialize our dictionary as empty
	for instance in train: # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		tokens.append('<STOP>')	# append STOP to the end of each sentence
		for i in range(0, len(tokens)): # For each word in the sentence
			w = tokens[i:i+n]
			if(w == []): # out of bound slices return [], so ignore
			   break
			word = tuple(w)
			#print(word)
			if word not in ngram_corpus.keys(): # if this is a new unigram
				ngram_corpus[word] = 1 # Initialize it's sightings to 1
			else:
				ngram_corpus[word] += 1 # Increment our counter by 1
	print("Counted n-grams in building for grams length ",n)
	#print( "Stop count is", ngram_corpus['<STOP>'])
	# Create new dictionary with frequent unigrams, mapping rare unigrams to 'UNK'
	freq_corpus = {'UNK':0} # Initialize the number of rare words to 0
	for ngram in ngram_corpus.keys(): # Go through each word and it's count
		if ngram_corpus[ngram] < 3: # If it was not sighted enough
			freq_corpus['UNK'] += 1 # Don't add to new dictionary, increment 'UNK' counter
		else: # Else, we saw it enough to not be considered a rare word
			freq_corpus[ngram] = ngram_corpus[ngram] # Put this unigram and its count into our new dictionary
	if(n == 1):
		print("Check corpus cardinality: 26602 == ",len(freq_corpus),"?") # Professor told us this should be 26602
	return freq_corpus

# Predict a sentence's probability via previously extracted vocabulary
def ngram_predict(vocab,test,n):
	yhat = [] # Initialize our vector of predictions as empty
	for instance in test: # for each sentence in our test data
		tokens = get_tokens(instance) # split the sentence into unigrams
		product = 1 # Initialize product as 1 b/c won't affect multiplication
		for i in range(0,len(tokens)-n-1): # go through ngrams in this test sentence
			w = tokens[i:i+n]
			word = tuple(w)
			if word in vocab.keys(): # if this word is in our dictionary
				if n == 1:
					count = float(vocab[word]) / len(vocab) # p(uni_i) = c(uni_i in train)/c(uni's in vocab)
				elif n == 2:
					# get context
					context = 0
					for bigram in vocab.keys():
						if bigram[0] == word[0]:
							context += 1
					count = float(vocab[word]) / context
				elif n == 3:
					# get context
					context = 0
					for trigram in vocab.keys():
						if trigram[:2] == word[:2]:
							context += 1
					count = float(vocab[word]) / context
			else: # else, map this rare word to 'UNK'
				count = float(vocab['UNK']) / len(vocab) # p('UNK') = c('UNK' in train)/c(uni's in vocab)
			product = float(product) * count # add this probability to our running product
		yhat.append(product) # append this instance's probability to our predictions
	return yhat # return predicted probabilities

def interpolate(test, unigram, bigram, trigram, lamb_1, lamb_2, lamb_3):
	yhat_interpolated = []
	for instance in test: # for each training instance
		init = 0 # signal we're at the beginning of a new sentence
		tokens = get_tokens(instance) 
		for word in tokens:
			if init == 0: # if we're on the first word, only unigrams
				theta_uni = float(lamb_1) * ngram_predict(unigram,[instance],1)[0]
				theta_bi = float(lamb_2) # TODO: how do we do the first couple unigrams?
				theta_tri = float(lamb_3)
				init += 1 # Increase location for next time
			elif init == 1: # if we're on the second word, only uni and bigrams
				theta_uni = float(lamb_1) * ngram_predict(unigram,[instance],1)[0]
				theta_bi = float(lamb_2) * ngram_predict(bigram,[instance],2)[0]
				theta_tri = float(lamb_3)
				init += 1
			else: # else, we incorporate uni, bi and trigrams
				theta_uni = float(lamb_1) * ngram_predict(unigram,[instance],1)[0]
				theta_bi = float(lamb_2) * ngram_predict(bigram,[instance],2)[0]
				theta_tri = float(lamb_3) * ngram_predict(trigram,[instance],3)[0]
				#if init == 2:
					#print("Unigram alone prediction: ", ngram_predict(unigram,[instance],1)[0])
					#print("Bigram alone prediction: ", ngram_predict(bigram,[instance],2)[0])
					#print("Trigram alone prediction: ", ngram_predict(trigram,[instance],3)[0])
					#init += 1
		theta_smoothed = theta_uni + theta_bi + theta_tri
		yhat_interpolated.append(theta_smoothed)
	return yhat_interpolated
			
if __name__ == "__main__":
    main()
