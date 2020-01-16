# UCSC CSE 143 Introduction to Natural Language Processing
# Assignment 1: Language Modeling and Smoothing
# Rebecca Dorn, [other names]
# cruzid: radorn, [other cruzid]

def main():
	# Get the training data
	train = [] # Initialize empty list for training data
	with open('A1-Data/1b_benchmark.train.tokens', 'r') as filehandle: # Open train data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			train.append(current_place) # Append this sentance to our list of train data
	unigram_count_vec = unigram_model(train) # Get probability distribution for unigram
	print("Generated Unigram Distribution") # Tell user where we are inj program

	# Get the dev data
	dev = [] # Initialize empty list for development
	with open('A1-Data/1b_benchmark.dev.tokens', 'r') as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			dev.append(current_place) # Append this sentance to our list of dev data
	yhat = unigram_predict(unigram_count_vec,dev) # Predict sentence likelihoods via unigram
	print("Calculated predictions for Unigram") # Update user
	print(yhat)

# Get list of words separated by spaces
def get_tokens(sentence): # Return a list of normalized words
	normalized = [] # intialize normalized works as empty
	for word in sentence.split(" "): # Split sentence into words via regex
		normalized.append(word) # place word in list
	return normalized # return our list of normalized words

# Extract dictionary of unigram vocabulary and counts of those unigrams
def unigram_model(train):
	unigram_corpus = {} # Initialize our dictionary as empty
	for instance in train: # For each sentance in train
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
	print("Check corpus cardinality: 26602 == ",len(freq_corpus),"?") # Professor told us this should be 26602
	return freq_corpus

# Predict a sentence's probability via previously extracted vocabulary
def unigram_predict(vocab,test):
	yhat = [] # Initialize our vector of predictions as empty
	for instance in test: # for each sentence in our test data
		tokens = get_tokens(instance) # split the sentence into unigrams
		product = 1 # Initialize product as 1 b/c won't affect multiplication
		for word in tokens: # go through unigrams in this test sentence
			if word in vocab.keys(): # if this word is in our dictionary
				count = float(vocab[word]) / len(vocab) # p(uni_i) = c(uni_i in train)/c(uni's in vocab)
			else: # else, map this rare word to 'UNK'
				count = float(vocab['UNK']) / len(vocab) # p('UNK') = c('UNK' in train)/c(uni's in vocab)
			product = float(product) * count # add this probability to our running product
		yhat.append(product) # append this instance's probability to our predictions
	return yhat # return predicted probabilities
			
if __name__ == "__main__":
    main()