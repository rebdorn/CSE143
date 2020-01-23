# UCSC CSE 143 Introduction to Natural Language Processing
# Assignment 1: Language Modeling and Smoothing
# Erica Fong, David Nguyen and Rebecca Dorn
# cruzid: radorn, [other cruzid]
import math

def main():
	# Get the training data
	train = [] # Initialize empty list for training data
	with open('A1-Data/1b_benchmark.train.tokens', 'r') as filehandle: # Open train data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			train.append(current_place) # Append this sentance to our list of train data

	# Get probability distribution for unigram, bigram and trigrams using the training data
	unigram_count_vec, unigram_count = unigram_model(train)
	bigram_count_vec, bigram_count = bigram_model(train) 
	trigram_count_vec, trigram_count = trigram_model(train)
	print("Generated Uni/Bi/Trigram Distributions") # Tell user where we are inj program

	# Get the dev data
	dev = [] # Initialize empty list for development
	with open('A1-Data/1b_benchmark.dev.tokens', 'r') as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			dev.append(current_place) # Append this sentance to our list of dev data

	# WHILE TESTING only do 10 instances
	dev = dev[:10]
	yhat_unigram = unigram_predict(unigram_count_vec,dev,unigram_count)
	yhat_bigram = bigram_predict(bigram_count_vec,dev,bigram_count)
	yhat_trigram = trigram_predict(trigram_count_vec,dev,trigram_count)
	print(yhat_unigram)
	print(yhat_bigram)
	print(yhat_trigram)

	sentence1_perplex = perplexity(dev, 1, unigram_count_vec, unigram_count)
	print("Sentence 1 perplexity: ",sentence1_perplex)

	
	lamb_1 = 0.33
	lamb_2 = 0.33
	lamb_3 = 0.34
	what = interpolate(dev, unigram_count_vec, bigram_count_vec, trigram_count_vec, lamb_1, lamb_2, lamb_3, unigram_count, bigram_count, trigram_count)
	print(what)
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
	unigram_corpus = {} # Initialize our dictionary as empty
	for instance in train: # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		for word in tokens: # For each word in the sentence
			if word not in unigram_corpus.keys(): # if this is a new unigram
				unigram_corpus[word] = 1 # Initialize it's sightings to 1
			else:
				unigram_corpus[word] += 1 # Increment our counter by 1
	# Create new list of list with frequent unigrams, mapping rare unigrams to 'UNK'
	total_count = 0 # initialize total number of unigrams in train to be 0
	freq_corpus = [['UNK',0]] # Initialize the number of rare words to 0
	for item in unigram_corpus.items(): # Go through each word and it's count
		unigram, count = item
		if count < 3: # If it was not sighted enough
			freq_corpus[0][1] += count # Don't add to new dictionary, increment 'UNK' counter
		else: # Else, we saw it enough to not be considered a rare word
			freq_corpus.append([unigram,count]) # Put this unigram and its count into our new dictionary
		total_count += count
	return freq_corpus, total_count

def bigram_model(train):
	bigram_corpus = {} # Initialize our dictionary as empty
	for j, instance in enumerate(train): # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		for i in range(0,len(tokens)-1): # for each bigram
			bigram = (tokens[i], tokens[i+1]) # set bigram, instead of tokens[i:i+2]
			if bigram not in bigram_corpus.keys(): # if this is a new bigram
				bigram_corpus[bigram] = 1 # initialize our bigram count to 1
			else:
				bigram_corpus[bigram] += 1 # increment bigram counter
	# go through bigram corpus, change UNKs
	total_count = 0
	freq_corpus = [['UNK','UNK',0]] # initialize our list of lists to be UNK
	for item in bigram_corpus.items():
		bigram, count = item
		bigram0, bigram1 = bigram
		if count < 3:
			freq_corpus[0][2] += count
		else:
			freq_corpus.append([bigram0,bigram1,count])
		total_count += count
	return freq_corpus, total_count

def trigram_model(train):
	trigram_corpus = {} # Initialize our dictionary as empty
	for j, instance in enumerate(train): # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		for i in range(0,len(tokens)-2): # for each trigram
			trigram = (tokens[i], tokens[i+1], tokens[i+2]) # set trigram 
			if trigram not in trigram_corpus.keys(): # if this is a new trigram
				trigram_corpus[trigram] = 1 # initialize our trigram count to 1
			else:
				trigram_corpus[trigram] += 1 # increment trigram counter
	# go through trigram corpus, change UNKs
	freq_corpus = [['UNK','UNK','UNK',0]] # initialize our list of lists to be UNK
	total_count = 0
	for item in trigram_corpus.items():  
		trigram, count = item
		trigram0, trigram1, trigram2 = trigram
		if count < 3:
			freq_corpus[0][3] += 1
		else:
			freq_corpus.append([trigram0,trigram1,trigram2,count])
		total_count += count
	return (freq_corpus, total_count)

# Predict a sentence's probability via previously extracted vocabulary
def unigram_predict(vocab,test,unigram_count):
	yhat = [] # Initialize our vector of predictions as empty
	for instance in test: # for each sentence in our test data
		print("instance is ", instance)
		tokens = get_tokens(instance) # split the sentence into unigrams
		product = 1 # Initialize product as count of stop
		for word in tokens: # go through unigrams in this test sentence
			print("word is ", instance)
			found = 0
			for unigram in vocab:
				if unigram[0] == word:
					count = float(unigram[1])/unigram_count
					found = 1
			if found == 0: # if we still havent found it
				count = float(unigram[0][1])/unigram_count # 'UNK' count
			product = float(product)*count # add this probability to our running product
		yhat.append(product / len(vocab)) # append this instance's probability to our predictions
	return yhat # return predicted probabilities

def bigram_predict(vocab,test,bigram_count):
	# Generate proabilities for sentences
	yhat = [] # initialize yhat as empty
	for i, instance in enumerate(test): # for each sentence
		tokens = get_tokens(instance) # split instance into list by " "
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		prob_sentence = 1 # initialize product variable, 1 * anything nonzero = 1
		for i in range(0,len(tokens)-1): # for each bigram in this sentence
			bigram = [tokens[i],tokens[i+1]] # set bigram
			count_similar = 0 # initialize similarity count to 0
			count_match = 0
			for vocab_instance in vocab: # for each bigram in train
				if vocab_instance[:2] == bigram:
					count_match = vocab_instance[2]
					count_similar += vocab_instance[2]
				elif instance[0] == tokens[i]: # only will hit this clause if not full match
					count_similar += vocab_instance[2]
			if count_match == 0: # this is a new bigram
				count_similar = bigram_count # number of bigrams
				count_match = vocab[0][2] # count of 'UNKS'
			prob_word = float(count_match) / count_similar
			prob_sentence = prob_sentence * prob_word
		yhat.append(prob_sentence)
	return yhat

def trigram_predict(vocab,test,trigram_count):
	# Generate proabilities for sentences
	print(trigram_count)
	yhat = [] # initialize yhat as empty
	for i, instance in enumerate(test): # for each sentence
		tokens = get_tokens(instance) # split instance into list by " "
		tokens.insert(0,'<START>') # prepend <START> token to each sentence
		tokens.append('<STOP>') # append <STOP> token to each sentence
		prob_sentence = 1 # initialize product variable, 1 * anything nonzero = 1
		for i in range(0,len(tokens)-2): 
			trigram = [tokens[i],tokens[i+1],tokens[i+2]] # set trigram
			count_match = 0 # initialize match count to 0
			count_similar = 0 # initialize similarity count to 0
<<<<<<< HEAD
			for instance in vocab: # for each trigram in train
				if instance == trigram: # if equal to trigram in test, increase count_similar
					count_match = instance[3]
					count_similar += instance[3]
				elif instance[0] == tokens[i] and instance[1] == tokens[i+1]: # only will hit this clause if not full match
					count_similar += instance[3]
			if count_match == 0: # this is a new trigram
=======
			for vocab_instance in vocab:
				if vocab_instance[:3] == trigram:
					count_match = vocab_instance[3]
					count_similar += vocab_instance[3]
				elif vocab_instance[0] == tokens[i] and vocab_instance[1] == tokens[i+1]: # only will hit this clause if not full match
					count_similar += vocab_instance[3]
			if count_match ==0: # this is a new bigram
>>>>>>> fa645713952e0bce5c81107763ea788c4d2f23ce
				count_similar = trigram_count # number of trigrams
				count_match = vocab[0][3] # number of 'UNK'
			prob_word = float(count_match) / count_similar
			prob_sentence = prob_sentence * prob_word
		yhat.append(prob_sentence)
	return yhat

def perplexity(instances, ngrammodel, ngramvocab, ngramcount):
	perplexities = []
	for instance in instances:
		tokens = get_tokens(instance) # split instance into list by " "
		tokens.append('<STOP>') # add the stop token
		logprob_sum = 0
		for word in tokens:
			if ngrammodel == 1: # if we're evaluating p() on unigrams
				foundword = 0
				for unigram in ngramvocab: # check if we know this word
					if word == unigram[0]: # we found this word
						curr_prob = float(unigram[1])/ngramcount
						foundword = 1
				if foundword == 0: # map to UNK
					curr_prob = float(ngramvocab[0][1])/ngramcount
				logprob_sum += float(math.log(curr_prob,2))
				print(math.log(curr_prob,2))
		l = float((-1/len(tokens))) * float(logprob_sum)
		toappend = 2 ** l
		perplexities.append(toappend)
	return perplexities


def interpolate(test, unigram, bigram, trigram, lamb_1, lamb_2, lamb_3, unigram_count, bigram_count, trigram_count):
	yhat_interpolated = []
	# piazza 37
	for instance in test: # for each training instance
		init = 0 # signal we're at the beginning of a new sentence
		tokens = get_tokens(instance) 
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		for i in range(0,len(tokens)-2):
			current_unigram = tokens[i]
			current_bigram = (tokens[i],tokens[i+1])
			current_trigram = (tokens[i],tokens[i+1],tokens[i+2])
			uni_prediction = unigram_predict(unigram,instance,unigram_count)
			bi_prediction = bigram_predict(bigram,instance,bigram_count)
			tri_prediction = trigram_predict(trigram,instance,trigram_count)
			if init == 0: # if we're on the first word, only unigrams
				print("Token is ", current_unigram)
				print(unigram_predict(unigram, current_unigram ,unigram_count))
				theta_uni = lamb_1 * float(unigram_predict(unigram, ,unigram_count))
				theta_bi = 1 # TODO: how do we do the first couple unigrams?
				theta_tri = 1
				init += 1 # Increase location for next time
			elif init == 1: # if we're on the second word, only uni and bigrams
				theta_uni = lamb_1 * unigram_predict(unigram,instance,unigram_count)
				theta_bi = lamb_2 * bigram_predict(bigram,instance,bigram_count)
				theta_tri = 1
				init += 1
			else: # else, we incorporate uni, bi and trigrams
				theta_uni = lamb_1 * unigram_predict(unigram,instance,unigram_count)
				theta_bi = lamb_2 * bigram_predict(bigram,instance,bigram_count)
				theta_tri = lamb_3 * trigram_predict(trigram,instance,trigram_count)
		theta_smoothed = theta_uni + theta_bi + theta_tri
		yhat_interpolated.append(theta_smoothed)
	return yhat_interpolated

if __name__ == "__main__":
    main()
