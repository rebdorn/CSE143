# UCSC CSE 143 Introduction to Natural Language Processing
# Assignment 1: Language Modeling and Smoothing
# Erica Fong, David Nguyen and Rebecca Dorn
# cruzid: radorn, [other cruzid]
import math

def main():
	# Get the training data
	train = [] # Initialize empty list for training data
	with open('A1-Data/1b_benchmark.train.tokens', 'r',encoding="utf8") as filehandle: # Open train data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			train.append(current_place) # Append this sentance to our list of train data
	processed_train, known_tokens = replace(train)

	# Get the dev data
	dev = [] # Initialize empty list for development
	with open('A1-Data/1b_benchmark.dev.tokens', 'r',encoding="utf8") as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			dev.append(current_place) # Append this sentance to our list of dev data
	dev = dev[:10] # WHILE TESTING only do 10 instances
	dev = other_rep(dev,known_tokens) # Map new words in dev to UNK

	# Get the test data 
	test = [] # Initialize empty list for development
	with open('A1-Data/1b_benchmark.test.tokens', 'r',encoding="utf8") as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			test.append(current_place) # Append this sentance to our list of dev data

	# Get probability distribution for unigram, bigram and trigrams using the training data
	unigram_count_vec, unigram_count = unigram_model(processed_train)
	print(unigram_count_vec)
	bigram_count_vec, bigram_count = bigram_model(processed_train) 
	trigram_count_vec, trigram_count = trigram_model(processed_train)
	print("Generated Uni/Bi/Trigram Distributions") # Tell user where we are inj program

	# Check that we sum to 1
	print("CHECKING PROBABILITY DISTRIBUTIONS SUM TO ONE")
	uni_prob = uni_tot_prob(unigram_count_vec,unigram_count)
	bi_prob = bi_tot_prob(bigram_count_vec,bigram_count)
	tri_prob = tri_tot_prob(trigram_count_vec,trigram_count)


	# Get sentence probabilities from computed unigram, bigram and trigram distributions
	yhat_unigram = unigram_predict(unigram_count_vec,dev,unigram_count)
	yhat_bigram = bigram_predict(bigram_count_vec,dev,bigram_count)
	yhat_trigram = trigram_predict(trigram_count_vec,dev,trigram_count)
	print("Unigram predictions: ",yhat_unigram)
	print("Bigram predictions: ",yhat_bigram)
	print("Trigram predictions: ",yhat_trigram)
	
	
	# Calculate perplexity for unigram, bigram and trigram distributions with our dev set
	print("FETCHING PERPLEXITY SCORES FOR UNI, BI AND TRIGRAM MODELS.....")
	#print("ON TRAIN SET")
	#sentence1_perplex = unigram_per(unigram_count_vec, train, unigram_count)
	#sentence2_per = bigram_per(bigram_count_vec, train, bigram_count)
	#sentence3_per = trigram_per(trigram_count_vec, train, trigram_count)
	#print("Unigram perplexity:",sentence1_perplex)
	#print("Bigram perplexity: ",sentence2_per)
	#print("Trigram perplexity: ",sentence3_per)
	print("ON DEV SET")
	sentence1_per = unigram_per(unigram_count_vec, dev, unigram_count)
	sentence2_per = bigram_per(bigram_count_vec, dev, bigram_count)
	sentence3_per = trigram_per(trigram_count_vec, dev, trigram_count)
	print("Unigram perplexity:",sentence1_per)
	print("Bigram perplexity: ",sentence2_per)
	print("Trigram perplexity: ",sentence3_per)
	#print("ON TEST SET")
	#sentence1_perplex = unigram_per(unigram_count_vec, test, unigram_count)
	#sentence2_per = bigram_per(bigram_count_vec, test, bigram_count)
	#sentence3_per = trigram_per(trigram_count_vec, test, trigram_count)
	#print("Unigram perplexity:",sentence1_perplex)
	#print("Bigram perplexity: ",sentence2_per)
	#print("Trigram perplexity: ",sentence3_per)
	
	
	print("TESTING DIFFERENT GAMMAS FOR LINEAR INTERPOLATION SMOOTHIING....")
	print("Gamma1 = 0.33, Gamma2 = 0.33, Gamma3 = 0.34")
	interpolated = interpolate(dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.33,0.33,0.34, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Gamma1 = 0.7, Gamma2 = 0.2, Gamma3 = 0.1")
	interpolated = interpolate(dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.7,0.2,0.1, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Gamma1 = 0.1, Gamma2 = 0.4, Gamma3 = 0.5")
	interpolated = interpolate(dev, unigram_count_vec, bigram_count_vec, trigram_count_vec, 0.1, 0.4, 0.5, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Gamma1 = 0.1, Gamma2 = 0.3, Gamma3 = 0.6")
	interpolated = interpolate(dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.1,0.3,0.6, unigram_count, bigram_count, trigram_count)
	print(interpolated)

# Get list of words separated by spaces
def get_tokens(sentence): # Return a list of normalized words
	normalized = [] # intialize normalized works as empty
	for word in sentence.split(" "): # Split sentence into words via regex
		normalized.append(word) # place word in list
	return normalized # return our list of normalized words

# Replace rare words of train with 'UNK' token
def replace(train):
	processed =[] # Initialize set of processed data to 0
	unigram_corpus = {} # Initialize our dictionary as empty
	set = {''}  # Initialize our set of frequent tokens to 0
	token_list = ['UNK']
	for instance in train: # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		for word in tokens: # For each word in the sentence
			if word not in unigram_corpus.keys(): # if this is a new unigram
				unigram_corpus[word] = 1 # Initialize it's sightings to 1
			else:
				unigram_corpus[word] += 1 # Increment our counter by 1
	# Create new list of list with frequent unigrams, mapping rare unigrams to 'UNK'
	freq_corpus = [['UNK',0]] # Initialize the number of rare words to 0
	for item in unigram_corpus.items(): # Go through each word and it's count
		#if item[1] >= 3: # If it was not sighted enough
		freq_corpus.append([item]) # Put this unigram and its count into our new dictionary 
		token_list.append(item[0])
	# Create set of frequent words for simplicity
	#for word in freq_corpus: # for each word that's made it this far
	#	set.add(word[0]) # add this word to uor set
	# Map rare words in train data to UNK
	for instance in train: # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		sent = ''
		for i, word in enumerate(tokens): # For each word in the sentencesen
			if word not in set:
				added = 'UNK'
			else:
				added = word
			sent += added
			if i != len(tokens) - 1:
				sent += ' '
		processed.append(sent)
	print("Processed train set: ",processed)
	return processed, token_list # return our processed data

def other_rep(data, known_tokens):
	processed =[]
	for instance in data: # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		sent = ''
		for i, word in enumerate(tokens): # For each word in the sentencese
			if word not in known_tokens:
				added = 'UNK'
			else:
				added = word
			sent += added
			if i != len(tokens) - 1:
				sent += ' '
		processed.append(sent)
	return processed

# Extract dictionary of unigram vocabulary and counts of those unigrams
def unigram_model(train):
	unigram_corpus = {} # Initialize our dictionary as empty
	for instance in train: # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
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
		if unigram == 'UNK': # If it was not sighted enough
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
	freq_corpus = [] # initialize our list of lists to be empty
	for item in bigram_corpus.items():
		bigram, count = item
		bigram0, bigram1 = bigram
		freq_corpus.append([bigram0,bigram1,count])
		total_count += count
	return freq_corpus, total_count

def trigram_model(train):
	trigram_corpus = {} # Initialize our dictionary as empty
	for j, instance in enumerate(train): # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		tokens.insert(0,'<START>')
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		tokens.append('<STOP>')
		for i in range(0,len(tokens)-2): # for each trigram
			trigram = (tokens[i], tokens[i+1], tokens[i+2]) # set trigram 
			if trigram not in trigram_corpus.keys(): # if this is a new trigram
				trigram_corpus[trigram] = 1 # initialize our trigram count to 1
			else:
				trigram_corpus[trigram] += 1 # increment trigram counter
	# go through trigram corpus, change UNKs
	freq_corpus = [['UNK','UNK','UNK',0]] # initialize our list of lists to be UNK
	total_count = 0 # intiailize the number of trigrams in our corpus to 0
	for item in trigram_corpus.items(): # for each trigram we've seen in train
		trigram, count = item # unpack this trigram
		trigram0, trigram1, trigram2 = trigram # unpack the unigrams in the trigram
		if count < 3: # if we've seen this trigram less than 3 times
			freq_corpus[0][3] += count # consider it rare, increase the number of UNK trigrams we've seen
		else: # else, this is not a rare word
			freq_corpus.append([trigram0,trigram1,trigram2,count]) # append this trigram and its count to our frequent corpus
		total_count += count # add the total number of trigrams to our total_count
	return (freq_corpus, total_count) # return our frequent_corpus and the total number of trigrams we saw in train

# Predict a sentence's probability via previously extracted vocabulary
def unigram_predict(vocab,test,unigram_count):
	yhat = [] # Initialize our vector of predictions as empty
	for instance in test: # for each sentence in our test data
		tokens = get_tokens(instance) # split the sentence into unigrams
		tokens.append('<STOP>')
		product = 1 # Initialize product as count of stop
		for word in tokens: # go through unigrams in this test sentence
			count = 1
			for unigram in vocab:
				if unigram[0] == word:
					if unigram[1] == 0:
						print("Count is 0 for ",unigram[0])
					count = float(unigram[1])/unigram_count
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
			# print("Computing probability for ",tokens[i+1],"given ",tokens[i])
			count_similar = 0 # initialize similarity count to 0
			count_match = 0
			for vocab_instance in vocab: # for each bigram in train
				if vocab_instance[:2] == bigram:
					count_match = vocab_instance[2]
					count_similar += vocab_instance[2]
				elif instance[0] == tokens[i]: # only will hit this clause if not full match
					count_similar += vocab_instance[2]
			if count_match == 0: # if this is a new bigram
				count_similar = bigram_count # number of bigrams
				count_match = vocab[0][2] # count of 'UNKS'
			prob_word = float(count_match) / count_similar
			prob_sentence = prob_sentence * prob_word
		yhat.append(prob_sentence)
	return yhat

def trigram_predict(vocab,test,trigram_count):
	# Generate proabilities for sentences
	yhat = [] # initialize yhat as empty
	for i, instance in enumerate(test): # for each sentence
		tokens = get_tokens(instance) # split instance into list by " "
		tokens.insert(0,'<START>') # prepend <START> token to each sentence
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		prob_sentence = 1 # initialize product variable, 1 * anything nonzero = 1
		for i in range(0,len(tokens)-2): 
			trigram = [tokens[i],tokens[i+1],tokens[i+2]] # set trigram
			# print("Computing probability for ",tokens[i+2],"given ",tokens[i],tokens[i+1])
			count_match = 0 # initialize match count to 0
			count_similar = 0 # initialize similarity count to 0
			for vocab_instance in vocab:
				#print("vocab instance[0-3]: ",vocab_instance)
				if vocab_instance[:3] == trigram:
					count_match = vocab_instance[3]
					count_similar += vocab_instance[3]
				elif vocab_instance[0] == tokens[i] and vocab_instance[1] == tokens[i+1]: # only will hit this clause if not full match
					count_similar += vocab_instance[3]
			if count_match ==0: # if this is a new trigram
				count_similar = trigram_count # number of trigrams
				count_match = vocab[0][3] # number of 'UNK'
			prob_word = float(count_match) / count_similar
			prob_sentence = prob_sentence * prob_word
		yhat.append(prob_sentence)
	return yhat

def unigram_per(vocab,test,unigram_count):
	perplexities = []
	logprob_sum = 0
	tot_word = 0
	for instance in test:
		tokens = get_tokens(instance) # split instance into list by " "
		tokens.append('<STOP>') # add the stop token
		for word in tokens:
			tot_word+=1
			foundword = 0
			for unigram in vocab: # check if we know this word
				if word == unigram[0]: # we found this word
					curr_prob = float(unigram[1])/unigram_count
					foundword = 1
			if foundword == 0: # map to UNK
				curr_prob = float(1)/unigram_count
			logprob_sum += float(math.log(curr_prob,2))
	l = float((-1/tot_word)) * float(logprob_sum)
	toappend = float(2 ** l)
	perplexities.append(toappend)
	return perplexities


def bigram_per(vocab,test,bigram_count):
	# Generate proabilities for sentences
	yhat = [] # initialize yhat as empty
	logprob_sum = 0
	tot_word = 0 
	for i, instance in enumerate(test): # for each sentence
		tokens = get_tokens(instance) # split instance into list by " "
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		prob_sentence = 1 # initialize product variable, 1 * anything nonzero = 1
		
		for i in range(0,len(tokens)-1): # for each bigram in this sentence
			tot_word+=1
			bigram = (tokens[i],tokens[i+1]) # set bigram
			count_similar = 0 # initialize similarity count to 0
			count_match = 0
			for instance in vocab: # for each bigram in train
				#print(instance[0:2], '::::', list(bigram))
				if instance[0:2] == list(bigram):
					#print('myes', instance)
					count_match = instance[2]
					count_similar += instance[2]
				elif instance[0] == tokens[i]: # only will hit this clause if not full match
					#print('mo')
					count_similar += instance[2]
			if count_match == 0: # this is a new bigram
				count_similar = bigram_count # number of bigrams
				count_match = vocab[0][2] # count of 'UNKS'
			prob_word = float(count_match) / count_similar
			logprob_sum += float(math.log(prob_word,2))
			#prob_sentence = prob_sentence * prob_word
	l = float((-1/tot_word)) * float(logprob_sum)
	toappend = 2 ** l
	yhat.append(toappend)	
	return yhat
	
	
def trigram_per(vocab,test,trigram_count):
	# Generate proabilities for sentences
	# print(trigram_count)
	yhat = [] # initialize yhat as empty
	logprob_sum = 0
	tot_word = 0
	for i, instance in enumerate(test): # for each sentence
		tokens = get_tokens(instance) # split instance into list by " "
		tokens.insert(0,'<START>')
		tokens.append('<STOP>')
		prob_sentence = 1 # initialize product variable, 1 * anything nonzero = 1
		for i in range(0,len(tokens)-2):
			tot_word +=1
			trigram = (tokens[i],tokens[i+1],tokens[i+2]) # set trigram
			count_match = 0
			count_similar = 0 # initialize similarity count to 0
			for instance in vocab:
				#print(instance[0:3],"::::",trigram)
				if instance[0:3] == list(trigram):
					#print('myes', instance)
					count_match = instance[3]
					count_similar += instance[3]
				elif instance[0] == tokens[i] and instance[1] == tokens[i+1]: # only will hit this clause if not full match
					#print('no')
					count_similar += instance[3]
			if count_match == 0: # this is a new bigram
				count_similar = trigram_count # number of trigrams
				count_match = vocab[0][3] # number of 'UNK'
			prob_word = float(count_match) / count_similar
			logprob_sum += float(math.log(prob_word,2))
			prob_sentence = prob_sentence * prob_word
	l = float((-1/tot_word)) * float(logprob_sum)
	toappend = 2 ** l
	yhat.append(toappend)	
	return yhat

def uni_tot_prob(vocab,unigram_count):
	tot = 0
	for i in vocab: 
		tot += i[1]
	print('total prob of uni: ',tot/unigram_count)
	return tot 
	
def bi_tot_prob(vocab,bigram_count):
	tot = 0
	for i in vocab:
		tot += i[2]
	print('total prob of bi: ',tot/bigram_count)
	return tot 
	
def tri_tot_prob(vocab,trigram_count):
	tot = 0
	for i in vocab:
		tot += i[3]
	print('total prob of tri: ',tot/trigram_count)
	return tot 

def interpolate(test, unigram, bigram, trigram, lamb_1, lamb_2, lamb_3, unigram_count, bigram_count, trigram_count):
	yhat_interpolated = [] # initialize return values as empty list
	for instance in test: # for each training instance
		init = 0 # signal we're at the beginning of a new sentence
		tokens = get_tokens(instance) # get tokens for this instance
		tokens.append('<STOP>') # add stop token

		# GET UNIGRAM TOKENS
		unigram_probs = [] # initialize list of word probabilities to empty
		for i in range(0,len(tokens)): # for each token in this instance
			found = 0
			for corpus_unigram in unigram:
				if corpus_unigram[0] == tokens[i]:
					unigram_probs.append(corpus_unigram[1]/unigram_count)
					found = 1
			if found == 0:
				unigram_probs.append(unigram[0][1]/unigram_count)

		# GET BIGRAMS
		tokens.insert(0,'<START>') # add start to give context to first bigram
		bigram_probs = [] # initialize bigram word probabilities as empty list
		for i in range(0,len(tokens)-1):
			count_similar = 0 # initialize similarity count to 0
			count_match = 0
			for vocab_instance in bigram: # for each bigram in train
				if vocab_instance[:2] == tokens[i:i+2]:
					count_match = vocab_instance[2]
					count_similar += vocab_instance[2]
				elif vocab_instance[0] == tokens[i]: # only will hit this clause if not full match
					count_similar += vocab_instance[2]
			if count_match == 0: # this is a new bigram
				count_similar = bigram_count # number of bigrams
				count_match = bigram[0][2] # count of 'UNKS'
			prob_word = float(count_match) / count_similar
			bigram_probs.append(prob_word)

		# Get trigrams
		tokens.insert(0,'<START>') # add one more start for the beginning trigram
		trigram_probs = [] # initialize trigram word probabilities as empty list
		for i in range(1,len(tokens)-2):
			count_similar = 0 # initialize similarity count to 0
			count_match = 0
			for vocab_instance in trigram: # for each bigram in train
				if vocab_instance[:3] == tokens[i:i+3]:
					count_match = vocab_instance[3]
					count_similar += vocab_instance[3]
				elif vocab_instance[0:2] == tokens[i:i+2]: # only will hit this clause if not full match
					count_similar += vocab_instance[3]
			if count_match == 0: # this is a new bigram
				count_similar = trigram_count # number of bigrams
				count_match = trigram[0][3] # count of 'UNKS'
			prob_word = float(count_match) / count_similar
			trigram_probs.append(prob_word)
		
		# get smoothed parameters
		smoothed_prob = 1
		for word_probabilities in zip(unigram_probs,bigram_probs,trigram_probs):
			uni, bi, tri = word_probabilities
			curr_prob = (uni * lamb_1) + (bi * lamb_2) + (tri * lamb_3)
			smoothed_prob = smoothed_prob * curr_prob
		yhat_interpolated.append(smoothed_prob)
	return yhat_interpolated

if __name__ == "__main__":
    main()
