# UCSC CSE 143 Introduction to Natural Language Processing
# Assignment 1: Language Modeling and Smoothing
# Erica Fong, David Nguyen and Rebecca Dorn
import math

def main():
	# Get the training data
	train = [] # Initialize empty list for training data
	with open('A1-Data/1b_benchmark.train.tokens', 'r',encoding="utf8") as filehandle: # Open train data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			train.append(current_place) # Append this sentance to our list of train data
	print("GETTING FREQUENT TOKENS FROM TRAIN...")
	known_tokens = get_knowntokens(train)
	print("REPLACING RARE TOKENS WITH 'UNK' IN TRAIN...")
	processed_train = replace_raretokens(train,known_tokens)
	print("PROCESSED TRAIN DATA")
	print("CHECK LENGTH OF KNOWN_TOKENS: ",len(known_tokens))

	# Get the dev data
	dev = [] # Initialize empty list for development
	with open('A1-Data/1b_benchmark.dev.tokens', 'r',encoding="utf8") as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_place = line[:-1] # Remove newline
			dev.append(current_place) # Append this sentance to our list of dev data
	#dev = dev[:10]  WHILE TESTING only do 10 instances
	print("REPLACING RARE TOKENS WITH 'UNK' IN DEV...")
	processed_dev = replace_raretokens(dev,known_tokens) # Map new words in dev to UNK
	print("PROCESSED DEV DATA")

	# Get the test data 
	test = [] # Initialize empty list for development
	with open('A1-Data/1b_benchmark.test.tokens', 'r',encoding="utf8") as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_pdlace = line[:-1] # Remove newline
			test.append(current_place) # Append this sentance to our list of dev data
	print("REPLACING RARE TOKENS WITH 'UNK' IN TRAIN...")
	processed_test = replace_raretokens(test,known_tokens) # process test data
	print("PROCESSED TEST DATA")

	# Get the distribution for unigram, bigram and trigrams using the training data
	unigram_count_vec, unigram_count = unigram_model(processed_train)
	print("GENERATING BIGRAM MODEL...")
	bigram_count_vec, bigram_count = bigram_model(processed_train) 
	print("GENERATING TRIGRAM MODEL...")
	trigram_count_vec, trigram_count = trigram_model(processed_train)
	print("Generated Uni/Bi/Trigram Distributions") # Tell user where we are inj program
	
	# Calculate perplexity for unigram, bigram and trigram distributions with our dev set
	print("FETCHING PERPLEXITY SCORES FOR UNI, BI AND TRIGRAM MODELS.....")
	print("ON TRAIN SET")
	sentence1_per = unigram_per(unigram_count_vec, processed_train, unigram_count)
	sentence2_per = bigram_per(bigram_count_vec, processed_train, bigram_count)
	sentence3_per = trigram_per(trigram_count_vec, processed_train, trigram_count)
	print("Unigram perplexity:",sentence1_per)
	print("Bigram perplexity: ",sentence2_per)
	print("Trigram perplexity: ",sentence3_per)
	print("ON DEV SET")
	sentence1_per = unigram_per(unigram_count_vec, processed_dev, unigram_count)
	sentence2_per = bigram_per(bigram_count_vec, processed_dev, bigram_count)
	sentence3_per = trigram_per(trigram_count_vec, processed_dev, trigram_count)
	print("Unigram perplexity:",sentence1_per)
	print("Bigram perplexity: ",sentence2_per)
	print("Trigram perplexity: ",sentence3_per)
	print("ON TEST SET")
	sentence1_perplex = unigram_per(unigram_count_vec, processed_test, unigram_count)
	sentence2_per = bigram_per(bigram_count_vec, processed_test, bigram_count)
	sentence3_per = trigram_per(trigram_count_vec, processed_test, trigram_count)
	print("Unigram perplexity:",sentence1_perplex)
	print("Bigram perplexity: ",sentence2_per)
	print("Trigram perplexity: ",sentence3_per)
	
	
	print("TESTING DIFFERENT Lambdas FOR LINEAR INTERPOLATION SMOOTHIING....")
	print("Lambda1 = 0.33, Lambda2 = 0.33, Lambda3 = 0.34")
	interpolated = interpolate_per(processed_dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.33,0.33,0.34, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Lambda1 = 0.7, Lambda2 = 0.2, Lambda3 = 0.1")
	interpolated = interpolate_per(processed_dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.7,0.2,0.1, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Lambda1 = 0.1, Lambda2 = 0.4, Lambda3 = 0.5")
	interpolated = interpolate_per(processed_dev, unigram_count_vec, bigram_count_vec, trigram_count_vec, 0.1, 0.4, 0.5, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Lambda1 = 0.1, Lambda2 = 0.3, Lambda3 = 0.6")
	interpolated = interpolate_per(processed_dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.1,0.3,0.6, unigram_count, bigram_count, trigram_count)
	print(interpolated)

def get_tokens(sentence): # Return a list of normalized words
	normalized = [] # intialize normalized works as empty
	for word in sentence.split(" "): # Split sentence into words via regex
		normalized.append(word) # place word in list
	return normalized # return our list of normalized words

def get_knowntokens(train):
	unigram_corpus = {} # Initialize our dictionary as empty
	token_list = ['UNK','<STOP>'] # initialize our token list with 'UNK' and '<STOP>'
	# get the count of each token
	for instance in train: # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		for word in tokens: # For each word in the sentence
			if word not in unigram_corpus.keys(): # if this is a new unigram
				unigram_corpus[word] = 1 # Initialize it's sightings to 1
			else: # else, we've already seen this token
				unigram_corpus[word] += 1 # Increment our counter by 1
	# put only the frequent words into our token list
	for word in unigram_corpus.keys(): # for each word in our dictionary
		if unigram_corpus[word] >= 3: # if we've seen it at least 3 times
			token_list.append(word) # add it to our list of frequent tokens
	return token_list # return the list of frequent tokens

def replace_raretokens(data, known_tokens):
	processed_sentences = [] # initialize our set of processed sentances as empty
	for instance in data: # for each new instance
		tokens = get_tokens(instance) # get the tokens for this instance (separate via spaces)
		sentence = [] # initialize the sentence as empty
		for word in tokens: # for each word
			if word in known_tokens: # if this word is in our list of frequent words
				sentence.append(word) # put it in the sentence
			else: # else, it's a rare token
				sentence.append("UNK") # put 'UNK' in its place
		processed_sentences.append(sentence) # append this sentence to our list of processed data
	return processed_sentences # return processed data

def unigram_model(train):
	unigram_corpus = {} # Initialize our dictionary as empty
	for tokens in train: # For each sentance in train
		tokens.append('<STOP>') # Add a stop token to the end
		for word in tokens: # For each word in the sentence
			if word not in unigram_corpus.keys(): # if this is a new unigram
				unigram_corpus[word] = 1 # Initialize it's sightings to 1
			else: # else, it's a new unigram
				unigram_corpus[word] += 1 # Increment our counter by 1
	# Loop through the seen unigrams, computing the total number of unigrams and changing datatype for ease
	total_count = 0 # initialize total number of unigrams in train as 0
	freq_corpus = [] # Initialize our new list of unigram,count as empty
	for item in unigram_corpus.items(): # For each word and it's count
		unigram, count = item # unpack the item
		freq_corpus.append([unigram,count]) # Put this unigram and its count into our new dictionary
		total_count += count # add this unigram's count to our total count
	return freq_corpus, total_count # return our list of unigram,count and the total number of unigrams in train

def bigram_model(train):
	bigram_corpus = {} # Initialize our dictionary as empty
	for j, tokens in enumerate(train): # For each sentance in train
		tokens.insert(0,'<START>') # append the start token to the beginning of our instnace
		for i in range(0,len(tokens)-1): # for each bigram
			bigram = (tokens[i], tokens[i+1]) # set bigram, instead of tokens[i:i+2]
			if bigram not in bigram_corpus.keys(): # if this is a new bigram
				bigram_corpus[bigram] = 1 # initialize our bigram count to 1
			else: # else, this is a new bigram
				bigram_corpus[bigram] += 1 # increment bigram counter
	# Compute total bigrams in data and change datatype
	total_count = 0 # initialize count of bigrams to 0
	freq_corpus = [] # initialize our list of lists to be empty
	for item in bigram_corpus.items(): # for each bigram in our dictionary
		bigram, count = item # unpack the bigram and count
		bigram0, bigram1 = bigram # unpack the bigram further into 2 separate objects
		freq_corpus.append([bigram0,bigram1,count]) # append list of [bigram[0],bigram[1],count] to list
		total_count += count # add the number of this bigram to our total count
	return freq_corpus, total_count # return our list of list and the total bigrams in train

def trigram_model(train):
	trigram_corpus = {} # Initialize our dictionary as empty
	for j, tokens in enumerate(train): # For each sentance in train
		tokens.insert(0,'<START>') # append a start token to this instance
		for i in range(0,len(tokens)-2): # for each trigram
			trigram = (tokens[i], tokens[i+1], tokens[i+2]) # set trigram 
			if trigram not in trigram_corpus.keys(): # if this is a new trigram
				trigram_corpus[trigram] = 1 # initialize our trigram count to 1
			else: # else, this is a new trigram
				trigram_corpus[trigram] += 1 # increment trigram counter
	# count total trigrams and change datatype
	total_count = 0 # intiailize the number of trigrams in our corpus to 0
	freq_corpus = [] # initialize our list of lists to be UNK
	for item in trigram_corpus.items(): # for each trigram we've seen in train
		trigram, count = item # unpack this trigram
		trigram0, trigram1, trigram2 = trigram # unpack the unigrams in the trigram
		freq_corpus.append([trigram0,trigram1,trigram2,count]) # append this trigram and its count to our frequent corpus
		total_count += count # add the total number of trigrams to our total_count
	return (freq_corpus, total_count) # return our list of trigrams and the total counts

def unigram_per(vocab,data,unigram_count):
	logprob_sum = 0 # Initialize our sum of log probabilities to 0
	tot_word = 0 # initialize the total words seen to 0
	for tokens in data: # for each sentence in our data
		tokens.append('<STOP>') # Append the stop token to our instance
		product = 1 # Initialize product as count of stop
		for word in tokens: # go through unigrams in this test sentence
			tot_word+=1 # another word! increment total word counter
			count = 1 # initialize count(this unigram) to 1 to avoid errors
			for unigram in vocab: # for each word in our stored vocabulary
				if unigram[0] == word: # if this word is the same as the one in the vocabulary
					count = unigram[1] # set count as this word's frequency in corpus
			prob_word = float(count)/unigram_count # the probability of this word is count/total unigrams
			logprob_sum += float(math.log(prob_word,2)) # add log(this word's probability) to our sum of log probabilities
	l = float((1/tot_word)) * float(logprob_sum) # compute l
	return float(2 ** (-l)) # return perplexity 2 ^ -l


def bigram_per(vocab,data,bigram_count):
	logprob_sum = 0 # initialize our log probability sum to 0
	tot_word = 0 # initialize the total words seen to 0
	for i, tokens in enumerate(data): # for each sentence
		tokens.insert(0,'<START>') # append <START> token 
		for i in range(0,len(tokens)-1): # for each bigram in this sentence
			tot_word += 1 # new word! add 1 to total words
			bigram = [tokens[i],tokens[i+1]] # set bigram
			count_similar = 0 # initialize similarity count to 0
			count_match = 0 # initialize match count to 0
			for vocab_instance in vocab: # for each bigram in train
				if vocab_instance[:2] == bigram: # if this bigram is the same as the one in our instance
					count_match = vocab_instance[2] # store the train count for this bigram
					count_similar += vocab_instance[2] # add the train count to the number of similar instances seen in train
				elif vocab_instance[0] == tokens[i]: # if this is a partial match
					count_similar += vocab_instance[2] # increment the number of similar bigrams by this count
			if count_match == 0: # if this is a new bigram
				count_match = 1 # set the number of matches to 1
				if count_similar == 0: # if there weren't even any similar ones
					count_similar = bigram_count # set the number of similar bigrams to total bigrams in train
			prob_word = float(count_match)/count_similar # the probability of this word is match/similar
			logprob_sum += float(math.log(prob_word,2)) # add log(prob_word) to our sum of log probabilities
	l = float(1/tot_word) * float(logprob_sum) # compute l
	return float(2**(-l)) # return perplexity 2 ^ -l
	
def trigram_per(vocab,data,trigram_count):
	logprob_sum = 0 # initialize sum of log probabilities to 0
	tot_word = 0 # initialize total words seen to 0
	for i, tokens in enumerate(data): # for each sentence
		tokens.insert(0,'<START>') # prepend <START> token to each sentence
		for i in range(0,len(tokens)-2): # for each bigram
			tot_word += 1 # seen another word! increment tot_word
			trigram = [tokens[i],tokens[i+1],tokens[i+2]] # set trigram
			count_match = 0 # initialize match count to 0
			count_similar = 0 # initialize similarity count to 0
			for vocab_instance in vocab: # for each trigram in train
				if vocab_instance[:3] == trigram: # if this trigram matches the one in train
					count_match = vocab_instance[3] # store the train count for this trigram
					count_similar += vocab_instance[3] # increment the count similar by this trigram count
				elif vocab_instance[0] == tokens[i] and vocab_instance[1] == tokens[i+1]: # if this is a context match
					count_similar += vocab_instance[3] # increment number of similar trigrams by this count
			if count_match == 0: # if this is a new trigram
				count_match = 1 # set matches to 1
				if count_similar == 0: # if there aren't even any similar trigrams
					count_similar = trigram_count # set the denominator to number of trigrams in train
			prob_word = float(count_match) / count_similar # store this words probability
			logprob_sum += float(math.log(prob_word,2)) # add logp(thisword) to our log probability sum
	l = float((1/tot_word)) * float(logprob_sum) # compute l 
	return float(2**(-l)) # return perplexity 2 ^ -l

def interpolate_per(data, unigram, bigram, trigram, lamb_1, lamb_2, lamb_3, unigram_count, bigram_count, trigram_count):
	tot_words = 0 # initialize total words seen to 0
	total_logp = 0 # initialize sum of log probabilities to 0
	for tokens in data: # for each training instance
		init = 0 # signal we're at the beginning of a new sentence

		# Get word probabilities via unigram model
		tokens.pop(0) # remove the 2 start starts from previous trigram perplexity calculations
		tokens.pop(0)
		unigram_wordprobs = [] # initialize list of unigram word probabilities to empty
		for i in range(0,len(tokens)): # for each token in this instance
			count = 0 # initialize our count as 0 to make compiler happy
			for corpus_unigram in unigram: # for each unigram in train 
				if corpus_unigram[0] == tokens[i]: # if the train unigram is the same as this one
					count = corpus_unigram[1] # store this count
			prob_word = float(count)/unigram_count # compute word probability
			unigram_wordprobs.append(prob_word) # append word probability to list

		# Generate word probabilities via bigram model
		tokens.insert(0,'<START>') # add start to give context to first bigram
		bigram_wordprobs = [] # initialize list of bigram word probabilites to 0
		for i in range(0,len(tokens)-1):
			count_similar = 0 # initialize similarity count to 0
			count_match = 0 # initialize match count to 0
			for vocab_instance in bigram: # for each bigram in train
				if vocab_instance[:2] == tokens[i:i+2]: # if this train bigram is the same
					count_match = vocab_instance[2] # store this count
					count_similar += vocab_instance[2] # increment similarity count
				elif vocab_instance[0] == tokens[i]: # if this is a context match
					count_similar += vocab_instance[2] # increment similarity count
			if count_match == 0: # this is a new bigram
				count_match = 1 # set to 1
				if count_similar == 0: # if nothing else is even similar
					count_similar = bigram_count # set similar to number of bigrams in train
			prob_word = float(count_match)/count_similar #store this words probability
			bigram_wordprobs.append(prob_word) # append to bigram probabilities

		# Get word probabilities via trigram model
		tokens.insert(0,'<START>') # add one more start for the beginning trigram
		trigram_wordprobs = [] # initialize trigram word probabilities to 0
		for i in range(1,len(tokens)-2):
			count_similar = 0 # initialize similarity count to 0
			count_match = 0 # initialize match count to 0
			for vocab_instance in trigram: # for each bigram in train
				if vocab_instance[:3] == tokens[i:i+3]: # if this is a match
					count_match = vocab_instance[3] # store match count
					count_similar += vocab_instance[3] # increment similarity count
				elif vocab_instance[0:2] == tokens[i:i+2]: # if similar but no match
					count_similar += vocab_instance[3] # increment similarity count
			if count_match == 0: # this is a new trigram
				count_match = 1 # set count match to 1
				if count_similar == 0: # if there's no similar ones
					count_similar = trigram_count # set count to number of trigrams in train
			prob_word = float(count_match)/count_similar # store the word probability
			trigram_wordprobs.append(prob_word) # append the word probability to our list
		
		# update perplexity
		sentence_prob = 1 # initialize this sentence's likelihood to 1
		for wordprob in zip(unigram_wordprobs,bigram_wordprobs,trigram_wordprobs):
			uniprob, biprob, triprob = wordprob # unpack the word probabilities
			current_prob = (lamb_1 * uniprob) + (lamb_2 * biprob) + (lamb_3 * triprob) # plug into eqn
			tot_words += 1 # increment number of words seen
			total_logp += math.log(current_prob,2) # add to log probability
			sentence_prob = float(sentence_prob) * current_prob # compute this sentence's probability
		
	# Return perplexity
	l = float(1/tot_words) * total_logp # compute l
	return(float(2**(-l))) # return perplexity 2 ^ -l

if __name__ == "__main__":
    main()
