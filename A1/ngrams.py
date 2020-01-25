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
	dev = dev[:10] # WHILE TESTING only do 10 instances
	print("REPLACING RARE TOKENS WITH 'UNK' IN DEV...")
	processed_dev = replace_raretokens(dev,known_tokens) # Map new words in dev to UNK
	print("PROCESSED DEV DATA")

	# Get the test data 
	test = [] # Initialize empty list for development
	with open('A1-Data/1b_benchmark.test.tokens', 'r',encoding="utf8") as filehandle: # Open dev data
		for line in filehandle: # For each line 
			current_pdlace = line[:-1] # Remove newline
			test.append(current_place) # Append this sentance to our list of dev data
	# test = replace_raretokens(test,known_tokens) # process test data
	# print("PROCESSED TEST DATA")

	# Get f distribution for unigram, bigram and trigrams using the training data
	unigram_count_vec, unigram_count = unigram_model(processed_train)
	print("GENERATING BIGRAM MODEL...")
	bigram_count_vec, bigram_count = bigram_model(processed_train) 
	print("GENERATING TRIGRAM MODEL...")
	trigram_count_vec, trigram_count = trigram_model(processed_train)
	print("Generated Uni/Bi/Trigram Distributions") # Tell user where we are inj program

	# Check that we sum to 1
	print("CHECKING PROBABILITY DISTRIBUTIONS SUM TO ONE...")
	uni_prob = uni_tot_prob(unigram_count_vec,unigram_count)
	bi_prob = bi_tot_prob(bigram_count_vec,bigram_count)
	tri_prob = tri_tot_prob(trigram_count_vec,trigram_count)
	
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
	sentence1_per = unigram_per(unigram_count_vec, processed_dev, unigram_count)
	sentence2_per = bigram_per(bigram_count_vec, processed_dev, bigram_count)
	sentence3_per = trigram_per(trigram_count_vec, processed_dev, trigram_count)
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
	interpolated = interpolate_per(processed_dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.33,0.33,0.34, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Gamma1 = 0.7, Gamma2 = 0.2, Gamma3 = 0.1")
	interpolated = interpolate_per(processed_dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.7,0.2,0.1, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Gamma1 = 0.1, Gamma2 = 0.4, Gamma3 = 0.5")
	interpolated = interpolate_per(processed_dev, unigram_count_vec, bigram_count_vec, trigram_count_vec, 0.1, 0.4, 0.5, unigram_count, bigram_count, trigram_count)
	print(interpolated)
	print("Gamma1 = 0.1, Gamma2 = 0.3, Gamma3 = 0.6")
	interpolated = interpolate_per(processed_dev, unigram_count_vec, bigram_count_vec, trigram_count_vec,0.1,0.3,0.6, unigram_count, bigram_count, trigram_count)
	print(interpolated)

# Get list of words separated by spaces
def get_tokens(sentence): # Return a list of normalized words
	normalized = [] # intialize normalized works as empty
	for word in sentence.split(" "): # Split sentence into words via regex
		normalized.append(word) # place word in list
	return normalized # return our list of normalized words

# Replace rare words of train with 'UNK' token
def get_knowntokens(train):
	unigram_corpus = {} # Initialize our dictionary as empty
	token_list = ['UNK','<STOP>']
	# get the count of each token
	for instance in train: # For each sentance in train
		tokens = get_tokens(instance) # split sentences into words
		for word in tokens: # For each word in the sentence
			if word not in unigram_corpus.keys(): # if this is a new unigram
				unigram_corpus[word] = 1 # Initialize it's sightings to 1
			else:
				unigram_corpus[word] += 1 # Increment our counter by 1
	# put only the frequent words into our token list
	for word in unigram_corpus.keys():
		if unigram_corpus[word] >= 3:
			token_list.append(word)
	return token_list # return our processed data

def replace_raretokens(data, known_tokens):
	processed_sentences = []
	for instance in data:
		tokens = get_tokens(instance)
		sentence = []
		for word in tokens:
			if word in known_tokens:
				sentence.append(word)
			else:
				sentence.append("UNK")
		processed_sentences.append(sentence)
	return processed_sentences

# Extract dictionary of unigram vocabulary and counts of those unigrams
def unigram_model(train):
	unigram_corpus = {} # Initialize our dictionary as empty
	for instance in train: # For each sentance in train
		tokens = instance
		tokens.append('<STOP>')
		for word in tokens: # For each word in the sentence
			if word not in unigram_corpus.keys(): # if this is a new unigram
				unigram_corpus[word] = 1 # Initialize it's sightings to 1
			else:
				unigram_corpus[word] += 1 # Increment our counter by 1
	# Create new list of list with frequent unigrams, mapping rare unigrams to 'UNK'
	total_count = 0 # initialize total number of unigrams in train to be 0
	freq_corpus = [] # Initialize the number of rare words to 0
	for item in unigram_corpus.items(): # Go through each word and it's count
		unigram, count = item
		freq_corpus.append([unigram,count]) # Put this unigram and its count into our new dictionary
		total_count += count
	return freq_corpus, total_count

def bigram_model(train):
	bigram_corpus = {} # Initialize our dictionary as empty
	for j, instance in enumerate(train): # For each sentance in train
		tokens = instance
		tokens.insert(0,'<START>')
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
		tokens = instance
		tokens.insert(0,'<START>')
		for i in range(0,len(tokens)-2): # for each trigram
			trigram = (tokens[i], tokens[i+1], tokens[i+2]) # set trigram 
			if trigram not in trigram_corpus.keys(): # if this is a new trigram
				trigram_corpus[trigram] = 1 # initialize our trigram count to 1
			else:
				trigram_corpus[trigram] += 1 # increment trigram counter
	# go through trigram corpus, change UNKs
	total_count = 0 # intiailize the number of trigrams in our corpus to 0
	freq_corpus = [] # initialize our list of lists to be UNK
	for item in trigram_corpus.items(): # for each trigram we've seen in train
		trigram, count = item # unpack this trigram
		trigram0, trigram1, trigram2 = trigram # unpack the unigrams in the trigram
		freq_corpus.append([trigram0,trigram1,trigram2,count]) # append this trigram and its count to our frequent corpus
		total_count += count # add the total number of trigrams to our total_count
	return (freq_corpus, total_count) # return our frequent_corpus and the total number of trigrams we saw in train

def unigram_per(vocab,test,unigram_count):
	yhat = [] # Initialize our vector of predictions as empty
	logprob_sum = 0
	tot_word = 0
	for tokens in test: # for each sentence in our test data
		tokens.append('<STOP>')
		product = 1 # Initialize product as count of stop
		for word in tokens: # go through unigrams in this test sentence
			tot_word+=1
			count = 1
			for unigram in vocab:
				if unigram[0] == word:
					count = unigram[1]
			prob_word = float(count)/unigram_count
			logprob_sum += float(math.log(prob_word,2))
	l = float((-1/tot_word)) * float(logprob_sum)
	yhat.append(float(2 ** l)) # append this instance's probability to our predictions
	return yhat # return predicted probabilities


def bigram_per(vocab,test,bigram_count):
	# Generate proabilities for sentences
	yhat = [] # initialize yhat as empty
	logprob_sum = 0
	tot_word = 0 
	for i, tokens in enumerate(test): # for each sentence
		tokens.insert(0,'<START>')
		for i in range(0,len(tokens)-1): # for each bigram in this sentence
			tot_word+=1
			bigram = [tokens[i],tokens[i+1]] # set bigram
			count_similar = 0 # initialize similarity count to 0
			count_match = 0
			for vocab_instance in vocab: # for each bigram in train
				if vocab_instance[:2] == bigram:
					count_match = vocab_instance[2]
					count_similar += vocab_instance[2]
				elif vocab_instance[0] == tokens[i]: # only will hit this clause if not full match
					count_similar += vocab_instance[2]
			if count_match == 0: # if this is a new bigram
				count_match = 1 # count of 'UNKS'
				if count_similar == 0: # if this is totally new
					count_similar = bigram_count
			prob_word = float(count_match)/count_similar
			logprob_sum += float(math.log(prob_word,2))
	l = float(-1/tot_word) * float(logprob_sum)
	yhat.append(float(2**l))	
	return yhat
	
def trigram_per(vocab,test,trigram_count):
	# Generate proabilities for sentences
	yhat = [] # initialize yhat as empty
	logprob_sum = 0
	tot_word = 0
	for i, tokens in enumerate(test): # for each sentence
		tokens.insert(0,'<START>') # prepend <START> token to each sentence
		for i in range(0,len(tokens)-2): 
			tot_word +=1
			trigram = [tokens[i],tokens[i+1],tokens[i+2]] # set trigram
			count_match = 0 # initialize match count to 0
			count_similar = 0 # initialize similarity count to 0
			for vocab_instance in vocab:
				if vocab_instance[:3] == trigram:
					count_match = vocab_instance[3]
					count_similar += vocab_instance[3]
				elif vocab_instance[0] == tokens[i] and vocab_instance[1] == tokens[i+1]: # only will hit this clause if not full match
					count_similar += vocab_instance[3]
			if count_match == 0: # if this is a new trigram
				count_match = 1 # number of 'UNK'
				if count_similar == 0:
					count_similar = trigram_count
			prob_word = float(count_match) / count_similar
			logprob_sum += float(math.log(prob_word,2))
	l = float((-1/tot_word)) * float(logprob_sum)
	yhat.append(float(2**l))	
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

def interpolate_per(test, unigram, bigram, trigram, lamb_1, lamb_2, lamb_3, unigram_count, bigram_count, trigram_count):
	# yhat_interpolated = [] # initialize return values as empty list
	unigram_probs = []
	bigram_probs = []
	trigram_probs = []
	tot_words = 0
	for sentence in test: # for each training instance
		init = 0 # signal we're at the beginning of a new sentence
		tokens = sentence
		# tokens.append('<STOP>') # add stop token

		# Get log probability for this sentence given unigram model
		tokens.pop(0) # remove start start from previous trigram perplexity calculations
		tokens.pop(0)
		# unigram_probs = [] # initialize list of word probabilities to empty
		prob_sentence = 1
		for i in range(0,len(tokens)): # for each token in this instance
			count = 0
			tot_words += 1
			for corpus_unigram in unigram:
				if corpus_unigram[0] == tokens[i]:
					count = corpus_unigram[1]
			prob_word = float(count)/unigram_count
			prob_sentence = float(prob_sentence) * prob_word
		unigram_probs.append(prob_sentence) # add the log probability for this sentence


		# Generate log probabilites for each unigram via bigram model
		tokens.insert(0,'<START>') # add start to give context to first bigram
		# bigram_probs = [] # initialize bigram word probabilities as empty list
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
			prob_word = float(count_match)/count_similar
			prob_sentence = float(prob_sentence) * prob_word
		bigram_probs.append(prob_sentence) # add the log probability for this sentence

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
			prob_word = float(count_match)/count_similar
			prob_sentence = float(prob_sentence) * prob_word
		trigram_probs.append(prob_sentence,2)
		
	# get smoothed parameters
	total_logp = 0
	for thisword_logps in zip(unigram_probs,bigram_probs,trigram_probs):
		uni_logp, bi_logp, tri_logp = thisword_logps
		total_logp += (math.log(uni_logp,2) * lamb_1) + (math.log(bi_logp,2) * lamb_2) + (math.log(tri_logp,2) * lamb_3)
	l = float(-1/tot_words) * total_logp
	return(float(2**l))

if __name__ == "__main__":
    main()
