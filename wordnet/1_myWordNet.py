from nltk.corpus import wordnet as wn 
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic

import pandas as pd 
import numpy as np
import gensim
import cPickle
import string
import time

data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')

#### 1: Create Parts-of-Speech (POS) tags for each word
def createPOStags(q):
	if q == 1:
		POSTAGS_text1 = []
		TOKENS_text1 = []
		tStart = time.time()
		for i, sentence in enumerate(data.text1):
			text = str(sentence).decode('utf-8')
			tokens = word_tokenize(text)
			TOKENS_text1.append(tokens)
			POSTAGS_text1.append(pos_tag(tokens))
			if i % 1000 == 0:
				print i
		tEnd = time.time()
		print "Finished tagging q1. Total Time: ", tEnd-tStart

		## Save vectors to pickle file
		cPickle.dump([TOKENS_text1, POSTAGS_text1], open('data/q1_POSTAGS.pkl', 'wb'), -1)
		print "Written to pickle file"

		return TOKENS_text1, POSTAGS_text1

	elif q == 2:
		POSTAGS_text2 = []
		TOKENS_text2 = []
		tStart = time.time()
		for i, sentence in enumerate(data.text2):
			text = str(sentence).decode('utf-8')
			tokens = word_tokenize(text)
			TOKENS_text2.append(tokens)
			POSTAGS_text2.append(pos_tag(tokens))
			if i % 1000 == 0:
				print i
		tEnd = time.time()
		print "Finished tagging q2. Total Time: ", tEnd-tStart

		## Save vectors to pickle file
		cPickle.dump([TOKENS_text2, POSTAGS_text2], open('data/q2_POSTAGS.pkl', 'wb'), -1)
		print "Written to pickle file"

		return TOKENS_text2, POSTAGS_text2
def loadPOStags():
	print "loading tokens and tags.."
	with open("data/q1_POSTAGS.pkl", "rb") as f:
	    g = cPickle.load(f)
	with open("data/q2_POSTAGS.pkl", "rb") as f:
	    h = cPickle.load(f)
	TOKENS_text1 = g[0]
	POSTAGS_text1 = g[1]
	TOKENS_text2 = h[0]
	POSTAGS_text2 = h[1]
	return TOKENS_text1, TOKENS_text2, POSTAGS_text1, POSTAGS_text2

# TOKENS_text1, POSTAGS_text1 = createPOStags(1)
# TOKENS_text2, POSTAGS_text2 = createPOStags(2)
# TOKENS_text1, TOKENS_text2, POSTAGS_text1, POSTAGS_text2 = loadPOStags()

#### 2: Create Sub-Sentences of Nouns, Verbs and Adjectives
def createSubSentsQ1():
	nouns_text1 = []
	verbs_text1 = []
	adj_text1 = []

	print "Generating Sub-Sentences for Q1 .. printing progress every 1000 sentences"
	tStart = time.time()
	for sent in range(0, len(POSTAGS_text1)):
		nouns = []
		verbs = []
		adjs = []
		for tok in range(0, len(POSTAGS_text1[:][sent])):
			tag = POSTAGS_text1[sent][tok][1]
			if tag[0] == 'N':  ## begins with 'N' is a Noun
				nouns.append(TOKENS_text1[sent][tok])
			elif tag[0] == 'V':  ## verbs
				verbs.append(TOKENS_text1[sent][tok])
			elif tag[0] == 'J': ## adjective
				adjs.append(TOKENS_text1[sent][tok])

		nouns_text1.append(nouns)
		verbs_text1.append(verbs)
		adj_text1.append(adjs)

		if sent % 1000 == 0:
			print sent

	tEnd = time.time()
	print "Done q1 sub-sentences, time elapsed: ", tEnd-tStart
	cPickle.dump([nouns_text1, verbs_text1, adj_text1], open('data/q1_subsents.pkl', 'wb'), -1)
	print "Written to pickle file"

	return nouns_text1, verbs_text1, adj_text1
##time: 6605.56968617s
def createSubSentsQ2():
	nouns_text2 = []
	verbs_text2 = []
	adj_text2 = []

	print "Generating Sub-Sentences for Q2 .. printing progress every 1000 sentences"
	tStart = time.time()
	for sent in range(0, len(POSTAGS_text2)):
		nouns = []
		verbs = []
		adjs = []
		for tok in range(0, len(POSTAGS_text2[:][sent])):
			tag = POSTAGS_text2[sent][tok][1]
			if tag[0] == 'N':  ## begins with 'N' is a Noun
				nouns.append(TOKENS_text2[sent][tok])
			elif tag[0] == 'V':  ## verbs
				verbs.append(TOKENS_text2[sent][tok])
			elif tag[0] == 'J': ## adjective
				adjs.append(TOKENS_text2[sent][tok])

		nouns_text2.append(nouns)
		verbs_text2.append(verbs)
		adj_text2.append(adjs)

		if sent % 1000 == 0:
			print sent

	tEnd = time.time()
	print "Done q1 sub-sentences, time elapsed: ", tEnd-tStart
	cPickle.dump([nouns_text2, verbs_text2, adj_text2], open('data/q2_subsents.pkl', 'wb'), -1)
	print "Written to pickle file"

	return nouns_text2, verbs_text2, adj_text2
def loadSubSents():
	print "loading subSents.."
	with open("data/q1_subsents.pkl", "rb") as f:
	    g = cPickle.load(f)
	with open("data/q2_subsents.pkl", "rb") as f:
	    h = cPickle.load(f)

	nouns_text1 = g[0]
	verbs_text1 = g[1]
	adj_text1 = g[2]

	nouns_text2 = h[0]
	verbs_text2 = h[1]
	adj_text2 = h[2] 

	return nouns_text1, verbs_text1, adj_text1, nouns_text2, verbs_text2, adj_text2
def removeStopWords():
	print "removing stop words from subsentences.."
	stop_words = stopwords.words('english')
	## remove stop words from Nouns, Verbs and Adjectives
	for i, subSent in enumerate(nouns_text1):
		nouns_text1[i] = [noun for noun in subSent if noun not in stop_words]	
	for i, subSent in enumerate(verbs_text1):
		verbs_text1[i] = [verb for verb in subSent if verb not in stop_words]
	for i, subSent in enumerate(adj_text1):
		adj_text1[i] = [adj for adj in subSent if adj not in stop_words]
	for i, subSent in enumerate(nouns_text2):
		nouns_text2[i] = [noun for noun in subSent if noun not in stop_words]
	for i, subSent in enumerate(verbs_text2):
		verbs_text2[i] = [verb for verb in subSent if verb not in stop_words]
	for i, subSent in enumerate(adj_text2):
		adj_text2[i] = [adj for adj in subSent if adj not in stop_words]
	
	return nouns_text1, verbs_text1, adj_text1, nouns_text2, verbs_text2, adj_text2

	# stop_words = stopwords.words('english') 
	# verbs_text1 = verbs_text1[0:10]
	# for subSent in verbs_text1:
	# 		print "before: ", subSent
	# 		subSent = [verb for verb in subSent if verb not in stop_words]
	# 		print "after: ", subSent
def lowerCase():
	print "making all words lowercase..."
	for i, (subSent1, subSent2) in enumerate(zip(nouns_text1, nouns_text2)):
		for j, noun in enumerate(subSent1):
			nouns_text1[i][j] = nouns_text1[i][j].lower()
		for j, noun in enumerate(subSent2):
			nouns_text2[i][j] = nouns_text2[i][j].lower()

	for i, (subSent1, subSent2) in enumerate(zip(verbs_text1, verbs_text2)):
		for j, verb in enumerate(subSent1):
			verbs_text1[i][j] = verbs_text1[i][j].lower()
		for j, verb in enumerate(subSent2):
			verbs_text2[i][j] = verbs_text2[i][j].lower()

	for i, (subSent1, subSent2) in enumerate(zip(adj_text1, adj_text2)):
		for j, adj in enumerate(subSent1):
			adj_text1[i][j] = adj_text1[i][j].lower()
		for j, adj in enumerate(subSent2):
			adj_text2[i][j] = adj_text2[i][j].lower()

	return nouns_text1, verbs_text1, adj_text1, nouns_text2, verbs_text2, adj_text2
def removeCommonWords():
	## assumes all words already in lower-case
	print "removing common words from subsentences..."
	for i, (subSent1, subSent2) in enumerate(zip(nouns_text1, nouns_text2)):
		commonners = set(subSent1).intersection(set(subSent2))
		nouns_text1[i] = set(subSent1) - commonners
		nouns_text2[i] = set(subSent2) - commonners

	for i, (subSent1, subSent2) in enumerate(zip(verbs_text1, verbs_text2)):
		commonners = set(subSent1).intersection(set(subSent2))
		verbs_text1[i] = set(subSent1) - commonners
		verbs_text2[i] = set(subSent2) - commonners		

	for i, (subSent1, subSent2) in enumerate(zip(adj_text1, adj_text2)):
		commonners = set(subSent1).intersection(set(subSent2))
		adj_text1[i] = set(subSent1) - commonners
		adj_text2[i] = set(subSent2) - commonners		

	return nouns_text1, verbs_text1, adj_text1, nouns_text2, verbs_text2, adj_text2

# nouns_text1, verbs_text1, adj_text1 = createSubSentsQ1()
# nouns_text2, verbs_text2, adj_text2 = createSubSentsQ2()

### Start Here to edit Similarity ###
## Load the subSentences
nouns_text1, verbs_text1, adj_text1, nouns_text2, verbs_text2, adj_text2 = loadSubSents()
## Lower-case all the words
nouns_text1, verbs_text1, adj_text1, nouns_text2, verbs_text2, adj_text2  = lowerCase()
## Remove stop words from subSentences
nouns_text1, verbs_text1, adj_text1, nouns_text2, verbs_text2, adj_text2 = removeStopWords()
# ## Remove common words  ## Tested: doesn't seem like a good idea!
nouns_text1, verbs_text1, adj_text1, nouns_text2, verbs_text2, adj_text2 = removeCommonWords()

#### 3. Lookup word in Wordnet using the tags
# iii) add them up
# dog = wn.synset('dog.n.01')
# wn.synsets('dog', pos=wn.VERB)
# wn.sentence_similarity("Are cats awesome?", "Are cats felines?")

## Types of Similarity Measures
# wn.path_similarity(hit, slap)
# synset1.lch_similarity(synset2)
# synset1.wup_similarity(synset2)
# synset1.jcn_similarity(synset2, ic)

## Return a score denoting how similar two word senses are, based on the shortest path that 
## connects the senses in the is-a (hypernym/hypnoym) taxonomy. 
## The score is in the range 0 to 1.
def computePathSimilarity():

	print "computing Path Similarity"
	tStart = time.time()
	## Compute the similarity between nouns
	ALLnouns_sim = []
	for subSent1, subSent2 in zip(nouns_text1, nouns_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			nounSim = np.zeros(len(subSent1)) 
			for i, noun1 in enumerate(subSent1):
				for noun2 in subSent2:
					try:
						w1 = noun1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = noun2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.path_similarity(w1, w2)
						if sim > nounSim[i]:
							nounSim[i] = sim
					except:
						continue
			# print nounSim
		else:
			nounSim = np.zeros(len(subSent2))
			for i, noun2 in enumerate(subSent2):
				for noun1 in subSent1:
					try:
						w1 = noun1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = noun2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.path_similarity(w1, w2)
						if sim > nounSim[i]:
							nounSim[i] = sim
					except:
						continue	
		
		ALLnouns_sim.append(nounSim)


	## Compute the similarity between verbs
	ALLverbs_sim = []
	for subSent1, subSent2 in zip(verbs_text1, verbs_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			verbSim = np.zeros(len(subSent1)) 
			for i, verb1 in enumerate(subSent1):
				for verb2 in subSent2:
					try:
						w1 = verb1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = verb2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.path_similarity(w1, w2)
						if sim > verbSim[i]:
							verbSim[i] = sim
					except:
						continue
		else:
			verbSim = np.zeros(len(subSent2))
			for i, verb2 in enumerate(subSent2):
				for verb1 in subSent1:
					try:
						w1 = verb1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = verb2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.path_similarity(w1, w2)
						if sim > verbSim[i]:
							verbSim[i] = sim
					except:
						continue	
		
		ALLverbs_sim.append(verbSim)


	## Compute the similarity between adjectives
	ALLadjs_sim = []
	for subSent1, subSent2 in zip(adj_text1, adj_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			adjSim = np.zeros(len(subSent1)) 
			for i, adj1 in enumerate(subSent1):
				for adj2 in subSent2:
					try:
						w1 = adj1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = adj2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.path_similarity(w1, w2)
						if sim > adjSim[i]:
							adjSim[i] = sim
					except:
						continue
			# print nounSim
		else:
			adjSim = np.zeros(len(subSent2))
			for i, adj2 in enumerate(subSent2):
				for adj1 in subSent1:
					try:
						w1 = adj1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = adj2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.path_similarity(w1, w2)
						if sim > adjSim[i]:
							adjSim[i] = sim
					except:
						continue	
		
		ALLadjs_sim.append(adjSim)

	tEnd = time.time()
	print ".. done. Time taken (PathSimilarity): ", tEnd-tStart
	return ALLnouns_sim, ALLverbs_sim, ALLverbs_sim

## Leacock-Chodorow Similarity: Return a score denoting how similar two word senses are, 
## based on the shortest path that connects the senses (as above) and the maximum depth of 
## the taxonomy in which the senses occur. The relationship is given as -log(p/2d) where p 
## is the shortest path length and d the taxonomy depth.
def computeLCHSimilarity():

	print "computing LCH Similarity"
	tStart = time.time()
	## Compute the similarity between nouns
	ALLnouns_sim = []
	for subSent1, subSent2 in zip(nouns_text1, nouns_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			nounSim = np.zeros(len(subSent1)) 
			for i, noun1 in enumerate(subSent1):
				for noun2 in subSent2:
					try:
						w1 = noun1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = noun2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.lch_similarity(w1, w2)
						if sim > nounSim[i]:
							nounSim[i] = sim
					except:
						continue
			# print nounSim
		else:
			nounSim = np.zeros(len(subSent2))
			for i, noun2 in enumerate(subSent2):
				for noun1 in subSent1:
					try:
						w1 = noun1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = noun2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.lch_similarity(w1, w2)
						if sim > nounSim[i]:
							nounSim[i] = sim
					except:
						continue	
		
		ALLnouns_sim.append(nounSim)


	## Compute the similarity between verbs
	ALLverbs_sim = []
	for subSent1, subSent2 in zip(verbs_text1, verbs_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			verbSim = np.zeros(len(subSent1)) 
			for i, verb1 in enumerate(subSent1):
				for verb2 in subSent2:
					try:
						w1 = verb1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = verb2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.lch_similarity(w1, w2)
						if sim > verbSim[i]:
							verbSim[i] = sim
					except:
						continue
		else:
			verbSim = np.zeros(len(subSent2))
			for i, verb2 in enumerate(subSent2):
				for verb1 in subSent1:
					try:
						w1 = verb1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = verb2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.lch_similarity(w1, w2)
						if sim > verbSim[i]:
							verbSim[i] = sim
					except:
						continue	
		
		ALLverbs_sim.append(verbSim)


	## Compute the similarity between adjectives
	ALLadjs_sim = []
	for subSent1, subSent2 in zip(adj_text1, adj_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			adjSim = np.zeros(len(subSent1)) 
			for i, adj1 in enumerate(subSent1):
				for adj2 in subSent2:
					try:
						w1 = adj1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = adj2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.lch_similarity(w1, w2)
						if sim > adjSim[i]:
							adjSim[i] = sim
					except:
						continue
			# print nounSim
		else:
			adjSim = np.zeros(len(subSent2))
			for i, adj2 in enumerate(subSent2):
				for adj1 in subSent1:
					try:
						w1 = adj1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = adj2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.lch_similarity(w1, w2)
						if sim > adjSim[i]:
							adjSim[i] = sim
					except:
						continue	
		
		ALLadjs_sim.append(adjSim)

	tEnd = time.time()
	print "..done. Time taken (LCHSimilarity): ", tEnd-tStart
	return ALLnouns_sim, ALLverbs_sim, ALLverbs_sim

## Jiang-Conrath Similarity Return a score denoting how similar two word senses are, based 
## on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) 
## and that of the two input Synsets. 
## The relationship is given by the equation 1 / (IC(s1) + IC(s2) - 2 * IC(lcs))
def computeInfContSimilarity():
	## Load an information content file from the wordnet_ic corpus
	brown_ic = wordnet_ic.ic('ic-brown.dat')

	print "computing Information Content Similarity..."
	tStart = time.time()
	## Compute the similarity between nouns
	ALLnouns_sim = []
	for subSent1, subSent2 in zip(nouns_text1, nouns_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			nounSim = np.zeros(len(subSent1)) 
			for i, noun1 in enumerate(subSent1):
				for noun2 in subSent2:
					try:
						w1 = noun1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = noun2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.jcn_similarity(w1, w2, brown_ic)
						if sim > nounSim[i]:
							nounSim[i] = sim
					except:
						continue
			# print nounSim
		else:
			nounSim = np.zeros(len(subSent2))
			for i, noun2 in enumerate(subSent2):
				for noun1 in subSent1:
					try:
						w1 = noun1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = noun2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.jcn_similarity(w1, w2, brown_ic)
						if sim > nounSim[i]:
							nounSim[i] = sim
					except:
						continue	
		
		ALLnouns_sim.append(nounSim)


	## Compute the similarity between verbs
	ALLverbs_sim = []
	for subSent1, subSent2 in zip(verbs_text1, verbs_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			verbSim = np.zeros(len(subSent1)) 
			for i, verb1 in enumerate(subSent1):
				for verb2 in subSent2:
					try:
						w1 = verb1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = verb2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.jcn_similarity(w1, w2, brown_ic)
						if sim > verbSim[i]:
							verbSim[i] = sim
					except:
						continue
		else:
			verbSim = np.zeros(len(subSent2))
			for i, verb2 in enumerate(subSent2):
				for verb1 in subSent1:
					try:
						w1 = verb1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = verb2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.jcn_similarity(w1, w2, brown_ic)
						if sim > verbSim[i]:
							verbSim[i] = sim
					except:
						continue	
		
		ALLverbs_sim.append(verbSim)


	## Compute the similarity between adjectives
	ALLadjs_sim = []
	for subSent1, subSent2 in zip(adj_text1, adj_text2):

		## if-else to use the longer sentence
		if (len(subSent1) > len(subSent2)):
			adjSim = np.zeros(len(subSent1)) 
			for i, adj1 in enumerate(subSent1):
				for adj2 in subSent2:
					try:
						w1 = adj1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = adj2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.jcn_similarity(w1, w2, brown_ic)
						if sim > adjSim[i]:
							adjSim[i] = sim
					except:
						continue
			# print nounSim
		else:
			adjSim = np.zeros(len(subSent2))
			for i, adj2 in enumerate(subSent2):
				for adj1 in subSent1:
					try:
						w1 = adj1 + ".n.01"
						w1 = wn.synset(w1)
						w2 = adj2 + ".n.01"
						w2 = wn.synset(w2)
						sim = wn.jcn_similarity(w1, w2, brown_ic)
						if sim > adjSim[i]:
							adjSim[i] = sim
					except:
						continue	
		
		ALLadjs_sim.append(adjSim)

	tEnd = time.time()
	print "..done. Time taken (InformationContentSimilarity): ", tEnd-tStart
	return ALLnouns_sim, ALLverbs_sim, ALLverbs_sim


# nouns_Pathsim, verbs_Pathsim, adjs_Pathsim = computePathSimilarity()
# nouns_LCHsim, verbs_LCHsim, adjs_LCHsim = computeLCHSimilarity()
nouns_ICsim, verbs_ICsim, adjs_ICsim = computeInfContSimilarity()

# ## Save to pickle file
# cPickle.dump([nouns_Pathsim, nouns_LCHsim, nouns_ICsim,
# 	          verbs_Pathsim, verbs_LCHsim, verbs_ICsim,
# 	          adjs_Pathsim, adjs_LCHsim, adjs_ICsim], open('data/nvaPathLCHicSim_WCOMMON.pkl', 'wb'), -1)

## Load from pickle file
# with open("data/nvaPathLCHsim.pkl", "rb") as f:
#     g = cPickle.load(f)
# nouns_Pathsim = g[0]
# nouns_LCHsim = g[1]
# verbs_Pathsim = g[2]
# verbs_LCHsim = g[3]
# adjs_Pathsim = g[4]
# adjs_LCHsim = g[5]

# with open("data/nvaPathLCHicSim_WCOMMON.pkl", "rb") as f:
# 	g = cPickle.load(f)
# nouns_Pathsim = g[0]
# nouns_LCHsim = g[1]
# nouns_ICsim = g[2]
# verbs_Pathsim = g[3]
# verbs_LCHsim = g[4]
# verbs_ICsim = g[5]
# adjs_Pathsim = g[6]
# adjs_LCHsim = g[7]
# adjs_ICsim = g[8]

## Add up the max similarities for Path Similarity from the feature vectors
# nouns_PathsimSUM = np.zeros(len(data))
# for i, vals in enumerate(nouns_Pathsim):
# 	for val in vals:
# 		nouns_Pathsim[i] += val
# verbs_PathsimSUM = np.zeros(len(data))
# for i, vals in enumerate(verbs_Pathsim):
# 	for val in vals:
# 		verbs_PathsimSUM[i] += val
# adjs_PathsimSUM = np.zeros(len(data))
# for i, vals in enumerate(adjs_Pathsim):
# 	for val in vals:
# 		# if val > 0.1:
# 		adjs_PathsimSUM[i] += val

## Add up the max similarities for LCH Similarity from the feature vectors
# nouns_LCHSUM = np.zeros(len(data))
# for i, vals in enumerate(nouns_LCHsim):
# 	nouns_LCHSUM[i] = sum(vals)

# verbs_LCHSUM = np.zeros(len(data))
# for i, vals in enumerate(verbs_LCHsim):
# 	verbs_LCHSUM[i] = sum(vals)

# adjs_LCHSUM = np.zeros(len(data))
# for i, vals in enumerate(adjs_LCHsim):
# 	adjs_LCHSUM[i] = sum(vals)

## Add up the max similarities for Information Content Similarity from the feature vectors
nouns_ICSUM = np.zeros(len(data))
for i, vals in enumerate(nouns_ICsim):
	nouns_ICSUM[i] = sum(vals)

verbs_ICSUM = np.zeros(len(data))
for i, vals in enumerate(verbs_ICsim):
	verbs_ICSUM[i] = sum(vals)

adjs_ICSUM = np.zeros(len(data))
for i, vals in enumerate(adjs_ICsim):
	adjs_ICSUM[i] = sum(vals)

data = data.drop(['id', 'qid1', 'qid2'], axis=1)
# data["nouns_PathsimSUM"] = nouns_PathsimSUM
# data["verbs_Pathsim"] = verbs_PathsimSUM
# data["adjs_PathsimSUM"] = adjs_PathsimSUM
# data["nouns_LCHSUM"] = nouns_LCHSUM
# data["verbs_LCHSUM"] = verbs_LCHSUM
# data["adj_LCHSUM"] = adjs_LCHSUM
data["nouns_icSUM"] = nouns_ICSUM
data["verbs_icSUM"] = verbs_ICSUM
data["adj_icSUM"] = adjs_ICSUM
# data.to_csv('data/WordNetFeaturesWCOMMON.csv', index=False)
data.to_csv('data/WordNetFeaturesNoCommon.csv', index=False)
























