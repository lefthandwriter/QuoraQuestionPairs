## Version 2: Extract multiple questions per .txt file, up to 100,000

import scipy as sci
import pandas as pd
import numpy as np
import cPickle
import time
import json
import os


###### combine the label embeddings - POS and NER   ####
data = pd.read_csv('data/kaggle/test.csv')
# data = data[0:1000000]
dataLength = len(data) 

# with open("stanford-corenlp-full-2016-10-31/sent1/json/q1.txt.json") as f:
# 	z = json.load(f)

featPOSNER1 = np.zeros((dataLength, 12))
featPOSNER2 = np.zeros((dataLength, 12))
featPOSNERcompEuc = np.zeros(dataLength)
featPOSNERcompCity = np.zeros(dataLength)
featPOSNERCosine = np.zeros(dataLength)
featPOSNERbrayCurtis = np.zeros(dataLength)
featPOSNERChebyshev = np.zeros(dataLength)

### Process the first question vector list
def extractPOSNER_list1():
	print "starting to process the first question list"
	## Loop over files that we saved the extrated tokens, POS and NER tags to
	for file in range(1, 25):

		filename = 'data/kaggle/sent1/Q1_'
		filename = filename + str(file) + '.txt.json'

		with open(filename) as f:
			z = json.load(f)

		for sentence in z["sentences"]:
			pos_tags = []
			ner_tags = []

			## Extract the index, pos_tags and ner_tags from the json files
			try:
				index = int(sentence["tokens"][0]["word"]) # index is the first word for each sentence
			except:
				index = index + 1
				continue
			for token in sentence["tokens"]:
				pos_tags.append(token["pos"])
				ner_tags.append(token["ner"])
			
			# delete the first two tags: (index and ,)
			del pos_tags[0:1]
			del ner_tags[0:1]

			## Create the POS and NER embedding
			for ner_text in ner_tags:
				if ner_text == 'LOCATION':
					featPOSNER1[index][0] += 1
				elif ner_text == 'PERSON':
					featPOSNER1[index][1] += 1
				elif ner_text == 'ORGANIZATION':
					featPOSNER1[index][2] += 1
				elif ner_text == 'MONEY':
					featPOSNER1[index][3] += 1
				else:
					featPOSNER1[index][4] += 1  # default to prevent 0's (this in a way encodes length): delete if we don't encounter 0 vectors
			for pos_text in pos_tags:
				if pos_text == 'CD':   # cardinal number
					featPOSNER1[index][5] += 1
				elif pos_text == 'FW':   # foreign word
					featPOSNER1[index][6] += 1
				elif pos_text[0] == 'J':  # adjective
					featPOSNER1[index][7] += 1	
				elif pos_text[0] == 'N':  # noun
					featPOSNER1[index][8] += 1
				elif pos_text[0] == 'R':   # adverb
					featPOSNER1[index][9] += 1
				elif pos_text[0] == 'V':  # verb
					featPOSNER1[index][10] += 1
				elif pos_text[0] == 'W':  # what,how etc
					featPOSNER1[index][11] += 1	
	print "done the first question list"
### Process the second question vector list
def extractPOSNER_list2():
	print "starting the second question list.. 24 files"
	for file in range(1, 25):

		filename = 'data/kaggle/sent2/Q2_'
		filename = filename + str(file) + '.txt.json'

		with open(filename) as f:
			z = json.load(f)

		for sentence in z["sentences"]:
			pos_tags = []
			ner_tags = []

			## Extract the index, pos_tags and ner_tags from the json files
			try:
				index = int(sentence["tokens"][0]["word"]) # index is the first word for each sentence
			except:
				index = index + 1
				continue
			index = int(sentence["tokens"][0]["word"]) # index is the first word for each sentence
			for token in sentence["tokens"]:
				pos_tags.append(token["pos"])
				ner_tags.append(token["ner"])
				# delete the first two tags: (index and ,)
			del pos_tags[0:1]
			del ner_tags[0:1]

			## Create the POS and NER embedding
			for ner_text in ner_tags:
				if ner_text == 'LOCATION':
					featPOSNER2[index][0] += 1
				elif ner_text == 'PERSON':
					featPOSNER2[index][1] += 1
				elif ner_text == 'ORGANIZATION':
					featPOSNER2[index][2] += 1
				elif ner_text == 'MONEY':
					featPOSNER2[index][3] += 1
				else:
					featPOSNER2[index][4] += 1  # default to prevent 0's (this in a way encodes length): delete if we don't encounter 0 vectors
			for pos_text in pos_tags:
				if pos_text == 'CD':   # cardinal number
					featPOSNER2[index][5] += 1
				elif pos_text == 'FW':   # foreign word
					featPOSNER2[index][6] += 1
				elif pos_text[0] == 'J':  # adjective
					featPOSNER2[index][7] += 1	
				elif pos_text[0] == 'N':  # noun
					featPOSNER2[index][8] += 1
				elif pos_text[0] == 'R':   # adverb
					featPOSNER2[index][9] += 1
				elif pos_text[0] == 'V':  # verb
					featPOSNER2[index][10] += 1
				elif pos_text[0] == 'W':  # what,how etc
					featPOSNER2[index][11] += 1	
	print "done the second question list"
### Normalize the vectors
def normPOSNER():
	print "starting to normalize vectors..."
	norm1 = np.linalg.norm(featPOSNER1, axis=1)
	norm2 = np.linalg.norm(featPOSNER2, axis=1)
	for vector in range(0, dataLength):
		if norm1[vector] != 0:
			featPOSNER1[vector] = featPOSNER1[vector]/norm1[vector]
		if norm2[vector] != 0:
			featPOSNER2[vector] = featPOSNER2[vector]/norm2[vector]
	print "done normalizing"
### Compute distances
def computePOSNERdist():
	## now build the POSNER comparison vector
	## Spatial distance functions:
	## https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
	print "starting to compute distances.."
	for sentence in range(0, dataLength):
		vec = np.vstack((featPOSNER1[sentence,:], featPOSNER2[sentence,:]))
		featPOSNERcompEuc[sentence] = sci.spatial.distance.pdist(vec, metric='euclidean')
		featPOSNERcompCity[sentence] = sci.spatial.distance.pdist(vec, metric='cityblock')
		featPOSNERCosine[sentence] = sci.spatial.distance.pdist(vec, metric='cosine')
		featPOSNERbrayCurtis[sentence] = sci.spatial.distance.pdist(vec, metric='braycurtis')
		featPOSNERChebyshev[sentence] = sci.spatial.distance.pdist(vec, metric='chebyshev')
	print "done computing distances"


print 'starting to generate embedding now ... '
tStart = time.time()
extractPOSNER_list1()
extractPOSNER_list2()
normPOSNER()
computePOSNERdist()
tEnd = time.time()
print 'time taken to process', dataLength*2, 'questions: ', tEnd-tStart, ' s'


## save results to pickle file
print "saving results to file..."
with open('data/kaggle/TEST_POSNER_vectors.p', 'w') as f:
	# cPickle.dump([featPOSNER1, featPOSNER2, featPOSNERCosine, featPOSNERbrayCurtis],f)
	# cPickle.dump([featPOSNER1, featPOSNER2, featPOSNERcompEuc, featPOSNERcompCity, featPOSNERbrayCurtis, featPOSNERChebyshev],f)
	cPickle.dump([featPOSNER1, featPOSNER2],f)

with open('data/kaggle/TEST_POSNER_feat.p', 'w') as f:
	# cPickle.dump([featPOSNER1, featPOSNER2, featPOSNERCosine, featPOSNERbrayCurtis],f)
	# cPickle.dump([featPOSNER1, featPOSNER2, featPOSNERcompEuc, featPOSNERcompCity, featPOSNERbrayCurtis, featPOSNERChebyshev],f)
	cPickle.dump([featPOSNERcompCity, featPOSNERbrayCurtis],f)



## save to data frame
# data['POSNER_Euc']  = featPOSNERcompEuc
data['POSNER_City'] = featPOSNERcompCity
# data['POSNER_Cosine'] = featPOSNERCosine
data['POSNER_BrayCurtis'] = featPOSNERbrayCurtis
# data['POSNER_Chebyshev'] = featPOSNERChebyshev

## write to csv file
data.to_csv('data/kaggle/TESTstanPOSNERfeat.csv', index=False)
print 'done writing to .csv and pickle file'

# print "loading from pickle file.."
# with open("data/kaggle/TESTstanPOSfeat.p", "rb") as f:
#     g = cPickle.load(f)
# featPOSNER1 = g[0]
# featPOSNER2 = g[1]
# featPOSNERcompEuc = g[2]













