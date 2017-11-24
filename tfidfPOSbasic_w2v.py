### This Python script trains on Google's Corpus to produce word vectors (word2vec).                         ###
### Then, we apply tf-idf weights on the word vectors to produce the sentence vector for each question pair  ###
## After that, append POSNER and basic embeddings

import sklearn.manifold

import pandas as pd 
import numpy as np
import gensim
import cPickle

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

import string
import time

### train model using Google's 3 million data corpus
# tStart = time.time()
# model = gensim.models.Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
# tEnd = time.time()
# model.save("myWord2Vec")
# print "Trained word2vec model (s): ", tEnd-tStart

# model = gensim.models.Word2Vec.load("myWord2Vec") 

## explore model
# print model.syn0.shape
# print model.syn0
# print model.most_similar("sushi")

### read Quora data
data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')

### apply Google Model to Quora question
## Method 1: tf-idf + average

## Get tf-idf for all Q-Pair 1
# vectorizer = TfidfVectorizer()
# tfidf = vectorizer.fit_transform(data.text1)

# # Get tokens of sentence without punctuation
# text = str(data.text1[404288]).decode('utf-8') # avoid errors
# text_nopunc = text.encode('utf-8').translate(None, string.punctuation)  # remove punctuation; re-encode for translate()
# tokens = word_tokenize(text_nopunc)

# # Get tf-idf scores for each word in the sentence
# tf_text = tfidf[404288].toarray()  # from the sparse matrix get the non-zero elements which represent the words in our sentence
# tf_text.sort()
# tf_weight = tf_text[0, -len(tokens):]

# # Multiply tf-idf with word2vec vector, for each word
# model[tokens[0]]*tf_weight[0]

############ Apply tf-idf + average to all Q-Pair 1 ##########

# ## Get tf-idf for all Q-Pair 1
# vectorizer = TfidfVectorizer()
# # tfidf = vectorizer.fit_transform(data.text1)
# tfidf = vectorizer.fit_transform(data['text1'].values.astype('U'))

# tStart = time.time()
# word2vecfeatsize = len(model["word"])
# sentVec1 = np.zeros( (len(data), word2vecfeatsize) )  # this should be (nx300)
# for i, sentence in enumerate(data.text1):

# 	# Get tokens of sentence without punctuation
# 	text = str(sentence).decode('utf-8').lower() # to avoid errors
# 	text_nopunc = text.encode('utf-8').translate(None, string.punctuation)  # remove punctuation; re-encode for translate()
# 	tokens = word_tokenize(text_nopunc)

# 	# Get tf-idf scores for each word in the sentence
# 	tf_text = tfidf[i].toarray()  # from the sparse matrix get the non-zero elements which represent the words in our sentence
# 	tf_text.sort()
# 	tf_weight = tf_text[0, -len(tokens):]

# 	# Multiply tf-idf with word2vec vector, for each word
# 	qWordsVec = np.zeros( (len(tokens), word2vecfeatsize) )
# 	for j, word in enumerate(tokens):
# 		try:
# 			qWordsVec[j] = model[word] * tf_weight[j]  # some words not in Google's Corpus
# 		except:
# 			continue
	
# 	# Take the average, now that we have taken the tf-idf weights into account
# 	v = qWordsVec.sum(axis=0)  # sum by features, all the word values
# 	# sentVec[i] =  v / v.sum()  # take the average
# 	sentVec1[i] = v / np.sqrt((v ** 2).sum())

# 	print "iteration ", i 

# tEnd = time.time()
# print "Applied tf-idf on word2vec model Q1 (s): ", tEnd-tStart

# ## Save vectors to pickle file
# cPickle.dump(sentVec1, open('data/q1_myw2v.pkl', 'wb'), -1)
# print "Saved sentVec1 to pickle file"


############ Apply tf-idf + average to all Q-Pair 2 ##########

# ## Get tf-idf for all Q-Pair 2
# vectorizer = TfidfVectorizer()
# tfidf = vectorizer.fit_transform(data['text2'].values.astype('U'))

# tStart = time.time()
# word2vecfeatsize = len(model["word"])
# sentVec2 = np.zeros( (len(data), word2vecfeatsize) )  # this should be (nx300)
# for i, sentence in enumerate(data.text2):

# 	## Get tokens of sentence without punctuation
# 	text = str(sentence).decode('utf-8').lower() # to avoid errors
# 	text_nopunc = text.encode('utf-8').translate(None, string.punctuation)  # remove punctuation; re-encode for translate()
# 	tokens = word_tokenize(text_nopunc)

# 	## Get tf-idf scores for each word in the sentence
# 	tf_text = tfidf[i].toarray()  # from the sparse matrix get the non-zero elements which represent the words in our sentence
# 	tf_text.sort()
# 	tf_weight = tf_text[0, -len(tokens):]

# 	## Multiply tf-idf with word2vec vector, for each word
# 	qWordsVec = np.zeros( (len(tokens), word2vecfeatsize) )
# 	for j, word in enumerate(tokens):
# 		try:
# 			qWordsVec[j] = model[word] * tf_weight[j]  # some words not in Google's Corpus
# 		except:
# 			continue
	
# 	## Take the average, now that we have taken the tf-idf weights into account
# 	v = qWordsVec.sum(axis=0)  # sum by features, all the word values
# 	# sentVec2[i] =  v / np.sqrt() # normalize the vectors
# 	sentVec2[i] = v / np.sqrt((v ** 2).sum()) # normalize the vectors

# 	print "iteration ", i 

# tEnd = time.time()
# print "Applied tf-idf on word2vec model Q2 (s): ", tEnd-tStart

# ## Save vectors to pickle file
# cPickle.dump(sentVec2, open('data/q2_myw2v.pkl', 'wb'), -1)
# print "Saved sentVec2 to pickle file"


# ##### Extract features: distances ####

# ## wmd = word mover's distance from raw text

# load sentence vectors from pickle file
with open("data/q1_tfidfw2v.pkl", "rb") as f:
    sentVec1 = cPickle.load(f)
with open("data/q2_tfidfw2v.pkl", "rb") as f:
    sentVec2 = cPickle.load(f)
with open("data/stanPOSfeat.p", "rb") as f:
	posVec = cPickle.load(f)
posVec1 = posVec[0]
posVec2 = posVec[1]

# Concatenate tfidf_w2v vectors and pos vectors
w2v_pos_vec1 = np.hstack((sentVec1, posVec1))
w2v_pos_vec2 = np.hstack((sentVec2, posVec2))

## Features for tfidfPOSNER_basic_w2v
a = pd.read_csv('data/quora_features.csv')
lenq1 = a.len_q1
lenq2 = a.len_q2
lencq1 = a.len_char_q1
lencq2 = a.len_char_q2
# cmmn = a.common_words

# Form basic vectors
basic_v1 = np.vstack((lenq1, lencq1)).T
basic_v2 = np.vstack((lenq2, lencq2)).T
basic_v1 = basic_v1.astype(float)
basic_v2 = basic_v2.astype(float)
# Normalize basic vectors
normbasic1 = np.linalg.norm(basic_v1, axis=1)
normbasic2 = np.linalg.norm(basic_v2, axis=1)
for vector in range(0, len(basic_v1)):
	if normbasic1[vector] != 0:
		basic_v1[vector] = basic_v1[vector]/normbasic1[vector]
	if normbasic2[vector] != 0:
		basic_v2[vector] = basic_v2[vector]/normbasic1[vector]

w2v_pos_basic_vec1 = np.hstack((w2v_pos_vec1, basic_v1))
w2v_pos_basic_vec2 = np.hstack((w2v_pos_vec2, basic_v2))


cPickle.dump([w2v_pos_basic_vec1, w2v_pos_basic_vec2], open('data/augmented_w2v.p', 'wb'), -1)


# ## Features for tfidfPOSNERbasic_w2v
# data['tfidfPOSNERbasicw2v_cos'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_basic_vec1),
#                                                                   np.nan_to_num(w2v_pos_basic_vec2))]

# data['tfidfPOSNERbasicw2v_braycurtis'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_basic_vec1),
#                                                                              np.nan_to_num(w2v_pos_basic_vec2))]

# data['tfidfPOSNERbasicw2v_cityb'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_basic_vec1),
#                                                                   np.nan_to_num(w2v_pos_basic_vec2))]

# data['tfidfPOSNERbasicw2v_jaccard'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_basic_vec1),
#                                                                   np.nan_to_num(w2v_pos_basic_vec2))]

# data['tfidfPOSNERbasicw2v_canberra'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_basic_vec1),
#                                                                   np.nan_to_num(w2v_pos_basic_vec2))]

# data['tfidfPOSNERbasicw2v_euc'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_basic_vec1),
#                                                                   np.nan_to_num(w2v_pos_basic_vec2))]


# ## Features for tfidfPOSNER_w2v
# data['tfidfPOSNERw2v_cos'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_vec1),
#                                                             np.nan_to_num(w2v_pos_vec2))]

# data['tfidfPOSNERw2v_braycurtis'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_vec1),
#                                                                         np.nan_to_num(w2v_pos_vec2))]

# data['tfidfPOSNERw2v_cityb'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_vec1),
#                                                                   np.nan_to_num(w2v_pos_vec2))]

# data['tfidfPOSNERw2v_jaccard'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_vec1),
#                                                                   np.nan_to_num(w2v_pos_vec2))]

# data['tfidfPOSNERw2v_canberra'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_vec1),
#                                                                   np.nan_to_num(w2v_pos_vec2))]

# data['tfidfPOSNERw2v_euc'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(w2v_pos_vec1),
#                                                                   np.nan_to_num(w2v_pos_vec2))]


# ## Features for tfidf_w2v
# data['tfidfw2v_cos'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(sentVec1),
#                                                           np.nan_to_num(sentVec2))]

# data['tfidfw2v_braycurtis'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(sentVec1),
#                                                           np.nan_to_num(sentVec2))]

# data['tfidfw2v_cityb'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(sentVec1),
#                                                                   np.nan_to_num(sentVec2))]

# data['tfidfw2v_jaccard'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(sentVec1),
#                                                                   np.nan_to_num(sentVec2))]

# data['tfidfw2v_canberra'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(sentVec1),
#                                                                   np.nan_to_num(sentVec2))]

# data['tfidfw2v_euc'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(sentVec1),
#                                                                   np.nan_to_num(sentVec2))]


# # ## Save features to csv file
# data.to_csv('data/augmented_word2vec.csv', index=False)







# feat = pd.read_csv('data/pos_tfidf_word2vec.csv')

#### Visualize using tSNe
# tsne = sklearn.manifold.TSNE(n_components=3, random_state=0)

# def tsne_plot(model):
#     "Creates and TSNE model and plots it"
#     labels = []
#     tokens = []

#     for word in model.wv.vocab:
#         tokens.append(model[word])
#         labels.append(word)
    
#     tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
#     new_values = tsne_model.fit_transform(tokens)

#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
        
#     plt.figure(figsize=(16, 16)) 
#     for i in range(len(x)):
#         plt.scatter(x[i],y[i])
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()







