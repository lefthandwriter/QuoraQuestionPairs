## Pick out all question pairs and write to individual .txt file
## To use for Stanford NER tagger
## Version 2: Extract multiple questions per .txt file, up to 100,000

# list number of files: ls | wc -l
import pandas as pd 
import time
# data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
## Write the full question list to a batched .txt file
# data[0:100000].text1.to_csv('data/sent1/Q1_1.txt')
# data[100000:200000].text1.to_csv('data/sent1/Q1_2.txt')
# data[200000:300000].text1.to_csv('data/sent1/Q1_3.txt')
# data[300000:].text1.to_csv('data/sent1/Q1_4.txt')

# data[0:100000].text2.to_csv('data/sent2/Q2_1.txt')
# data[100000:200000].text2.to_csv('data/sent2/Q2_2.txt')
# data[200000:300000].text2.to_csv('data/sent2/Q2_3.txt')
# data[300000:].text2.to_csv('data/sent2/Q2_4.txt')

data = pd.read_csv("data/kaggle/test.csv")

### Process the first 1 million questions
# data[0:100000].question1.to_csv('data/kaggle/sent1/Q1_1.txt')
# data[100000:200000].question1.to_csv('data/kaggle/sent1/Q1_2.txt')
# data[200000:300000].question1.to_csv('data/kaggle/sent1/Q1_3.txt')
# data[300000:400000].question1.to_csv('data/kaggle/sent1/Q1_4.txt')
# data[400000:500000].question1.to_csv('data/kaggle/sent1/Q1_5.txt')
# data[500000:600000].question1.to_csv('data/kaggle/sent1/Q1_6.txt')
# data[600000:700000].question1.to_csv('data/kaggle/sent1/Q1_7.txt')
# data[700000:800000].question1.to_csv('data/kaggle/sent1/Q1_8.txt')
# data[800000:900000].question1.to_csv('data/kaggle/sent1/Q1_9.txt')
# data[900000:1000000].question1.to_csv('data/kaggle/sent1/Q1_10.txt')

# data[0:100000].question2.to_csv('data/kaggle/sent2/Q2_1.txt')
# data[100000:200000].question2.to_csv('data/kaggle/sent2/Q2_2.txt')
# data[200000:300000].question2.to_csv('data/kaggle/sent2/Q2_3.txt')
# data[300000:400000].question2.to_csv('data/kaggle/sent2/Q2_4.txt')
# data[400000:500000].question2.to_csv('data/kaggle/sent2/Q2_5.txt')
# data[500000:600000].question2.to_csv('data/kaggle/sent2/Q2_6.txt')
# data[600000:700000].question2.to_csv('data/kaggle/sent2/Q2_7.txt')
# data[700000:800000].question2.to_csv('data/kaggle/sent2/Q2_8.txt')
# data[800000:900000].question2.to_csv('data/kaggle/sent2/Q2_9.txt')
# data[900000:1000000].question2.to_csv('data/kaggle/sent2/Q2_10.txt')


### Process the second 1 million questions
# length = 100000
# a = 11
# print "processing the second 1 million questions..."
# for i in range(1000000,2000000, length):
# 	filename_Q1 = "data/kaggle/sent1/Q1_" + str(a) + ".txt"
# 	filename_Q2 = "data/kaggle/sent2/Q2_" + str(a) + ".txt"
# 	data[i:i+length].question1.to_csv(filename_Q1)
# 	data[i:i+length].question2.to_csv(filename_Q2)
# 	# print filename
# 	# print i
# 	# print i + length
# 	a += 1
# print "done!"


### Process the third 1 million questions
length = 100000
a = 21
print "processing the second 1 million questions..."
for i in range(2000000,3000000, length):
	filename_Q1 = "data/kaggle/sent1/Q1_" + str(a) + ".txt"
	filename_Q2 = "data/kaggle/sent2/Q2_" + str(a) + ".txt"
	data[i:i+length].question1.to_csv(filename_Q1)
	data[i:i+length].question2.to_csv(filename_Q2)
	# print filename
	# print i
	# print i + length
	a += 1
print "done!"

# data[0:100000].text1.to_csv('data/sent1/Q1_1.txt', index=None) # to write without the index


# # create list of sentences file
# textarray = []
# for i in range(1, len(data)+1):
# 	textfilename = 'sent1/'
# 	textfilename =  textfilename + 'q' + str(i) + '.txt'
# 	textarray.append(textfilename)
# data['q1text'] = textarray
# # data['q2text'] = textarray
# data.q1text.to_csv('list_of_sentences_1.txt', index=False)
# data.q2text.to_csv('list_of_sentences_2.txt', index=False)

# with open('sent2/q201841.txt', "w") as text_file:
# 	text_file.write(data.text2[201841])

## Process Question 1:
# tStart = time.time()
# i = 1
# print('starting to write... ')
# for sentence in data.text1:
# 	filename = 'sent1/q' + str(i) + '.txt'
# 	if pd.isnull(data.text1[i-1]) == True:
# 		with open(filename, "w") as text_file:
# 			text_file.write('nan')
# 	else:
# 		with open(filename, "w") as text_file:
# 			text_file.write(sentence)
# 	i += 1
# tEnd = time.time()
# print "time taken to process ", len(data), "lines: ", tEnd-tStart

## Process Question 2:
# for sentence in data.text2:
# 	filename = 'sent2/q' + str(i) + '.txt'
# 	if pd.isnull(data.text2[i-1]) == True:
# 		with open(filename, "w") as text_file:
# 			text_file.write('nan')
# 	else:
# 		with open(filename, "w") as text_file:
# 			text_file.write(sentence)
# 	i += 1
# 	# print "done sentence ", i