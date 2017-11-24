## Speed up Stanford NER tagger
# Version 2: Extract multiple questions per .txt file, up to 100,000

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import os
import time

###################################
# http://stackoverflow.com/questions/33748554/how-to-speed-up-ne-recognition-with-stanford-ner-with-python-nltk
###################################

# First download Stanford CoreNLP 3.5.2 from here: 
# http://nlp.stanford.edu/software/corenlp.shtml

# Lets say you put the download at /User/username/stanford-corenlp-full-2015-04-20

##########     Type A:   Create a text file for each question      ##########

# for question 1
# stanford_distribution_dir = "/Users/lingesther/Desktop/Gatech-School/Spring17/ECE6254-StatsML/TermProject/Code/stanford-corenlp-full-2016-10-31"
# list_of_sentences_path = "/Users/lingesther/Desktop/Gatech-School/Spring17/ECE6254-StatsML/TermProject/Code/list_of_sentences_1.txt"
# outputDirectory = "/Users/lingesther/Desktop/Gatech-School/Spring17/ECE6254-StatsML/TermProject/Code/stanford-corenlp-full-2016-10-31/sent1/json"
# stanford_command = "cd %s ; java -Xmx2g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -ssplit.eolonly -filelist %s -outputFormat json -outputDirectory %s" % (stanford_distribution_dir, list_of_sentences_path, outputDirectory)
# os.system(stanford_command)  # Execute the command (a string) in a subshell

## for question 2
# tStart = time.time()
# print "starting to process questions to json.txt ... "
# stanford_distribution_dir = "/Users/lingesther/Desktop/Gatech-School/Spring17/ECE6254-StatsML/TermProject/Code/stanford-corenlp-full-2016-10-31"
# list_of_sentences_path = "/Users/lingesther/Desktop/Gatech-School/Spring17/ECE6254-StatsML/TermProject/Code/list_of_sentences_2.txt"
# outputDirectory = "/Users/lingesther/Desktop/Gatech-School/Spring17/ECE6254-StatsML/TermProject/Code/stanford-corenlp-full-2016-10-31/sent2/json"
# stanford_command = "cd %s ; java -Xmx2g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -ssplit.eolonly -filelist %s -outputFormat json -outputDirectory %s" % (stanford_distribution_dir, list_of_sentences_path, outputDirectory)
# os.system(stanford_command)  # Execute the command (a string) in a subshell
# tEnd = time.time()
# print "done"
# print "time taken to process ", tEnd-tStart, "s"



##########    Type B:      Create a text file for all questions     ##########
def CreateTextFileQuestion1(textName):
	print "processing ", textName
	### Enter your base path here  ###
	basePath = "/Users/lingesther/Desktop/Gatech-School/Spring17/ECE6254-StatsML/TermProject/Code" 
	tStart = time.time()
	print "starting to process questions to json.txt ... "
	stanford_distribution_dir = "stanford-corenlp-full-2016-10-31"
	#### Change the text file name here
	# sentencePath = basePath + "/data/kaggle/sent1/Q1_2.txt"
	sentencePath = basePath + "/data/kaggle/sent1/"
	sentencePath = sentencePath + textName
	#### Change the output folder here
	outputDirectory = basePath + "/data/kaggle/sent1"
	stanford_command = "cd %s ; java -Xmx2g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -ssplit.eolonly -file %s -outputFormat json -outputDirectory %s" % (stanford_distribution_dir, sentencePath, outputDirectory)
	os.system(stanford_command)  # Execute the command (a string) in a subshell
	tEnd = time.time()
	print "done ", textName
	print "time taken to process ", tEnd-tStart, "s"


def CreateTextFileQuestion2(textName):
	print "processing ", textName
	### Enter your base path here  ###
	basePath = "/Users/lingesther/Desktop/Gatech-School/Spring17/ECE6254-StatsML/TermProject/Code" 
	tStart = time.time()
	print "starting to process questions to json.txt ... "
	stanford_distribution_dir = "stanford-corenlp-full-2016-10-31"
	#### Change the text file name here
	# sentencePath = basePath + "/data/kaggle/sent2/Q2_1.txt"
	sentencePath = basePath + "/data/kaggle/sent2/"
	sentencePath = sentencePath + textName
	#### Change the output folder here
	outputDirectory = basePath + "/data/kaggle/sent2"
	stanford_command = "cd %s ; java -Xmx2g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -ssplit.eolonly -file %s -outputFormat json -outputDirectory %s" % (stanford_distribution_dir, sentencePath, outputDirectory)
	os.system(stanford_command)  # Execute the command (a string) in a subshell
	tEnd = time.time()
	print "done ", textName
	print "time taken to process ", tEnd-tStart, "s"


### Process the next batch of 1 million questions
for i in range(24,31):
	fileNameQ1 = "Q1_" + str(i) + ".txt"
	fileNameQ2 = "Q2_" + str(i) + ".txt"
	# print "processing files ", fileNameQ1, "and", fileNameQ2
	CreateTextFileQuestion1(fileNameQ1)
	CreateTextFileQuestion2(fileNameQ2)
print "finished all questions"

## Here is some sample Python code for loading in a .json file for reference
# import json
# # sample_json = json.loads(file("sent1/json/q1.txt.json")).read()
# with open("data/sent1/Q1.txt.json") as f:
# 	z = json.load(f)

## At this point sample_json will be a nice dictionary with all the sentences from the file in it.
# i = 0
# for sentence in z["sentences"]:
# 	print i
# 	i = i+1
	# tokens = []
	# ner_tags = []
	# for token in sentence["tokens"]:
	# 	tokens.append(token["word"])
	# 	ner_tags.append(token["ner"])
	# print (tokens, ner_tags)


# list_of_sentences.txt should be your list of files with sentences, something like:
# input_file_1.txt
# input_file_2.txt
# ...
# input_file_100.txt

# So input_file.txt (which should have one sentence per line) will generate input_file.txt.json 
# once the Java command is run and that .json files will have the NER tags. 
# You can just load the .json for each input file and easily get 
# (sentence, ner tag sequence) pairs. You can experiment with "text" as an alternative 
# output format if you like that better. But "json" will create a nice .json file that 
# you can load in with json.loads(...) and then you'll have a nice dictionary that you 
# can use to access the sentences and annotations.

# This way you'll only load the pipeline once for all the files.







