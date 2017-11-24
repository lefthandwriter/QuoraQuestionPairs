
**** Note: The code files above still need cleaning to be more comprehensible ****
Date: 7 Apr 2017

### ###
The code in here is to extract Parts of Speech (POS) tags and Named Entity Taggers (NER) from the sentences. It uses the Stanford NLP library.

There are three files, to be run in order:
1_sentence2txt.py
2_extractjson.py
3_stanfordfeatures.py

### ###

Steps:

1. First download Stanford CoreNLP 3.5.2 from here: 
	http://nlp.stanford.edu/software/corenlp.shtml

	Save this file to a folder.

	(Pre-reqs, need Java 1.8+)
	A how to install Java 1.8:
	https://www3.ntu.edu.sg/home/ehchua/programming/howto/JDK_Howto.html#zz-2

2. Run 1_sentence2txt.py. This file extracts the questions from the csv file and saves as a separate .txt file. For example, if there are 400,000 question pairs, then 400,000 x 2 .txt files will be created.


Then, run the code to create the questions list.

3. When (1) has finished downloading, run 2_extractjson.py. You will need to edit the path name based on which folder you saved (1) to. This file processes all the .txt files created in (2), extracts the TOKENS, POS, NER, LEMMA, creates and saves them into an equivalently named .json file.

4. Run 3_stanfordfeatures.py. This creates the embeddings based on the .json files created in step (3). For around 800,000 questions, this takes approximately 5 minutes.


