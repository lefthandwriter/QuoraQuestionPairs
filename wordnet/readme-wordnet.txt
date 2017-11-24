The scripts in this folder extracts the word sense of verbs, nouns and adjectives in the two sentences, and compares how similar they are using similarity measures from the WordNet library. WordNet is a lexical database for English: https://wordnet.princeton.edu

Open source libraries utilized:
nltk - for both POS tags extractor and wordnet library.

Steps:
i) 	Extract Parts-of-Speech (POS) tags for each word in the two sentences, using the nltk library. The three POS tags we extracted were Verbs, Nouns and Adjectives. (Sidenote: Although another part of this project used the Stanford NLP library to extract POS tags, we found that this took a longer time compared to the nltk library.)

ii) Generate Sub-Sentences of Nouns, Verbs and Adjectives, for each sentence.

iii) Lookup word in Wordnet using the generated tags.

iv) Compute different measures of similarity between verb-verb, adjective-adjective and noun-noun pairs:
	path_similarity:
		- a score denoting how similar two word senses are, based on the shortest path that connects the senses in the is-a (hypernym/hypnoym) taxonomy
	lch_similarity:
		- a score denoting how similar two word senses are, based on the shortest path that connects the senses (as above) and the maximum depth of the taxonomy in which the senses occur. The relationship is given as -log(p/2d) where p is the shortest path length and d the taxonomy depth.
	Jiang-Conrath Similarity:
		- a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets. 
		The relationship is given by the equation 1 / (IC(s1) + IC(s2) - 2 * IC(lcs))


v) For each similarity score type, add the maximum scores in the sub-sentence vectors, to create a score for that sentence.

vi) Write the results to a .csv file.


Python Scripts:
1_myWordNet.py

