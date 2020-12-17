import sys
from Corpus import Corpus

corpus = Corpus()
dirr = input("Input directory of corpus:")
# dirr = "resources/corpus_mini.txt"
corpus.read(dirr)
corpus.preprocess()
corpus.ner()







