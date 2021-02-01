## This project used 100% python (v 3.7)

## Features:
### Corpus search tool:
Our tool can search in a corpus by:
- Ambiguous: you can search everything such as character, number, morpheme,...
- Noun (POS tagging)
- Verb (POS tagging)
- Adjective (POS tagging)
- Name of Person (NER model)
- Name of Location (NER model)
- Name of Organization (NER model)
- Show the top 10 similar words of your input (gensim word2Vec)


### Corpus dataset:
- I did web scrapping and got 12k description lines on vnexpress.net


## Libraries used:
- Dash + Dash bootstrap components
- Plotly 
- Gensim
- Underthesea (now Underthesea requires pytorch 1.4.0)
- nltk
- numpy
- pandas
- statsmodels

## How to run:
- Open terminal in the following directory: "Vietnamese-corpus-search-and-analysis-Web-app/"
### Using Corpus search app
- Run terminal "python src/app.py"
```console
python src/app.py
```
- Wait about 1 minute for the server, if you see the local host link in terminal, then ctrl click open it or copy and paste it into browser

### Using corpus statistical analysis app
- Run terminal
```console
python src_statistics/app.py
```
- Wait about 1 minute for the server, if you see the local host link in terminal, then ctrl click open it or copy and paste it into browser

### Using another corpus
- Rename your corpus file to "vn_express.txt" and replace it in resources/
- You have to run "python src/create_NER_pickle.py", then type in your corpus' directory: "resources/vn_express.txt" to build the NER model and Word2vec model, output as 2 files ner.pik and w2v.pik
- You only need to run once when using a new corpus

### Folders structure:
- docs/: documentation folder
	+ NLP.pptx: slides
- src/: source code of corpus search app
- src_statistics/: source code of corpus statistical analysis app
- resources/:
  - ner.pik: pickle file of NER model
  - w2v.pik: pickle file of Word2vec model
  - vn_express.txt: main corpus data
  - corpus_mini.txt: small 2k corpus for fast debugging 
  - stop_words.txt: File contains Vietnamese stopwords 

## Demo
### Corpus search tool
![image](https://drive.google.com/uc?export=view&id=1dainTJzqsBRTHOEOwKOAwuwoS4wPBhcb)
### Statistical analysis tool
![image](https://drive.google.com/uc?export=view&id=1UhwPVNJoBqmwD1vu_5CcvHoj8lnYFKxs)

