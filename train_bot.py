#Text Data Preprocessing Lib
import nltk
from nltk.stem import PorterStemmer

stemmer= PorterStemmer()
import json
import pickle
import numpy as np

words=[]
classes=[]
word_tags_list=[]

ignore_words=['!','?',',','.',"'s","'m"]

train_data_file= open('intents.json').read()
intents=json.loads(train_data_file)



# function for appending stem words


def get_stem_words(words,ignore_words):
        stem_words=[]
        for word in words:
                if word not in ignore_words:
                        w= stemmer.stem(word.lower())
                        stem_words.append(w)
        return stem_words

    
        # Add all words of patterns to list
        
      
         

#Create word corpus for chatbot


def create_bot_corpus(words,classes,word_tags_list,ignore_words):
        for intent in intents['intents']:
                for pattern in intent['patterns']:
                        pattern_word= nltk.word_tokenize(pattern)
                        words.extend(pattern_word)
                        word_tags_list.append((pattern_word,intent['tag']))
  # Add all tags to the classes list
                if intent['tag'] not in classes:
                        classes.append(intent['tag'])
                        stem_words= get_stem_words(words,ignore_words)

create_bot_corpus(words,classes,word_tags_list,ignore_words)

print(classes)
print(word_tags_list)
