import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.util import ngrams
import nltk

class LanguageModel:
    
    def __init__(self,k,b,l1,l2,l3):
        
        self.k = k
        self.b = b
        self.lambda1 = l1
        self.lambda2 = l2
        self.lambda3 = l3
        self.Probabilities = {}
        
    pass

    def get_corpus(self,dataset):
        
        Corpus = dataset.split()
        
        return Corpus
    
    def create_N_grams(self,Corpus):
        
        unigrams = list(ngrams(Corpus,1))

        bigrams = list(ngrams(Corpus,2))

        trigrams = list(ngrams(Corpus,3))
        
        return unigrams, bigrams, trigrams
    
    
    def get_unique_N_grams(self,n_grams):
        
        unique_N_grams = []
        for gram in n_grams:
            if gram in unique_N_grams:
                continue
            else:
                unique_N_grams.append(gram)
                
        return unique_N_grams
    
    
    #calculating Conditional Probability of Unigrams given the length of cache k and decay speed b
    def calculate_CP_unigram(self,word,corpus,k,b):
        prob = 0.0
        count = 0.0
        for i in range(len(corpus)):
            if(i-k >= 0):
                for j in range(i-k,i-1):
                    x = -1*b*(i-j)
                    e = np.exp(x)                            
                    if (corpus[i] == corpus[j]):
                        prob += e
                    count += e                 
        
        if count > 0.0:
            CP = prob / count
        else:
            CP = 0.0

        return CP
    
    #calculating Conditional Probability of bigrams given the length of cache k and decay speed b
    def calculate_CP_bigram(self,word,corpus,k,b):    
        count = 0.0
        prob = 0.0  
        for i in range(len(corpus)):
            if(corpus[i] == word[1]):
                for j in range(i-k,i-1):
                    x = -1*b*(i-j)
                    e = np.exp(x)
                    if(word[0] == corpus[j] and word[1] == corpus[j+1]):
                        prob += e
                    if(word[0] == corpus[j]):
                        count += e
        if count > 0.0:
            CP = prob / count
        else:
            CP = 0.0

        return CP
    
    
    #calculating Conditional Probability of trigrams
    def calculate_CP_trigram(self,trigram,bigrams,trigrams):

        freq_of_trigram = trigrams.count(trigram)

        bigram = (trigram[0],trigram[1])

        freq_of_bigram = bigrams.count(bigram)

        CP = freq_of_trigram / freq_of_bigram

        return CP

    
    
    def train(self, dataset):
        
        Corpus = self.get_corpus(dataset)
        
        unigrams, bigrams, trigrams = self.create_N_grams(Corpus)
        
        unique_trigrams = self.get_unique_N_grams(trigrams)

        for i in range(len(unique_trigrams)):

            unigram = trigrams[i][2]
            CP_unigram = self.calculate_CP_unigram(unigram,Corpus,self.k,self.b)

            bigram = (trigrams[i][1],trigrams[i][2])
            CP_bigram = self.calculate_CP_bigram(bigram,Corpus,self.k,self.b)

            CP_trigram = self.calculate_CP_trigram(trigrams[i],bigrams,trigrams)

            probability = (self.lambda1 * CP_unigram) + (self.lambda2 * CP_bigram) + (self.lambda3 * CP_trigram)

            self.Probabilities[trigrams[i]] = probability

            
    def predict(self):
        
        starting_trigram = max(self.Probabilities.iteritems(), key=operator.itemgetter(1))[0]
        
        content = starting_trigram[0]
        content += " "
        content += starting_trigram[1]
        content += " "
        content +=  starting_trigram[2]

        bigram = (starting_trigram[1], starting_trigram[2])
        
        max_value = 0
        
        for i in range(0,40):
            max_value = 0
            for key,value in self.Probabilities.iteritems():
                temp = (key[0],key[1])
                if temp == bigram:
                    if value > max_value:   
                        max_value = value 
                        content += " "
                        content += key[2]
                        bigram = (key[1],key[2])
                        
        return content




fileReader = open("Reviews.txt","r")

dataset = fileReader.read()

dataset.replace(". "," </s> <s> ")
dataset.replace(".\n"," </s> <s> ")
dataset.replace("\n"," </s> <s> ")

# Hyper-Parameters

k = 300  #cache size

b = 10 # Decay speed

lambda1 = 0.1

lambda2 = 0.3

lambda3 = 0.6


LM = LanguageModel(k,b,lambda1,lambda2,lambda3)

LM.train(dataset)

# testing the model by generating content

content = LM.predict()

print "Content : ",content