# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:19:04 2020

@author: hp
"""

import spacy
import re
import nltk
import string
import en_core_web_sm
import itertools, string

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

nlp = en_core_web_sm.load()
nlp=spacy.load("en_core_web_sm")

POS_KEPT = ["ADJ", "NOUN", "PROPN", "VERB"]

#good_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS','VB','VBN','VBD','VBG','VBZ'])  #Allowed POS tags
stop_words = set(nltk.corpus.stopwords.words('english'))

punct = list(string.punctuation)
punct.remove('.')

        # tokenize and POS-tag words
def findtermfrequency(doc):
    lemmawords=[]
    countoccurence=[]
    for sent in doc.sents:
        print(">", sent.start, sent.end)
        for i in range(sent.start,sent.end):
            token=doc[i]
            if token.pos_ in POS_KEPT:
                if(token.lemma_ not in lemmawords):
                    lemmawords+=[token.lemma_]
                    countoccurence+=[1]
                else:
                    countoccurence[lemmawords.index(token.lemma_)]+=1
    return lemmawords,countoccurence



def addedge(adjMatrix,source,dest,lemmawords):
    print("added ",source, "  ", dest)
    adjMatrix[lemmawords.index(source)][lemmawords.index(dest)]+=1
    
    return adjMatrix



def generategraph(doc,lemmawords):
    adjMatrix=[]
    for i in range(len(lemmawords)):
            adjMatrix.append([0 for i in range(len(lemmawords))])
    
    for sent in doc.sents:
        print(">", sent.start, sent.end)
            
        for i in range(sent.start,sent.end):
            token=doc[i]
            if(str(token) in punct):
                continue
            else:
                if(token.lemma_ in lemmawords):
                    
                    for countocc in range(1,4):
                        if((i+countocc) < sent.end and (i+countocc) < sent.end and doc[i+countocc].lemma_ in lemmawords):
                            adjMatrix=addedge(adjMatrix,str(doc[i].lemma_),str(doc[i+countocc].lemma_),lemmawords)
                            
                else:
                    continue
    
    return (adjMatrix)
        
        
def ModifiedPageRank(adjMatrix,countoccurence,titleend,doc):
    initialrank,finalrank=[1/len(adjMatrix)]*len(adjMatrix),[0]*len(adjMatrix)
    #print(doc)
    titlenodes=0
    for i in range(titleend):
        if(doc[i].lemma_ in lemmawords):
            titlenodes+=1
    #print(titlenodes)
    primarycount,secondarycount=0,0
    for i in range(titlenodes):
        primarycount+=countoccurence[i]
    for i in range(titlenodes,len(lemmawords)):
        secondarycount+=countoccurence[i]
    
    print(primarycount,secondarycount)
    residue=(primarycount/(primarycount+secondarycount))*initialrank[0]
    
    for i in range(titlenodes,len(lemmawords)):   #decrease weight of the secondary nodes
        initialrank[i]-=residue
    
    residue*=(len(lemmawords)-titlenodes)
    residue/=titlenodes 
    
    for i in range(titlenodes):        #increase weight of the primary nodes
        initialrank[i]+=residue
    
    for i in range(len(adjMatrix)):
        for j in range(len(adjMatrix)):
            if(adjMatrix[i][j]!=0):
                num_of_edges=0
                for k in range(len(adjMatrix)):
                    num_of_edges+=adjMatrix[j][k]
                if(num_of_edges!=0):
                    finalrank[i]=finalrank[i]+(initialrank[j]/num_of_edges)
                    num_of_edges=0
                
        
    
    return finalrank       
    

def UndirectedMatrix(adjMatrix):
    for i in range(len(adjMatrix)):
        for j in range(i,len(adjMatrix)):
            a=adjMatrix[i][j]
            b=adjMatrix[j][i]
            adjMatrix[i][j]=a+b
            adjMatrix[j][i]=a+b
    return adjMatrix
    

def collect_phrases (chunk, phrases, counts):
    print(chunk)
    chunk_len = chunk.end - chunk.start + 1
    sq_sum_rank = 0.0
    non_lemma = 0
    compound_key = set([])

    for i in range(chunk.start, chunk.end):
        token = doc[i]
        key = (token.lemma_)
        
        if key in lemmawords:
            node_id = lemmawords.index(key)
            rank = RankScore[node_id]
            sq_sum_rank += rank
            compound_key.add(key)
        
            print(" {} {} {} {}".format(token.lemma_, token.pos_, node_id, rank))
        else:
            non_lemma += 1
    
    # although the noun chunking is greedy, we discount the ranks using a
    # point estimate based on the number of non-lemma tokens within a phrase
    import math
    non_lemma_discount = chunk_len / (chunk_len + (2.0 * non_lemma) + 1.0)

    # use root mean square (RMS) to normalize the contributions of all the tokens
    phrase_rank = math.sqrt(sq_sum_rank / (chunk_len + non_lemma))
    phrase_rank *= non_lemma_discount

    # remove spurious punctuation
    phrase = chunk.text.lower().replace("'", "")

    # create a unique key for the the phrase based on its lemma components
    compound_key = tuple(sorted(list(compound_key)))
    
    if not compound_key in phrases:
        phrases[compound_key] = set([ (phrase, phrase_rank) ])
        counts[compound_key] = 1
    else:
        phrases[compound_key].add( (phrase, phrase_rank) )
        counts[compound_key] += 1

    print("{} {} {} {} {} {}".format(phrase_rank, chunk.text, chunk.start, chunk.end, chunk_len, counts[compound_key]))
    

text=abstract.lower()                   #convert to lower case
doc=str(text)
for i in range(len(doc)):
    if(doc[i] in punct):
        doc=doc.replace(doc[i]," ")
doc = nlp(text)

lemmawords,countoccurence=findtermfrequency(doc)
adjMatrix=generategraph(doc,lemmawords)
adjMatrix=UndirectedMatrix(adjMatrix)
for sent in doc.sents:
    print(">", sent.start, sent.end)
    break
    
RankScore=ModifiedPageRank(adjMatrix,countoccurence,sent.end,doc)
    
Ranked_Lemma=sorted(((RankScore[i], lemmawords[i]) for i, s in enumerate(RankScore)),reverse=True)


text=abstract.lower()                   #convert to lower case
doc=nlp(text)    
phrases = {}
counts = {}
    
for chunk in doc.noun_chunks:
    print(chunk)
    collect_phrases(chunk, phrases, counts)
        
import operator
    
min_phrases = {}
    
for compound_key, rank_tuples in phrases.items():
   l = list(rank_tuples)
   l.sort(key=operator.itemgetter(1), reverse=True)
        
   phrase, rank = l[0]
   count = counts[compound_key]
   min_phrases[phrase] = (rank, count)

print()
print("*****************keyterms******************")

for phrase, (rank, count) in sorted(min_phrases.items(), key=lambda x: x[1][0], reverse=True):
    print(phrase, count, rank)
    
    
abstract='Sensor Based Hand Gesture Recognition System for English Alphabets Used in Sign Language of Deaf-Mute People. Hand gesture and sign language are significant ways of communication for deaf-mute people. It puts a barrier in comprehension of a conversation between a mute person and a normal person, because a normal person does not understand the sign language. In this paper we developed a sensor based device which deciphers this sign language of hand gesture for English alphabets. We propose that if this wearable device, which is a hand glove, is put on by a mute person, the device would recognize the 26 letters almost accurately. We discuss the challenges and future potential of this device so that it would completely be able to facilitate communication of such class of people. Hand gesture recognition is also a challenging problem of the human computer interface area.'
abstract='An accelerometer based handwriting recognition System of English alphabets using basic strokes. This paper focuses on an efficient method to recognize the English alphabets written in real time. For a perfect documented work or even for the formal written communication, a writer is expected to minimize the literature mistakes, spelling mistakes, grammatical errors, punctuation mark misplaced, etc. It would be interesting to device some technique if could be recognized by non-human intervention in case of handwriting. Hence, a training-less system has been developed that is capable of recognizing the English alphabets in the real time and thereby the words, and make the suggestions to the writer regarding the wrong word. We have designed a digital pen which has an accelerometer and uses an efficient algorithm for the detection of the English letters. The technique is simple, based on the waveform analysis of the strokes made while writing a letter, the digital pen recognized the letters with the accuracy of 84% which after inclusion of the nearest mapping algorithm leads to the efficiency of over 96%. In this paper we discuss results only pertaining to recognition of capital alphabets which is not written in cursive writing style.'
abstract='Use of discrete wavelet transform method for detection and localization of tampering in a digital medical image. Use of digital images has increased tremendously in medical science and with that has increased the query on authenticity of the image. Authenticity of the digital image is very important in the area of scientific research, forensic investigations, government documents, etc. With the help of powerful and user friendly image editing software like Microsoft Paint and Photoshop it became extremely easy to tamper with a digital image for malicious objective. Of late this problem has been encountered in medical imaging also for the purpose of fake insurance claims. We propose an algorithm to address this problem by which one can detect and localize tampering in a digital medical image. This algorithm is based on hash based representation of such image and uses discrete wavelet transform method to carry out detection and localization of tampering. We will show that our algorithm is robust against harmless manipulation and sensitive enough for even a minute tampering. Our proposed technique consumes less resource as it works with smaller hash function in comparison with the similar available techniques.'
abstract='Superpixel-Driven Optimized Wishart Network for Fast PolSAR Image Classification Using Global k -Means Algorithm. Limitation of optical remote sensing technology gave rise to synthetic aperture radar (SAR) imaging. SAR is a microwave imaging technique, which promises to have a long-range propagation characteristic allowing imaging under harsh weather conditions or in hostile lighting situation. This has opened up a domain of classification using polarimetric SAR (PolSAR) images. In this article, we propose a fast PolSAR image classification algorithm, which uses not only pixel-based feature but also spatial features around each pixel. This is achieved by introducing superpixel-driven optimized Wishart network. The first improvement suggested in this article is to take advantage of a fast global k-means algorithm for obtaining optimal cluster centers within each class. It uses real-valued vector representation of PolSAR coherency matrix along with fast matrix inverse and determinant algorithms to reduce computational overhead. Our method then exploits the information of neighboring pixels by forming a superpixel so that even a noisy pixel may not be assigned a wrong class label. The proposed network uses dual-branch architecture to efficiently combine pixel and superpixel features. We concluded that our proposed method has better efficiency in terms of classification accuracy and computational overhead compared with other deep learning-based methods available in the literature.'
abstract='Classification of Polarimetric Synthetic Aperture Radar Images Using Revised Wishart Distance. Synthetic Aperture Radar (SAR) is an advanced active radar imaging technology. It uses polarized electromagnetic waves to capture images of the earth surface. In this paper we have proposed polarimetric synthetic aperture radar (PolSAR) image classification technique. Proposed technique uses single hidden layer neural network to achieve classification task. We have proposed linearization model of revised Wishart distance such that it can be used for training the network. It is used along with k-means algorithm to calculate initial weights of network prior to training. Pre-calculating weights helps network to converge quickly with high classification accuracy. Performance evaluation of proposed network is conducted on NASA/JPL Airborne Synthetic Aperture Radar (AIRSAR) data acquired over Flevoland in Netherlands. It achieves 93.01% overall classification accuracy on Flevoland dataset.'
abstract="An Improved PageRank Algorithm Based on Reachability to Reduce Mutual Reinforcement Effect. It is very difficult to find relevant information in this rapidly growing hyper structure. Basically web site owner creates web site to provide relevant information to cater user's need. Therefore, it is very important to retrieve web pages based on user's interest and behavior by finding the content of the web page. Link analysis is the significant parameter to find related information. PageRank and HITS are two basic algorithms which works on web structure mining. Numerous algorithms have been developed to improve performance based on these two basic algorithms. An Improved PageRank Algorithm (IPRA), is the extension of the pageRank algorithm, is introduced in this paper. IPRA takes into account the number of different domain inlink and outlink and distributes the rank score based on the reachability of the web pages. The result of our model studies shows that IPRA algorithms give better result than the standard PageRank algorithm."
    
#lemmawords,keyterms,Ranked_Lemma,adjMatrix=startfunction(abstract)


    

#-----------------print adjMATRIX-------------------
for i in range (len(adjMatrix)):
    for j in range(len(adjMatrix)):
        print (adjMatrix[i][j], end=' ')
    print()
    
    
    