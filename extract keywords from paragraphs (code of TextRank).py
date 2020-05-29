# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:46:42 2020

@author: hp
"""

import spacy
import pytextrank
import re
import nltk
import string
import en_core_web_sm

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english')) 

nlp = en_core_web_sm.load()

tr = pytextrank.TextRank()

nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)

#text='An accelerometer based handwriting recognition of English alphabets using basic strokes. This paper focuses on an efficient method to recognize the English alphabets written in real time. For a perfect documented work or even for the formal written communication, a writer is expected to minimize the literature mistakes, i.e., spelling mistakes, grammatical errors, punctuation mark misplaced, etc. It would be interesting to device some technique if such mistakes could be recognized by non-human intervention in case of handwriting. Hence, a training-less system has been developed that is capable of recognizing the English alphabets in the real time and thereby the words, and make the suggestions to the writer regarding the wrong word. We have designed a digital pen which has an accelerometer and uses an efficient algorithm for the detection of the English letters. The technique is simple, based on the waveform analysis of the strokes made while writing a letter, the digital pen recognized the letters with the accuracy of 84% which after inclusion of the nearest mapping algorithm leads to the efficiency of over 96%. In this paper we discuss results only pertaining to recognition of capital alphabets which is not written in cursive writing style.'
text='An accelerometer based handwriting recognition of English alphabets using basic strokes.'

text=text.lower()                   #convert to lower case

   #remove digits
text = "".join([c for c in text if c not in string.punctuation])      #remove punctuations


doc = nlp(text)

z=[]

for p in doc._.phrases:
    print('{:.4f} {:5d}  {}'.format(p.rank, p.count, p.text))
    print(p.chunks)
    z=z+[p.text]


filtered_sentence = [] 
  
for w in z: 
    if (w not in stop_words): 
        filtered_sentence.append(w)


#https://github.com/DerwenAI/pytextrank/blob/master/explain_algo.ipynb
for chunk in doc.noun_chunks:
    print(chunk.text)


import networkx as nx

def increment_edge (graph, node0, node1):
    print("link {} {}".format(node0, node1))
    
    if graph.has_edge(node0, node1):
        graph[node0][node1]["weight"] += 1.0
    else:
        graph.add_edge(node0, node1, weight=1.0)

POS_KEPT = ["ADJ", "NOUN", "PROPN", "VERB"]
#POS_KEPT = ["ADJ", "NOUN", "PROPN"]
def link_sentence (doc, sent, lemma_graph, seen_lemma):
    visited_tokens = []
    visited_nodes = []

    for i in range(sent.start, sent.end):
        token = doc[i]

        if token.pos_ in POS_KEPT:
            key = (token.lemma_, token.pos_)

            if key not in seen_lemma:
                seen_lemma[key] = set([token.i])
            else:
                seen_lemma[key].add(token.i)

            node_id = list(seen_lemma.keys()).index(key)

            if not node_id in lemma_graph:
                lemma_graph.add_node(node_id)

            print("visit {} {}".format(visited_tokens, visited_nodes))
            print("range {}".format(list(range(len(visited_tokens) - 1, -1, -1))))
            
            for prev_token in range(len(visited_tokens) - 1, -1, -1):
                print("prev_tok {} {}".format(prev_token, (token.i - visited_tokens[prev_token])))
                
                if (token.i - visited_tokens[prev_token]) <= 3:
                    increment_edge(lemma_graph, node_id, visited_nodes[prev_token])
                else:
                    break

            print(" -- {} {} {} {} {} {}".format(token.i, token.text, token.lemma_, token.pos_, visited_tokens, visited_nodes))

            visited_tokens.append(token.i)
            visited_nodes.append(node_id)


for sent in doc.sents:
    print(">", sent.start, sent.end)
    
lemma_graph = nx.Graph()
seen_lemma = {}

for sent in doc.sents:
    link_sentence(doc, sent, lemma_graph, seen_lemma)
    #break # only test one sentence

print(seen_lemma)


labels = {}
keys = list(seen_lemma.keys())

for i in range(len(seen_lemma)):
    labels[i] = keys[i][0].lower()

labels


#%matplotlib inline
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 9))
pos = nx.spring_layout(lemma_graph)

nx.draw(lemma_graph, pos=pos, with_labels=False, font_weight="bold")
nx.draw_networkx_labels(lemma_graph, pos, labels)



ranks = nx.pagerank(lemma_graph)
ranks


for node_id, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
    print(node_id, rank, labels[node_id])


def collect_phrases (chunk, phrases, counts):
    chunk_len = chunk.end - chunk.start + 1
    sq_sum_rank = 0.0
    non_lemma = 0
    compound_key = set([])

    for i in range(chunk.start, chunk.end):
        token = doc[i]
        key = (token.lemma_, token.pos_)
        
        if key in seen_lemma:
            node_id = list(seen_lemma.keys()).index(key)
            rank = ranks[node_id]
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
    

phrases = {}
counts = {}

for chunk in doc.noun_chunks:
    collect_phrases(chunk, phrases, counts)

for ent in doc.ents:
    collect_phrases(ent, phrases, counts)


import operator

min_phrases = {}

for compound_key, rank_tuples in phrases.items():
    l = list(rank_tuples)
    l.sort(key=operator.itemgetter(1), reverse=True)
    
    phrase, rank = l[0]
    count = counts[compound_key]
    
    min_phrases[phrase] = (rank, count)
    
for phrase, (rank, count) in sorted(min_phrases.items(), key=lambda x: x[1][0], reverse=True):
    print(phrase, count, rank)