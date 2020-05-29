# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:39:05 2020

@author: hp
"""

from rank_bm25 import BM25Okapi

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

corpus = [
    "Reviewer Assignment Problem (RAP) is an important issue in peer-review of academic\
writing. This issue directly influences the quality of the publication and as such is the\
brickwork of scientific authentication. Due to the obvious limitations of manual assignment,\
automatic approaches for RAP is in demand. In this paper, we conduct a survey\
on those automatic approaches appeared in academic literatures. In this paper, regardless\
of the way reviewer assignment is structured, we formally divide the RAP into\
three phases: reviewer candidate search, matching degree computation, and assignment\
optimization. We find that current research mainly focus on one or two phases, but obviously,\
these three phases are correlative. For each phase, we describe and classify the\
main issues and methods for addressing them. Methodologies in these three phases have\
been developed in a variety of research disciplines, including information retrieval, artificial\
intelligence, operations research, etc. Naturally, we categorize different approaches\
by these disciplines and provide comments on their advantages and limitations. With an\
emphasis on identifying the gaps between current approaches and the practical needs, we\
point out the potential future research opportunities, including integrated optimization,\
online optimization, etc.",

"Automatic expert assignment is a common problem encountered in both industry and academia. For example, for conference program chairs and journal editors, in order to\
collect good judgments for a paper it is necessary for them to assign the paper to the most appropriate reviewers Choosing appropriate reviewers of course includes a number of considerations such as expertise and authority, but also\
diversity and avoiding conflicts. In this paper, we explore the expert retrieval problem and implement an automatic paper-reviewer recommendation system that considers aspects of expertise, authority, and diversity. In particular,\
a graph is first constructed on the possible reviewers and the query paper, incorporating expertise and authority information. Then a Random Walk with Restart (RWR) model is employed on the graph with a sparsity constraint, incorporating diversity information. Extensive experiments\
on two reviewer recommendation benchmark datasets show that the proposed method obtains performance gains over state-of-the-art reviewer recommendation systems in terms of expertise, authority, diversity, and, most importantly, relevance as judged by human experts.",

"There are a number of issues which are involved with organizing a conference. Among these issues, assigning conference-papers to reviewers is one of the most difficult tasks. Assigning conference-papers to reviewers is automatically the most crucial part. In this paper, we address this issue of paper-to-reviewer assignment, and we propose a method to\
model the reviewers, based on the matching degree between the reviewers and the papers by combining a preference-based approach and a topic-based approach. We explain the assignment algorithm and show the evaluation results in comparison with the Hungarian algorithm.",

"Refereed conferences require every submission to be reviewed by members of a program committee (PC) in charge of selecting the conference program. There are many software packages available to manage the review process. Typically, in a bidding phase PC members express their personal preferences by ranking the submissions. This information is used by the system to compute an assignment of the papers to referees (PC members). We study the problem of assigning papers to referees. We propose to optimize a number of criteria that aim at achieving fairness among referees/papers. Some of these variants can be solved optimally in polynomial time, while others are NP-hard, in which case we design approximation algorithms. Experimental results strongly suggest that the assignments computed by our algorithms are considerably better than  those computed by popular conference management software.",

"The 117 manuscripts submitted for the Hypertext’91 conference were assigned to members of the review committee, using a variety of automated methods based on information retrieval principles and Latent Semantic Indexing. Fifteen reviewers provided exhaustive ratings for the submitted abstracts, indicating how well each abstract matched their interests. The automated methods do a fairly good job of assigning relevant papers for review, but they are still somewhat poorer than assignments made manually by human experts and substantially poorer than an assignment perfectly matching the reviewers’ own ranking of the papers. A new automated assignment method called n of 2n achieves better performance than human experts by sending reviewers more papers than they actually have to review and then allowing them to choose part of their review load themselves.",

"An essential part of an expert-
nding task, such as matching reviewers to submitted papers, is the ability to model the expertise of a person based on documents. We evaluate several measures of the association between a document to be re- viewed and an author, represented by their previous papers. We compare language-model-based approaches with a novel topic model, Author-Persona-Topic (APT). In this model, each author can write under one or more personas, which are represented as independent distributions over hidden topics. Examples of previous papers written by prospective reviewers are gathered from the Rexa database, which extracts and disambiguates author mentions from documents gathered from the web. We evaluate the models using a reviewer matching task based on human relevance judgments determining how well the expertise of proposed reviewers matches a submission. We 
nd that the APT topic model outperforms the other models."


]

tokenized_corpus = [doc.split(" ") for doc in corpus]


bm25 = BM25Okapi(tokenized_corpus)



#Ranking the documents
query = "Peer review has become the most common practice for judging papers submitted to a conference for decades. An extremely important task involved in peer review is to assign submitted papers to reviewers with appropriate expertise which is referred to as paper-reviewer assignment. In this paper, we study the paper-reviewer assignment problem from both the goodness aspect and the fairness aspect. For the goodness aspect, we propose to maximize the topic coverage of the paper-reviewer assignment. This objective is new and the problem based on this objective is shown to be NP-hard. To solve this problem efficiently, we design an approximate algorithm which gives a 1 3 - approximation. For the fairness aspect, we perform a detailed study on conflict-of-interest (COI) types and discuss several issues related to using COI, which, we hope, can raise some open discussions among researchers on the COI study. Finally, we conducted experiments on real datasets which verified the effectiveness of our algorithm and also revealed some interesting results of COI."
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

bm25.get_top_n(tokenized_query, corpus, n=1)




corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

#Ranking the documents
query = "windy London"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

bm25.get_top_n(tokenized_query, corpus, n=1)
