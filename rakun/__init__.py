## The RaKUn algorithm, Skrlj, Repar and Pollak 2019
"""
RaKUn is an algorithm for graph-absed keyword extraction.
"""
import itertools
import time
import nltk
import string
from nltk.corpus import stopwords as stpw
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
import operator
from nltk.stem import WordNetLemmatizer
import networkx as nx
import numpy as np
import glob
import editdistance
import os

import re
import pandas
from nltk.corpus import wordnet as wn
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
try:
    from py3plex.visualization.multilayer import *
except:
    pass

def visualize_network(network, name, centrality):

    print(nx.info(network))
    centrality = np.array([centrality[node] for node in network.nodes()])
    top_10 = list(centrality.argsort()[-20:][::-1])
    node_list = set(x for enx,x in enumerate(network.nodes()) if enx in top_10)
    rgba = ["red" if enx in set(top_10) else "black" for enx, x in enumerate(list(centrality))]
    labels = [x for x in network.nodes() if x in node_list]
    node_sizes = [8 if x in node_list else 2 for x in network.nodes()]

    # visualize the network's communities!
    hairball_plot(network,
                  labels = labels,
                  label_font_size = 10,
                  color_list=rgba,
                  text_color = "red",
                  node_sizes = node_sizes,
                  layout_parameters={"iterations": 50},
                  scale_by_size=True,
                  edge_width=0.08,
                  alpha_channel = 0.5,
                  layout_algorithm="force",
                  legend=False)

def generate_hypervertex(G,nodes, new_node, attr_dict=None):

    '''
    This node generates hypervertices.
    '''

    G.add_node(new_node,type="hyper")
    for n1,n2,data in list(G.edges(data=True)):
        if n1 in nodes:
            G.add_edge(new_node,n2,**data)
        elif n2 in nodes:
            G.add_edge(n1,new_node,**data)

    for n in nodes: # remove the merged nodes
        if n in G.nodes():
            G.remove_node(n)

def corpus_graph(language_file,limit_range=3000000,verbose=False,lemmatizer=None,stopwords=None, min_char = 3,stemmer=None):

    G = nx.DiGraph()
    ctx = 0
    reps = False
    dictionary_with_counts_of_pairs = {}
    with open(language_file) as lf:
        for line in lf:
            stop = list(string.punctuation)
            line = line.strip()
            line = [i for i in word_tokenize(line.lower()) if i not in stop]
            
            if not stopwords is None:
                line = [w for w in line if not w in stopwords]
                
            if not stemmer is None:
                line = [stemmer.stem(w) for w in line]
                
            if not lemmatizer is None:
                line = [lemmatizer.lemmatize(x) for x in line]
                
            line = [x for x in line if len(x) > min_char]
            if len(line) > 1:
                ctx+=1
                if ctx % 15000 == 0:
                    logging.info("Processed {} sentences.".format(ctx))
                if ctx % limit_range == 0:
                    break
                for enx, el in enumerate(line):     
                    if enx > 0:                            
                        edge_directed = (line[enx-1],el)
                        if edge_directed[0] != edge_directed[1]:
                            G.add_edge(edge_directed[0], edge_directed[1])
                        else:
                            edge_directed = None
                    if enx < len(line)-1:
                        edge_directed = (el,line[enx+1])
                        if edge_directed[0] != edge_directed[1]:
                            G.add_edge(edge_directed[0],edge_directed[1])
                        else:
                            edge_directed = None
                    if edge_directed:
                        if edge_directed in dictionary_with_counts_of_pairs:
                            dictionary_with_counts_of_pairs[edge_directed] += 1
                            reps = True
                        else:
                            dictionary_with_counts_of_pairs[edge_directed] = 1

    ## assign edge properties.
    for edge in G.edges(data=True):
        try:
            edge[2]['weight'] = dictionary_with_counts_of_pairs[(edge[0],edge[1])]
        except Exception as es:
            raise (es)
    if verbose:
        print(nx.info(G))
    return (G,reps)

def hypervertex_prunning(graph,edit_threshold,pair_diff_max = 2):
    
    to_merge = []
    stemmer =  nltk.stem.snowball.SnowballStemmer(language="english")
    for pair in itertools.combinations(graph.nodes(),2):
        if np.abs(len(pair[0]) - len(pair[1])) < pair_diff_max:
            ed = calculate_edit_distance(pair[0],pair[1])
            if ed < edit_threshold:
                to_merge.append(pair)
                new_node = stemmer.stem(pair[0])
                generate_hypervertex(graph,pair, new_node)

def find_rakun_keywords(document, limit_num_keywords = 10, lemmatizer=None,double_weight_threshold=2,stopwords = {"and"}, num_tokens = [1,2,3], edit_distance_threshold = 2, pair_diff_length = 2, visualize_network_keywords = False):

    all_terms = set()
    klens = {}
    weighted_graph,reps = corpus_graph(document,lemmatizer=lemmatizer,stopwords=stopwords)
    nn = len(list(weighted_graph.nodes()))
    if edit_distance_threshold > 0:
        hypervertex_prunning(weighted_graph, edit_distance_threshold, pair_diff_max = pair_diff_length)
    nn2 = len(list(weighted_graph.nodes()))
    logging.info("Number of nodes reduced from {} to {}".format(nn,nn2))
    pgx = nx.load_centrality(weighted_graph)
    keyword_plot = None
    if visualize_network_keywords:
        keyword_plot = visualize_network(weighted_graph,name = np.random.randint(312),centrality=pgx)
    keywords_with_scores = sorted(pgx.items(), key=operator.itemgetter(1),reverse=True)
    kw_map = dict(keywords_with_scores)
    if reps and 2 in num_tokens or 3 in num_tokens:
        
        higher_order_1 = []
        higher_order_2 = []
        frequent_pairs = []
        ## Check potential edges
        for edge in weighted_graph.edges(data=True):
            if edge[0] != edge[1]:
                if "weight" in edge[2]:
                    if edge[2]['weight'] > double_weight_threshold:
                        frequent_pairs.append(edge[0:2])
                        
        ## Traverse the frequent pairs
        for pair in frequent_pairs:        
            w1 = pair[0]
            w2 = pair[1]        
            if w1 in kw_map and w2 in kw_map:                               
                score = np.mean([kw_map[w1],kw_map[w2]])
                if not w1+" "+w2 in all_terms:
                    higher_order_1.append((w1+" "+w2,score))
                    all_terms.add(w1+" "+w2)

        ## Three word keywords are directed paths.
        three_gram_candidates = []
        for pair in frequent_pairs:
            for edge in weighted_graph.in_edges(pair[0]):
                if edge[0] in kw_map:
                    trip_score = [kw_map[edge[0]],kw_map[pair[0]],kw_map[pair[1]]]
                    term = edge[0] + " "  + pair[0] +" "+pair[1]
                    score = np.mean(trip_score)
                    if not term in all_terms:
                        higher_order_2.append((term, score))
                        all_terms.add(term)
                    
            for edge in weighted_graph.out_edges(pair[1]):
                if edge[1] in kw_map:
                    trip_score = [kw_map[edge[1]],kw_map[pair[0]],kw_map[pair[1]]]
                    term = pair[0] +" "+pair[1] +" "+edge[1]
                    score = np.mean(trip_score)
                    if not term in all_terms:
                        higher_order_2.append((term, score))
                        all_terms.add(term)
    else:
        higher_order_1 = []
        higher_order_2 = []
    
    total_keywords = []    
    if 1 in num_tokens:
        total_keywords += keywords_with_scores
    if 2 in num_tokens:
        total_keywords += higher_order_1
    if 3 in num_tokens:
        total_keywords += higher_order_2
    
    total_kws = sorted(set(total_keywords), key=operator.itemgetter(1),reverse=True)[0:limit_num_keywords]    
    
    return total_kws,keyword_plot

def calculate_edit_distance(key1, key2):
    return editdistance.eval(key1,key2)

def compare2gold (filename, keys_directory, keywords, fuzzy=False, fuzzy_threshold = 0.8,stemming=True,language = "en"):

    f = ".".join(filename.split('.')[:-1])
    try:
        f = f.replace(".txt","")
    except:
        pass
    if language == "en":            
        stemmer = nltk.stem.snowball.SnowballStemmer(language="english")
    goldKeys = []
    with open(keys_directory+f+'.key', 'r') as f:
        for line in f:
            goldKeys.append(stemmer.stem(line.strip()))
    c = 0
    matches = []
    if fuzzy == False:
        for el in keywords:
            stemmed_cand = stemmer.stem(el[0])
            if stemmed_cand in goldKeys:
                matches.append(el[0])
                c = c + 1
    else:
        for el in keywords:
            for gl in goldKeys:
                ed = calculate_edit_distance(el[0], gl)
                if ed > fuzzy_threshold:
                    matches.append(el[0]+'\t'+gl+'\n')
                    c = c + 1
                    break
    
    return c, len(goldKeys), matches

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def validate_on_corpus(datafolder):

    ## 500N-KPCrowd Blaž
    import glob
    import argparse
    
    ##################### parameters -- this is to be tweaked for each dataset
    lemmatizer_grid = WordNetLemmatizer()
    stopwords_grid = set(stpw.words('english'))
    stopwords_grid.add("might")
    stopwords_grid.add("also")
    stopwords_grid.add("use")
    num_keywords = 10
    num_tokens_grid = [1]#[[1,2,3],[1],[2],[2,3]] ## n-grams which will be considered for top k keywords
    count_threshold_grid = 3#[1,2,10] ## how common must a bigram be to be considered    
    final_dataframe = []
    edit_distance_threshold = 2
    pair_diff_length = 3
    #####################
    folds = 5     
    directory = datafolder
    opt_prec = 0
    optmimum_stup = []
    
    ## some form of grid search
    totalPrec = 0
    totalRcl = 0
    totalGoldSize = 0
    all_docs = os.listdir(directory+'/docsutf8')
    part_size = int(len(all_docs)/folds)
    chunker = chunks(all_docs,part_size)
    parts = [x for x in chunker][0:folds]

    ## use train to get the hyperparams.
    all_f_scores = []
    for j in range(len(parts)):
        logging.info("Working with fold {}".format(j))
        train_corpora = []
        test_corpora = []

        for enx, el in enumerate(parts):
            if enx != j:
                for px in el:                    
                    train_corpora.append(px)
            else:
                for px in el:
                    test_corpora.append(px)

        counts = 0
        totalGsize = 0
        total_ks = 0
        for filename in test_corpora:
            start = time.time()
            keywords,_ = find_rakun_keywords(directory+'/docsutf8/'+filename,
                                           limit_num_keywords=num_keywords,
                                           lemmatizer = lemmatizer_grid,
                                           double_weight_threshold=count_threshold_grid,
                                           stopwords=stopwords_grid,
                                           pair_diff_length = pair_diff_length,
                                           edit_distance_threshold = edit_distance_threshold,
                                           num_tokens = num_tokens_grid)

            count, goldSize, matches = compare2gold(filename, directory+'/keys/', keywords)
            print(keywords)

            counts += count
            if goldSize > num_keywords:
                goldSize = num_keywords
            totalGsize += goldSize
            total_ks += num_keywords

            try:
                precision = float(counts)/total_ks
                recall = float(counts)/totalGsize
                F1 = 2* (precision * recall)/(precision + recall)
                end = time.time() - start
                logging.info("Intermediary F1: {}".format(F1))

                counts += count
                totalGsize += goldSize
                total_ks += num_keywords
            except:
                print("Could not compute")
        all_f_scores.append(F1)
        optimum_setup = [precision,recall,F1,directory,pair_diff_length,edit_distance_threshold,count_threshold_grid,num_tokens_grid]
        print("RESULT_LINE"+"\t"+"\t".join([str(x) for x in optimum_setup]))
