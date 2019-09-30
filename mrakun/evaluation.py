## a set of methods for evaluation of keyword quality.
import editdistance
import numpy as np

def compare_with_gold_fuzzy(detected_keywords, gold_standard_keywords, fuzzy_threshold = 3, keyword_separator = ";"):


    """
    Fuzzy comparison of keyword matches. Given a fuzzy edit distance threshold, how many  keywords out of the top 10 are OK?
    input: detected_keywords (list of string).
    input: gold_standard_keywords (list of strings).
    input: fuzzy_threshold (int) -> max acceptable edit distance.    
    """

    precision_correct = 0
    precision_overall = 0

    recall_correct = 0
    recall_overall = 0

    for enx, keyword_set in enumerate(detected_keywords):

        gold_standard_set = gold_standard_keywords[enx]
        count = 0
        method_keywords = keyword_set.split(keyword_separator)

        if type(gold_standard_set) is float: ## this is np.nan -> not defined.
            continue
        
        gold_standard_set = set(gold_standard_set.split(keyword_separator))

        top_n = len(gold_standard_set)
        if top_n >= len(method_keywords):
            top_n = len(method_keywords)

        ## recall
        parsed_rec = set()
        for el in method_keywords:
            if not el in parsed_rec:
                parsed_rec.add(el)
                if el in gold_standard_set:
                    recall_correct += 1
        recall_overall += top_n

        ## precision
        parsed_prec = set()
        for el in method_keywords:
            if not el in parsed_prec:
                parsed_prec.add(el)
                if el in gold_standard_set:
                    precision_correct += 1
        precision_overall += len(method_keywords)                
        
    precision = float(precision_correct) / (precision_overall) ## Number of correctly predicted over all predicted (num gold)
    
    recall = float(recall_correct) / (recall_overall) ## Correct over all detected keywords
    
    if (precision + recall) > 0:
        F1 = 2* (precision * recall)/(precision + recall)
        
    else:
        F1 = 0
        
    return precision, recall, F1
