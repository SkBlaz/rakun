from mrakun import *
## setup parameters
num_keywords = 10
lemmatizer = WordNetLemmatizer() ## any lemmatizer can be used instead of nltk default
bigram_count_threshold = 5 ## useful if n-gram keywords are looked for
stopwords = set(stpw.words('english')) # default english stopwords
word_length_difference = 3
edit_distance_threshold = 2
num_tokens = [1] ## this can also be [[1,2]] and [[1,2,3]] for bi and three gram keywords. (or just [2] or [3] for that matter)

keywords, visualization = find_rakun_keywords("datasets/wiki20/docsutf8/20782.txt",
                               limit_num_keywords=num_keywords,
                               lemmatizer = lemmatizer,
                               double_weight_threshold=bigram_count_threshold,
                               stopwords=stopwords,
                               pair_diff_length = word_length_difference,
                               edit_distance_threshold = edit_distance_threshold,
                               num_tokens = num_tokens,
                               visualize_network_keywords = True)

print(keywords)
plt.show()
