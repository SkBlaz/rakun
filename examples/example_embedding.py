from mrakun import RakunDetector
from nltk.corpus import stopwords
## https://fasttext.cc/docs/en/pretrained-vectors.html Download the model from here (bin)
hyperparameters = {"distance_threshold":0.2,
                   "distance_method": "fasttext",
                   "pretrained_embedding_path": '../pretrained_models/fasttext/wiki.en.bin', #change path accordingly
                   "num_keywords" : 10,
                   "pair_diff_length":2,
                   "stopwords" : stopwords.words('english'),
                   "bigram_count_threshold":2,
                   "num_tokens":[1]}

keyword_detector = RakunDetector(hyperparameters)
example_data = "../datasets/wiki20/docsutf8/7183.txt"
keywords = keyword_detector.find_keywords(example_data)
print(keywords)
