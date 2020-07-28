from mrakun import RakunDetector
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from os import path

if not path.exists("../pretrained_models/fasttext/wiki.en.bin"):
    print(
        "Please, load a fasttext english pretrained model binary into folder ../pretrained_models."
    )

hyperparameters = {
    "distance_threshold": 0.2,
    "distance_method": "fasttext",
    "pretrained_embedding_path": '../pretrained_models/fasttext/wiki.en.bin',
    "num_keywords": 10,
    "pair_diff_length": 2,
    "stopwords": stopwords.words('english'),
    "bigram_count_threshold": 2,
    "lemmatizer": WordNetLemmatizer(),
    "num_tokens": [1]
}

keyword_detector = RakunDetector(hyperparameters)
example_data = "../datasets/wiki20/docsutf8/7183.txt"
keywords = keyword_detector.find_keywords(example_data)
print(keywords)
keyword_detector.visualize_network()
