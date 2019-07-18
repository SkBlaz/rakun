from mrakun import RakunDetector
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def test_basic_keywords():

    hyperparameters = {"edit_distance_threshold":3,
                       "num_keywords" : 10,
                       "pair_diff_length":2,
                       "stopwords" : stopwords.words('english'),
                       "bigram_count_threshold":2,
                       "lemmatizer" : WordNetLemmatizer(),
                       "num_tokens":[1]}

    keyword_detector = RakunDetector(hyperparameters)
    example_data = "./datasets/wiki20/docsutf8/7183.txt"
    keywords = keyword_detector.find_keywords(example_data)


def test_basic_visualization():

    hyperparameters = {"edit_distance_threshold":3,
                       "num_keywords" : 10,
                       "pair_diff_length":2,
                       "stopwords" : stopwords.words('english'),
                       "bigram_count_threshold":2,
                       "lemmatizer" : WordNetLemmatizer(),
                       "num_tokens":[1]}

    keyword_detector = RakunDetector(hyperparameters)
    example_data = "./datasets/wiki20/docsutf8/7183.txt"
    keywords = keyword_detector.find_keywords(example_data)    
    keyword_detector.visualize_network()
