from mrakun import RakunDetector
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import glob
from random import randrange
import pytest


def test_basic_keywords1():

    hyperparameters = {
        "distance_threshold": 3,
        "num_keywords": 10,
        "pair_diff_length": 2,
        "distance_method": "editdistance",
        "stopwords": stopwords.words('english'),
        "bigram_count_threshold": 2,
        "num_tokens": [1]
    }

    keyword_detector = RakunDetector(hyperparameters)
    example_data = "./datasets/wiki20/docsutf8/7183.txt"
    keywords = keyword_detector.find_keywords(example_data)


def test_basic_visualization():

    hyperparameters = {
        "distance_threshold": 3,
        "num_keywords": 10,
        "pair_diff_length": 2,
        "stopwords": stopwords.words('english'),
        "distance_method": "editdistance",
        "bigram_count_threshold": 2,
        "num_tokens": [1]
    }

    keyword_detector = RakunDetector(hyperparameters)
    example_data = "./datasets/wiki20/docsutf8/7183.txt"
    keywords = keyword_detector.find_keywords(example_data)
    keyword_detector.visualize_network(display=False)


all_relevant = glob.glob("./datasets/wiki20/docsutf8/*")


@pytest.mark.parametrize("infile", all_relevant)
def test_basic_keywords2(infile):
    """
    A test across multiple files + parameter sets
    """

    for flx in all_relevant[0:10]:
        hyperparameters = {
            "distance_threshold": randrange(5),
            "num_keywords": randrange(10),
            "pair_diff_length": randrange(5),
            "distance_method": "editdistance",
            "stopwords": stopwords.words('english'),
            "bigram_count_threshold": randrange(5),
            "max_occurrence": randrange(10),
            "max_similar": randrange(4),
            "num_tokens": [1, 2, 3]
        }
        keyword_detector = RakunDetector(hyperparameters)
        keywords = keyword_detector.find_keywords(flx)
