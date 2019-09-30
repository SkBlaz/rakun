# RaKUn algorithm

<img src="example_images/rakun.png" width="300" height="300">
This is the official repository of RaKUn. This keyword detection algorithm exploits graph-based language representations for efficient denoising and keyword detection.
Key ideas of RaKUn:
1. Transform texts to graphs
2. Prune graphs based on token similarity (meta vertex introduction)
3. Rank nodes -> keywords

[paper] (https://arxiv.org/abs/1907.06458)

## The core functionality
Packed as a Python library, it can be installed via:

```
pip3 install mrakun
```

or

```
python3 setup.py install
```

## Requirements
To install the required libraries, we suggest:
```
pip3 install -r requirements.txt
```
For visualization, one also needs Py3plex library (>=0.66). (pip3 install py3plex)

## Tests
To test whether the core functionality is ok, you can run
```
python3 -m pytest tests/test_core_functionality.py
```

## Usage with editdistance
Using RaKUn is simple! Simply call the main detector method with optional arguments (as described in the paper)

```python
from mrakun import RakunDetector
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

hyperparameters = {"distance_threshold":3,
                   "num_keywords" : 10,
                   "distance_method": "editdistance",
                   "pair_diff_length":2,
                   "stopwords" : stopwords.words('english'),
                   "bigram_count_threshold":2,
                   "lemmatizer" : WordNetLemmatizer(),
                   "num_tokens":[1]}

keyword_detector = RakunDetector(hyperparameters)
example_data = "./datasets/wiki20/docsutf8/7183.txt"
keywords = keyword_detector.find_keywords(example_data)
print(keywords)

keyword_detector.verbose = False
## do five fold CV on a given corpus (results for each fold need to be aggregated!)
keyword_detector.validate_on_corpus("./datasets/Schutz2008")
```
Two results are returned. First one are keywords with corresponding centrality scores, e.g.,

```
[('knowledge', 0.09681271338743186), ('system', 0.07564796507791872), ('level', 0.05109912821797258), ('model', 0.04258209551663402), ('domain', 0.04148756282878477), ('task', 0.03965601439030798), ('structure', 0.03819960122342131), ('1989', 0.03287462707574183), ('diagnosis', 0.03236736673125384), ('method', 0.030969444684564095)]
```

And the plot of the keyword graph:

```python
## once the find_keywords() was invoked
keyword_detector.visualize_network()
```

![Keyword graph](example_images/keywords.png)

## Usage with fasttext
Using RaKUn with fasttext requires pretrained emmbeding model. Download .bin file from https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md for chosen language and save it.

```python
from mrakun import RakunDetector
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

hyperparameters = {"distance_threshold":0.2,
                   "distance_method": "fasttext",
                   "pretrained_embedding_path": '../pretrained_models/fasttext/wiki.en.bin', #change path accordingly
                   "num_keywords" : 10,
                   "pair_diff_length":2,
                   "stopwords" : stopwords.words('english'),
                   "bigram_count_threshold":2,
                   "lemmatizer" : WordNetLemmatizer(),
                   "num_tokens":[1]}

keyword_detector = RakunDetector(hyperparameters)
example_data = "./datasets/wiki20/docsutf8/7183.txt"
keywords = keyword_detector.find_keywords(example_data)
print(keywords)

keyword_detector.verbose = False
## do five fold CV on a given corpus (results for each fold need to be aggregated!)
keyword_detector.validate_on_corpus("./datasets/Schutz2008")
```
Two results are returned. First one are keywords with corresponding centrality scores, e.g.,

```
[('structure', 0.013043478260869565), ('model', 0.010507246376811594), ('level', 0.009420289855072464), ('knowledge', 0.006400966183574879), ('system', 0.005595813204508856), ('application', 0.004589371980676328), ('inference', 0.002657004830917874), ('reasoning', 0.002536231884057971), ('abstraction', 0.0020933977455716585), ('problem', 0.0020933977455716585)]
```

And the plot of the keyword graph:

```python
## once the find_keywords() was invoked
keyword_detector.visualize_network()
```

![Keyword graph](example_images/keywords2.png)


## Acknowledgements
The logo was created by Luka Skrlj

# Citation
If you use this, please cite:
(Accepted at SLSP2019)
TBA
