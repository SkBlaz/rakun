# RaKUn algorithm

```
@misc{krlj2019language,
    title={Language comparison via network topology},
    author={Blaž Škrlj and Senja Pollak},
    year={2019},
    eprint={1907.06944},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

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

## Usage
Using RaKUn is simple! Simply call the main detector method with optional arguments (as described in the paper)

```python
from mrakun import RakunDetector
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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
print(keywords)

keyword_detector.verbose = False
## do five fold CV on a given corpus (results for each fold need to be aggregated!)
keyword_detector.validate_on_corpus("./datasets/Schutz2008")
```
Two results are returned. First one are keywords with corresponding centrality scores, e.g.,

```
[('system', 0.07816398111051845), ('knowledg', 0.07806649568191038), ('structur', 0.05978269674796454), ('diagnost', 0.041354892225684135), ('problem', 0.04052608382511414), ('domain', 0.031261954442705624), ('intellig', 0.030126748143180067), ('medic', 0.026760043407613995), ('caus', 0.02639800122506548), ('method', 0.026292347276388087)]
```

And the plot of the keyword graph:

```python
## once the find_keywords() was invoked
keyword_detector.visualize_network()
```

![Keyword graph](example_images/keywords.png)

## Acknowledgements
The logo was created by Luka Skrlj

# Citation
If you use this, please cite:
(Accepted at SLSP2019)
TBA
