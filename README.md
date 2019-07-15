### RaKUn algorithm
This is the official repository of RaKUn. This keyword detection algorithm exploits graph-based language representations for efficient denoising and keyword detection.

## The core functionality
Packed as a Python library, it can be installed via:

```
pip3 install rakun
```

## Requirements
To install the required libraries, we suggest:
```
pip3 install -r requirements.txt
```

## Tests
To test whether the core functionality is ok, you can run
```
python3 -m pytest tests/test_core_functionality.py
```

## Usage
Using RaKUn is simple! Simply call the main detector method with optional arguments (as described in the paper)

```python
from rakun import *

## setup parameters
num_keywords = 10
lemmatizer = WordNetLemmatizer() ## any lemmatizer can be used instead of nltk default
bigram_count_threshold = 5 ## useful if n-gram keywords are looked for
stopwords = set(stpw.words('english')) # default english stopwords
word_length_difference = 3
edit_distance_threshold = 2
num_tokens = [1] ## this can also be [[1,2]] and [[1,2,3]] for bi and three gram keywords. (or just [2] or [3] for that matter)

## call the detector
keywords, _ = find_rakun_keywords("datasets/wiki20/docsutf8/20782.txt",
                                           limit_num_keywords=num_keywords,
                                           lemmatizer = lemmatizer,
                                           double_weight_threshold=bigram_count_threshold,
                                           stopwords=stopwords,
                                           pair_diff_length = word_length_difference,
                                           edit_distance_threshold = edit_distance_threshold,
                                           num_tokens = num_tokens_grid)
print(keywords)
```
Two results are returned. First one are keywords with corresponding centrality scores, e.g.,

```
[('data', 0.13913581780729972), ('processor', 0.08008456081980497), ('declustering', 0.062096783723763815), ('polygon', 0.060648271010513816), ('problem', 0.04348463689528755), ('method', 0.041232505193712056), ('using', 0.03989587664860619), ('query', 0.039519671621550546), ('architecture', 0.034608065395788895), ('application', 0.03342422281475986)]
```

while the second one is the plot of the keyword graph, which is only computed if visualize_network_keywords=True is passed.

```python
keywords, visualization = find_rakun_keywords("datasets/wiki20/docsutf8/20782.txt",
                                           limit_num_keywords=num_keywords,
                                           lemmatizer = lemmatizer,
                                           double_weight_threshold=bigram_count_threshold,
                                           stopwords=stopwords,
                                           pair_diff_length = word_length_difference,
                                           edit_distance_threshold = edit_distance_threshold,
                                           num_tokens = num_tokens_grid)
plt.show()					 

```

![Keyword graph](example_images/keywords.png)

### reproducing sota results
The code for reproducing sota results on datasets (in ./datasets/) is given in
```
examples/validation.py
```
