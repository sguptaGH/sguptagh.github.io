# my jupter notebook as blog post
Creating subtitle

### imports


```python
import numpy as np
from glob import glob
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### basic function to pre-process text data 
#### convert to lowercase, remove punctuation and stopwords, stemming


```python
PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = stopwords.words('english')
stemmer = PorterStemmer()

def preprocess(line):
    line.lower()
    line = "".join([char for char in line if char not in PUNCT_TO_REMOVE])
    line = " ".join([word for word in line.split() if word not in STOPWORDS])
    line = " ".join([stemmer.stem(word) for word in line.split()])
    return line
```

### create corpus


```python
corpus = []
title = []
title_tag = []
for file in glob("literature/*/*"):
    f = open(file, "r")
    lines = [line.rstrip('\n') for line in f.readlines()]
    title.append(file.split('\\')[2])
    title_tag.append(file.split('\\')[1])
    corpus.append(preprocess(' '.join(lines)))
```

### generate the tf-idf , vector representation of articles


```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
```

### function to recommend most similar and category wise similar documents


```python
def recommendation(article):
    idx = title.index(article)
    similarity = cosine_similarity(X[idx], X)[0]
    most_recommended = title[np.argsort(similarity)[-2]]
    cat_recommendation = ''
    for tag in set(title_tag):
        doc_ids = np.where(np.array(title_tag) == tag)[0]
        if idx in doc_ids:
            selected = np.argsort(similarity[doc_ids])[-2]
        else:
            selected = np.argsort(similarity[doc_ids])[-1]
        cat_recommendation = cat_recommendation+"cat : "+tag+" , Title : "+title[doc_ids[selected]]+"\n"
        
    return "Most Recommended : "+most_recommended+"\n\nCategory Wise Recommendation : \n"+cat_recommendation+"\n\n"
```

### call the recommentation function with the article that has been liked/read to fetch similar results


```python
print(recommendation("The_Ass_and_the_Lapdog.txt"))
```

    Most Recommended : The_Man_and_the_Serpent.txt
    
    Category Wise Recommendation : 
    cat : Fables , Title : The_Man_and_the_Serpent.txt
    cat : Shakespeare , Title : HenryV.txt
    
    
    

