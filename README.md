# wiki-use-annoy
Short wikipedia articles lookup using Google's USE (Universal Sentence Encoder) and Annoy (Approximate Nearest Neighbors Oh Yeah)

### Note: Not the entire wikipedia articles lookup ;). Checkout the disclaimer below

## Usage

### Model Download
1. Download the `universal-sentence-encoder-large` model using `download-use.py` script

    `python download-use.py`

### Build Annoy Index
2. Build annoy index for the `short-wiki.csv` file

    `python build-short-wiki-annoy-index.py`

### Find Similarities
3. Find the similarties by providing the id

    `python find-similar-wiki-articles.py`

Key in the id, say for example `music-wikipedia`. 
You'll see the following results (in the form of id for similicity)
```
pop-wikipedia
guitar-wikipedia
brain-wikipedia
world-wikipedia
science-wikipedia
malayalam-wikipedia
sourashtra-wikipedia
apple-wikipedia
usa-wikipedia
```


## Disclaimer
I started to create (short-wiki.csv) a short intro on some of the articles (source: wikipedia) about places, people, culture etc. So this application will lookup from that articles. Checkout `short-wiki.csv` for more information on this. You can imagine this as a cleaned up data lookup. If you want to contribute (either code or data part), please feel free to fork it and create a PR.

## References
https://medium.com/@vineet.mundhra/finding-similar-sentences-using-wikipedia-and-tensorflow-hub-dee2f52ed587
https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15

