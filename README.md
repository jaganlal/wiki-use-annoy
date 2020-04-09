# wiki-use-annoy
Short wikipedia articles lookup using Google's USE (Universal Sentence Encoder) and Annoy (Approximate Nearest Neighbors Oh Yeah)

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


## References
https://medium.com/@vineet.mundhra/finding-similar-sentences-using-wikipedia-and-tensorflow-hub-dee2f52ed587
https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15

