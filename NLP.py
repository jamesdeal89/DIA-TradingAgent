import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem.snowball import PorterStemmer
from nltk import download
from nltk.corpus import stopwords
from collections import defaultdict
import re
from numpy.linalg import norm


def readIntentsCSV():
    intents = []
    with open('intents.csv', 'r', encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            # formatted as prompt,intent.
            intents.append(row)
    return intents

download('stopwords', quiet=True)
p_stemmer = PorterStemmer()
def stemmedWithFilteredStopWords(doc):
    tokens = re.findall(r'\b\w+\b', doc.lower())
    return [p_stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]

def stemmingVectorisationWeighting(intents):
    # Vectorise training prompts.
    prompts = []
    for pair in intents: 
        prompts.append(pair[0])
    # Stem, remove stop-words, and run a count-based vectoriser on the prompts.
    countVect = CountVectorizer(tokenizer=stemmedWithFilteredStopWords, lowercase=True)
    XTrainCounts = countVect.fit_transform(prompts)
    tfTransformer = TfidfTransformer(use_idf=True, sublinear_tf=True, norm=None).fit(XTrainCounts)
    xTrainTf = tfTransformer.transform(XTrainCounts)
    return (xTrainTf, countVect, tfTransformer)

def createFloatDict():
    return defaultdict(float)

def genInvertedIndex(countVect, XTrainTf):
    invIdx = defaultdict(createFloatDict)
    featNames = countVect.get_feature_names_out()
    tfidfMat = XTrainTf.tocoo()
    for docId, termId, score in zip(tfidfMat.row, tfidfMat.col, tfidfMat.data):
        term = featNames[termId]
        invIdx[term][docId] = score

    # Precompute document level l2 norms to save doing it later.
    dVecs = XTrainTf.toarray()
    # list of floats: norm of each document vector
    norms = [float(norm(dVec)) for dVec in dVecs]

    indexWithNorms = {
        'index': invIdx,
        'norms': norms
    }
    return indexWithNorms

    
def searchIntent(indexWithNorms, prompt, vectoriser, tfidf):
    invIdx = indexWithNorms['index']
    norms = indexWithNorms['norms']
    queryCounts = vectoriser.transform([prompt])
    queryTfidf = tfidf.transform(queryCounts)
    queryCoo = queryTfidf.tocoo()
    queryVector = queryTfidf.toarray().flatten()
    queryNorm = norm(queryVector)
    
    accumulator = defaultdict(float)
    feature_names = vectoriser.get_feature_names_out()
    for termId, q_weight in zip(queryCoo.col, queryCoo.data):
        term = feature_names[termId]
        postings = invIdx.get(term)
        if not postings:
            continue
        for docId, d_weight in postings.items():
            accumulator[docId] += q_weight * d_weight

    # Calculate cosine-based similarity for documents in accumulator
    similarity = []
    for docId, dotprod in accumulator.items():
        denom = queryNorm * norms[docId]
        score = dotprod / denom if denom != 0 else 0.0
        similarity.append((docId, score))

    # Return the predicted intent based on most similar training prompt's label,
    # or None if no match.
    similarity.sort(key=lambda x: x[1], reverse=True)
    if not similarity:
        return None
    elif similarity[0][1] > 0.6:
        return similarity[0]
    else:
        return None
