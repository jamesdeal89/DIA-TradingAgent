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
            intents.append(row)
    return intents
download('stopwords', quiet=True)
p_stemmer = PorterStemmer()

def stemmedWithFilteredStopWords(doc):
    tokens = re.findall('\\b\\w+\\b', doc.lower())
    return [p_stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]

def stemmingVectorisationWeighting(intents):
    prompts = []
    for pair in intents:
        prompts.append(pair[0])
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
    dVecs = XTrainTf.toarray()
    norms = [float(norm(dVec)) for dVec in dVecs]
    indexWithNorms = {'index': invIdx, 'norms': norms}
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
    similarity = []
    for docId, dotprod in accumulator.items():
        denom = queryNorm * norms[docId]
        score = dotprod / denom if denom != 0 else 0.0
        similarity.append((docId, score))
    similarity.sort(key=lambda x: x[1], reverse=True)
    if not similarity:
        return None
    elif similarity[0][1] > 0.6:
        return similarity[0]
    else:
        return None