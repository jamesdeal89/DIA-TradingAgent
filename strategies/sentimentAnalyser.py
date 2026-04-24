"""
Sentiment Analysis Utilities for Agent Strategies.
Provides sentiment analysis functions for agents to apply to headline data from StockExchange.getNewsForStock().
"""
import numpy as np
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch
logger = logging.getLogger(__name__)

class SentimentAnalyser:
    """Runtime FinBERT scorer and sentiment aggregation utility."""

    modelName = 'ProsusAI/finBERT'
    fineTunedPath = './models/finbert_finetuned'

    def __init__(self, modelPath=None):
        self.modelPath = modelPath or self.fineTunedPath
        self.deviceId = 0 if torch.cuda.is_available() else -1
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._loadModel()

    def _loadModel(self):
        loadPath = self.modelPath if os.path.exists(self.modelPath) else self.modelName
        if loadPath == self.modelPath:
            logger.info(f'[Sentiment] Loading fine-tuned FinBERT from {self.modelPath}')
        else:
            logger.info(f'[Sentiment] Loading pre-trained FinBERT from {self.modelName}')
        self.tokenizer = AutoTokenizer.from_pretrained(loadPath)
        self.model = AutoModelForSequenceClassification.from_pretrained(loadPath)
        self.pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=self.deviceId)

    def scoreHeadline(self, headline):
        """Score one headline and return sentiment, score, and confidence."""
        try:
            result = self.pipeline(headline, truncation=True, max_length=512)
            label = result[0]['label']
            modelScore = float(result[0]['score'])
            if label == 'positive':
                score = modelScore
            elif label == 'negative':
                score = -modelScore
            else:
                score = 0.0
            return {'sentiment': label, 'score': score, 'confidence': modelScore}
        except Exception as error:
            logger.warning(f'[Sentiment] Failed to score headline: {error}')
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}

    def enrichHeadlines(self, headlines):
        """Fill missing sentiment fields in headline dicts (needed if not cached previously.)"""
        enrichedHeadlines = []
        for headlineData in headlines:
            sentiment = headlineData.get('sentiment')
            score = headlineData.get('score')
            if sentiment is None or score is None:
                result = self.scoreHeadline(headlineData.get('headline', ''))
                enriched = dict(headlineData)
                enriched['sentiment'] = result['sentiment']
                enriched['score'] = result['score']
                enriched['confidence'] = result['confidence']
                enriched['sentimentInferred'] = True
                enrichedHeadlines.append(enriched)
            else:
                enriched = dict(headlineData)
                enriched['sentimentInferred'] = False
                if enriched.get('confidence') is None:
                    enriched['confidence'] = 0.0
                enrichedHeadlines.append(enriched)
        return enrichedHeadlines

    def analyse(self, headlines):
        """
        Aggregate sentiment analysis from headline data.

        headlines: list of headline dicts with sentiment/score fields.

        Returns a dict of aggregated sentiment metrics.
       """
        if not headlines:
            return {'totalHeadlines': 0, 'positiveCount': 0, 'neutralCount': 0, 'negativeCount': 0, 'avgSentiment': 0.0, 'sentimentStd': 0.0, 'confidence': 0.0, 'positiveRatio': 0.0, 'negativeRatio': 0.0, 'headlines': []}
        total = len(headlines)
        scores = []
        positiveCount = 0
        neutralCount = 0
        negativeCount = 0
        for headline in headlines:
            sentiment = headline.get('sentiment') if headline.get('sentiment') else 'neutral'
            scoreValue = headline.get('score')
            score = float(scoreValue) if scoreValue is not None else 0.0
            scores.append(score)
            if sentiment == 'positive':
                positiveCount += 1
            elif sentiment == 'negative':
                negativeCount += 1
            else:
                neutralCount += 1
        avgSentiment = np.mean(scores) if scores else 0.0
        sentimentStd = np.std(scores) if len(scores) > 1 else 0.0
        nonNeutral = positiveCount + negativeCount
        confidence = nonNeutral / total if total > 0 else 0.0
        positiveRatio = positiveCount / total if total > 0 else 0.0
        negativeRatio = negativeCount / total if total > 0 else 0.0
        return {'totalHeadlines': total, 'positiveCount': positiveCount, 'neutralCount': neutralCount, 'negativeCount': negativeCount, 'avgSentiment': round(avgSentiment, 4), 'sentimentStd': round(sentimentStd, 4), 'confidence': round(confidence, 4), 'positiveRatio': round(positiveRatio, 4), 'negativeRatio': round(negativeRatio, 4), 'headlines': headlines}


