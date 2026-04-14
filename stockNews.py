"""
Stock News Sentiment Analysis Pipeline.

Downloads the Kaggle stock news dataset, extracts raw headlines, fine-tunes FinBERT
for financial sentiment classification, and caches sentiment scores for fast access
during agent trading simulations.

Architecture:
1. Download Kaggle dataset via kagglehub
2. Extract raw_partner_headlines.csv (6 columns: index, headline, URL, publisher, date, stock_ticker)
3. Fine-tune ProsusAI/finBERT on the dataset headlines
4. Generate sentiments.csv cache (index, sentiment, score, confidence)
5. Populate MySQL news_headlines table in StockExchange

Usage:
    from stockNews import NewsAnalyser, initialiseNewsDatabase
    from stockExchange import StockExchange
    
    # Initialise
    analyser = NewsAnalyser()
    exchange = StockExchange()
    
    # Fine-tune FinBERT once (cached after first run)
    analyser.fineTune()
    
    # Populate MySQL with sentiment data
    initialiseNewsDatabase(exchange, analyser)
    
    # Query headlines during simulation
    news_metrics = exchange.getNewsForStock('AAPL', 'XNAS', '2015-01-15')
"""
import os
import csv
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import kagglehub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
import torch
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

class NewsAnalyser:
    """
    Sentiment analysis engine using fine-tuned FinBERT model.
    
    Handles:
    - Building fine-tuned FinBERT from pre-trained weights
    - Classifying headline sentiment (positive/neutral/negative)
    - Caching results to sentiments.csv for deterministic simulation runs
    - Mapping headlines back to tickers and dates for query efficiency
    """
    MODEL_NAME = 'ProsusAI/finBERT'
    FINE_TUNED_PATH = './models/finbert_finetuned'
    SENTIMENTS_CACHE = 'sentiments.csv'
    RAW_HEADLINES_FILE = 'raw_partner_headlines.csv'
    DEVICE = 0 if torch.cuda.is_available() else -1

    def __init__(self, batch_size=16, num_epochs=3):
        """
        Initialise NewsAnalyser.
        
        Args:
            batch_size: Batch size for fine-tuning (reduce if memory-constrained)
            num_epochs: Number of fine-tuning epochs
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'NewsAnalyser initialised. Using device: {self.device}')
        logger.info(f'Batch size: {batch_size}, Epochs: {num_epochs}')
        self._loadOrInitModel()

    def _loadOrInitModel(self):
        """Load fine-tuned model if exists, otherwise load pre-trained FinBERT."""
        if os.path.exists(self.FINE_TUNED_PATH):
            logger.info(f'Loading fine-tuned model from {self.FINE_TUNED_PATH}')
            self.tokenizer = AutoTokenizer.from_pretrained(self.FINE_TUNED_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.FINE_TUNED_PATH)
        else:
            logger.info(f'Loading pre-trained FinBERT from {self.MODEL_NAME}')
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=self.DEVICE)

    def getSentiment(self, headline):
        """
        Classify sentiment of a single headline.
        
        Args:
            headline: Text of the headline to analyse
            
        Returns:
            Dict with keys:
            - sentiment: 'positive', 'neutral', or 'negative'
            - score: float in range [-1.0, 1.0] (-1=negative, 0=neutral, 1=positive)
            - confidence: float in range [0.0, 1.0] (model confidence)
        """
        try:
            result = self.pipeline(headline, truncation=True, max_length=512)
            label = result[0]['label']
            model_score = float(result[0]['score'])
            if label == 'positive':
                score = model_score
            elif label == 'negative':
                score = -model_score
            else:
                score = model_score * 0.1
            logger.debug(f'[FinBERT] {label:8} (raw={model_score:.4f}) score={score:.4f} | {headline[:60]}')
            return {'sentiment': label, 'score': score, 'confidence': model_score}
        except Exception as e:
            logger.error(f"Error classifying headline '{headline[:50]}...': {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}

    def fineTune(self, headlines_df=None, num_samples=None, batch_chunk_size=50000, enable_fine_tuning=False):
        """
        Batch-based fine-tuning of FinBERT on financial headlines.
        
        Processes headlines in chunks to fit within GPU memory constraints.
        Uses gradient accumulation and smaller per-device batch sizes for efficiency.
        
        Args:
            headlines_df: Optional DataFrame with headlines. If None, loads from RAW_HEADLINES_FILE
            num_samples: Limit to first N samples (optional)
            batch_chunk_size: Number of headlines to process per batch (default 50k)
            enable_fine_tuning: Set to True to actually fine-tune. Default False (use pre-trained)
        
        Returns:
            True if fine-tuning completed, False if skipped
        """
        if not enable_fine_tuning:
            logger.info('Fine-tuning DISABLED by default: Using pre-trained FinBERT (already trained on financial data)')
            logger.info('To enable batched fine-tuning, call: analyser.fineTune(enable_fine_tuning=True)')
            logger.info('This will process headlines in 50k batches with gradient accumulation')
            return False
        if headlines_df is None:
            if not os.path.exists(self.RAW_HEADLINES_FILE):
                logger.error(f'{self.RAW_HEADLINES_FILE} not found')
                return False
            logger.info(f'Loading headlines from {self.RAW_HEADLINES_FILE}')
            headlines_df = pd.read_csv(self.RAW_HEADLINES_FILE)
        if num_samples is not None:
            headlines_df = headlines_df.head(num_samples)
        total_headlines = len(headlines_df)
        logger.info(f'Fine-tuning on {total_headlines} headlines in batches of {batch_chunk_size}')
        try:
            for chunk_idx, start_idx in enumerate(range(0, total_headlines, batch_chunk_size)):
                end_idx = min(start_idx + batch_chunk_size, total_headlines)
                chunk_df = headlines_df.iloc[start_idx:end_idx]
                chunk_size = len(chunk_df)
                logger.info(f'\n[Batch {chunk_idx + 1}] Processing headlines {start_idx}-{end_idx} ({chunk_size} headlines)')
                dataset = Dataset.from_pandas(chunk_df[['headline']].reset_index(drop=True))

                def tokenize_function(examples):
                    return self.tokenizer(examples['headline'], padding='max_length', truncation=True, max_length=512)
                logger.info(f'  [1/3] Tokenizing batch {chunk_idx + 1}...')
                tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=1000)
                training_args = TrainingArguments(output_dir=f'{self.FINE_TUNED_PATH}/checkpoint-batch-{chunk_idx}', num_train_epochs=1, per_device_train_batch_size=8, gradient_accumulation_steps=8, save_steps=500, save_total_limit=1, logging_steps=100, log_level='warning', use_cpu=False if torch.cuda.is_available() else True)
                logger.info(f'  [2/3] Training on batch {chunk_idx + 1}...')
                trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized_dataset)
                trainer.train()
                logger.info(f'  [3/3] Batch {chunk_idx + 1} complete')
            os.makedirs(self.FINE_TUNED_PATH, exist_ok=True)
            logger.info(f'\nSaving fine-tuned model to {self.FINE_TUNED_PATH}')
            self.model.save_pretrained(self.FINE_TUNED_PATH)
            self.tokenizer.save_pretrained(self.FINE_TUNED_PATH)
            logger.info('Fine-tuning completed successfully!')
            return True
        except Exception as e:
            logger.error(f'Error during fine-tuning: {e}')
            logger.info('Continuing with current model state...')
            return False

    def generateSentimentCache(self, force=False, batch_size=64):
        """
        Generate sentiments.csv cache by classifying all headlines in raw_partner_headlines.csv.
        Uses direct model inference with GPU batching for efficiency.
        Automatically reduces batch size if OOM occurs.
        
        Args:
            force: If True, regenerate even if cache exists
            batch_size: Number of headlines to process in each GPU batch (default 64, reduced for low-VRAM GPUs)
            
        Returns:
            True if cache was generated, False otherwise
        """
        if os.path.exists(self.SENTIMENTS_CACHE) and (not force):
            logger.info(f'Sentiment cache already exists at {self.SENTIMENTS_CACHE}. Use force=True to regenerate.')
            return False
        if not os.path.exists(self.RAW_HEADLINES_FILE):
            logger.error(f'{self.RAW_HEADLINES_FILE} not found. Run download_and_extract_kaggle_data() first.')
            return False
        logger.info(f'Generating sentiment cache from {self.RAW_HEADLINES_FILE}')
        logger.info(f'Using direct model inference with batch_size={batch_size}')
        headlines_df = pd.read_csv(self.RAW_HEADLINES_FILE)
        total_headlines = len(headlines_df)
        logger.info(f'Loaded {total_headlines} headlines')
        sentiments = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_batch_size = batch_size
        batch_start = 0
        while batch_start < total_headlines:
            batch_end = min(batch_start + current_batch_size, total_headlines)
            batch_df = headlines_df.iloc[batch_start:batch_end]
            headlines_batch = batch_df['headline'].tolist()
            indices_batch = batch_df['index'].tolist()
            try:
                inputs = self.tokenizer(headlines_batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_labels = torch.argmax(logits, dim=-1)
                predicted_scores = torch.max(probabilities, dim=-1).values
                label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                for idx, headline_text, label_idx, score in zip(indices_batch, headlines_batch, predicted_labels.cpu().tolist(), predicted_scores.cpu().tolist()):
                    label = label_map.get(label_idx, 'neutral')
                    if label == 'positive':
                        final_score = score
                    elif label == 'negative':
                        final_score = -score
                    else:
                        final_score = score * 0.1
                    sentiments.append({'index': idx, 'sentiment': label, 'score': round(float(final_score), 4), 'confidence': round(float(score), 4)})
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                percent_done = 100 * batch_end / total_headlines
                logger.info(f'Processed {batch_end}/{total_headlines} headlines ({percent_done:.1f}%)')
                batch_start = batch_end
            except torch.cuda.OutOfMemoryError:
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(f'GPU OOM: reducing batch_size to {current_batch_size}, retrying...')
                    torch.cuda.empty_cache()
                else:
                    logger.error("Batch size is 1 but still OOM. This shouldn't happen.")
                    return False
        sentiments_df = pd.DataFrame(sentiments)
        sentiments_df.to_csv(self.SENTIMENTS_CACHE, index=False)
        logger.info(f'Sentiment cache saved to {self.SENTIMENTS_CACHE}')
        logger.info(f'Total: {len(sentiments_df)} sentiments classified')
        return True

    def loadSentimentCache(self):
        """Load cached sentiments from CSV."""
        if not os.path.exists(self.SENTIMENTS_CACHE):
            logger.warning(f'Sentiment cache not found at {self.SENTIMENTS_CACHE}')
            return None
        return pd.read_csv(self.SENTIMENTS_CACHE)

def download_and_extract_kaggle_data():
    """
    Download stock news dataset from Kaggle and extract raw headlines.
    
    Creates raw_partner_headlines.csv with columns:
    index, headline, URL, publisher, date, stock_ticker
    
    Returns:
        True if successful, False otherwise
    """
    output_file = NewsAnalyser.RAW_HEADLINES_FILE
    if os.path.exists(output_file):
        logger.info(f'{output_file} already exists. Skipping download.')
        return True
    try:
        logger.info('Downloading Kaggle stock news dataset...')
        dataset_path = kagglehub.dataset_download('miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests')
        logger.info(f'Dataset downloaded to: {dataset_path}')
        extracted_count = 0
        headlines_data = []
        for csv_file in Path(dataset_path).glob('**/*.csv'):
            logger.info(f'Processing {csv_file.name}...')
            try:
                df = pd.read_csv(csv_file)
                headline_cols = [col for col in df.columns if 'headline' in col.lower() or 'title' in col.lower() or 'news' in col.lower()]
                ticker_cols = [col for col in df.columns if 'ticker' in col.lower() or 'symbol' in col.lower() or 'stock' in col.lower()]
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                url_cols = [col for col in df.columns if 'url' in col.lower() or 'link' in col.lower()]
                if headline_cols and ticker_cols:
                    for idx, row in df.iterrows():
                        headline = str(row[headline_cols[0]]).strip() if headline_cols else ''
                        ticker = str(row[ticker_cols[0]]).strip() if ticker_cols else ''
                        date_str = str(row[date_cols[0]]).strip() if date_cols else ''
                        url = str(row[url_cols[0]]).strip() if url_cols else ''
                        publisher = str(row.get('publisher', '')).strip() if 'publisher' in row else ''
                        if publisher.lower() == 'benzinga':
                            continue
                        try:
                            date_obj = pd.to_datetime(date_str)
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except:
                            date_str = ''
                        if headline and ticker and date_str:
                            headlines_data.append({'index': extracted_count, 'headline': headline, 'URL': url, 'publisher': publisher, 'date': date_str, 'stock_ticker': ticker})
                            extracted_count += 1
            except Exception as e:
                logger.warning(f'Could not parse {csv_file.name}: {e}')
                continue
        if headlines_data:
            df_output = pd.DataFrame(headlines_data)
            df_output.to_csv(output_file, index=False)
            logger.info(f'Extracted {len(df_output)} headlines to {output_file}')
            return True
        else:
            logger.error('No headlines extracted from Kaggle dataset')
            return False
    except Exception as e:
        logger.error(f'Error downloading/extracting Kaggle data: {e}')
        logger.info('Ensure you have Kaggle API credentials at ~/.kaggle/kaggle.json')
        return False

def initialiseNewsDatabase(stockExchange, newsAnalyser, compute_sentiments=False):
    """
    Populate MySQL news_headlines table with extracted headlines.
    
    Args:
        stockExchange: StockExchange instance with MySQL connection
        newsAnalyser: NewsAnalyser instance for sentiment computation
        compute_sentiments: If True, compute and store sentiments. If False, store NULL (default).
                           Sentiments will be computed on-demand during runtime.
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(NewsAnalyser.RAW_HEADLINES_FILE):
        logger.error(f'{NewsAnalyser.RAW_HEADLINES_FILE} not found. Run download_and_extract_kaggle_data() first.')
        return False
    headlines_df = pd.read_csv(NewsAnalyser.RAW_HEADLINES_FILE)
    logger.info(f'Populating MySQL with {len(headlines_df)} raw headlines...')
    cursor = stockExchange.connection.cursor()
    insert_query = '\n        INSERT INTO news_headlines \n        (id, ticker, headline, url, publisher, date, sentiment, sentiment_score, confidence)\n        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n        ON DUPLICATE KEY UPDATE \n        sentiment = VALUES(sentiment), \n        sentiment_score = VALUES(sentiment_score),\n        confidence = VALUES(confidence)\n    '
    inserted = 0
    for idx, row in headlines_df.iterrows():
        try:
            if compute_sentiments:
                sentiment_result = newsAnalyser.getSentiment(row['headline'])
                sentiment = sentiment_result['sentiment']
                score = sentiment_result['score']
                confidence = sentiment_result['confidence']
            else:
                sentiment = None
                score = None
                confidence = None
            cursor.execute(insert_query, (int(row['index']), row['stock_ticker'], row['headline'], row['URL'], row['publisher'], row['date'], sentiment, score, confidence))
            inserted += 1
            if (idx + 1) % 5000 == 0:
                stockExchange.connection.commit()
                logger.info(f'Inserted {inserted}/{len(headlines_df)} records ({100 * inserted / len(headlines_df):.1f}%)')
        except Exception as e:
            logger.error(f'Error inserting record {idx}: {e}')
            continue
    stockExchange.connection.commit()
    cursor.close()
    logger.info(f'Successfully populated news_headlines with {inserted} records')
    if not compute_sentiments:
        logger.info('Sentiments are NULL - they will be computed on-demand when agent queries headlines')
    return True
if __name__ == '__main__':
    '\n    Main execution block: Download data, fine-tune model, generate cache, and populate database.\n    \n    Usage:\n        python3 stockNews.py              # Default: no fine-tuning, use pre-trained FinBERT\n        python3 stockNews.py --finetune   # Enable batched fine-tuning on 4.65M headlines\n    '
    parser = argparse.ArgumentParser(description='Stock news sentiment analysis pipeline', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python3 stockNews.py              # Use pre-trained FinBERT (fast, ~30-60 min)\n  python3 stockNews.py --finetune   # Fine-tune on data (slower, ~2-3 hours)\n        ')
    parser.add_argument('--finetune', action='store_true', help='Enable batched fine-tuning on 4.65M headlines (slower but potentially better accuracy)')
    args = parser.parse_args()
    logger.info('Starting stock news sentiment analysis pipeline...')
    if args.finetune:
        logger.info('Mode: FINE-TUNING ENABLED (batched processing, ~2-3 hours)')
    else:
        logger.info('Mode: PRE-TRAINED MODEL (fast, ~30-60 minutes)')
        logger.info('       Use --finetune flag to enable fine-tuning for comparison')
    logger.info('\n=== STEP 1: Download and Extract Kaggle Data ===')
    if not download_and_extract_kaggle_data():
        logger.error('Failed to download/extract Kaggle data. Exiting.')
        exit(1)
    logger.info('\n=== STEP 2: Fine-tune FinBERT ===')
    analyser = NewsAnalyser(batch_size=16, num_epochs=3)
    analyser.fineTune(enable_fine_tuning=args.finetune)
    logger.info('\n=== STEP 3: Skip Sentiment Pre-computation ===')
    logger.info('Sentiments will be computed on-demand during runtime')
    logger.info('This is faster for typical trading simulations that use <1% of headlines')
    logger.info('\n=== STEP 4: Populate MySQL with Raw Headlines ===')
    try:
        from stockExchange import StockExchange
        exchange = StockExchange()
        if initialiseNewsDatabase(exchange, analyser, compute_sentiments=False):
            logger.info('MySQL database populated successfully with raw headlines')
            logger.info('Sentiments will be computed on-demand when agent queries headlines')
        else:
            logger.error('Failed to populate MySQL database')
    except Exception as e:
        logger.warning(f'Could not initialise database: {e}')
        logger.info('You can call initialiseNewsDatabase(exchange, analyser) manually later')
    logger.info('\n=== Pipeline Complete ===')