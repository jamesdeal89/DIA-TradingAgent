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
    from stockNews import NewsAnalyzer, initializeNewsDatabase
    from stockExchange import StockExchange
    
    # Initialize
    analyzer = NewsAnalyzer()
    exchange = StockExchange()
    
    # Fine-tune FinBERT once (cached after first run)
    analyzer.fineTune()
    
    # Populate MySQL with sentiment data
    initializeNewsDatabase(exchange, analyzer)
    
    # Query headlines during simulation
    news_metrics = exchange.getNewsForStock('AAPL', 'XNAS', '2015-01-15')
"""

import os
import csv
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Third-party imports
import kagglehub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
import torch
from datasets import Dataset

# For compatibility with numpy versions
import warnings
warnings.filterwarnings('ignore')


class NewsAnalyzer:
    """
    Sentiment analysis engine using fine-tuned FinBERT model.
    
    Handles:
    - Building fine-tuned FinBERT from pre-trained weights
    - Classifying headline sentiment (positive/neutral/negative)
    - Caching results to sentiments.csv for deterministic simulation runs
    - Mapping headlines back to tickers and dates for query efficiency
    """
    
    MODEL_NAME = "ProsusAI/finBERT"
    FINE_TUNED_PATH = "./models/finbert_finetuned"
    SENTIMENTS_CACHE = "sentiments.csv"
    RAW_HEADLINES_FILE = "raw_partner_headlines.csv"
    
    # Device selection (GPU if available, else CPU)
    DEVICE = 0 if torch.cuda.is_available() else -1
    
    def __init__(self, batch_size: int = 16, num_epochs: int = 3):
        """
        Initialize NewsAnalyzer.
        
        Args:
            batch_size: Batch size for fine-tuning (reduce if memory-constrained)
            num_epochs: Number of fine-tuning epochs
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"NewsAnalyzer initialized. Using device: {self.device}")
        logger.info(f"Batch size: {batch_size}, Epochs: {num_epochs}")
        
        # Load or initialize model
        self._load_or_init_model()
    
    def _load_or_init_model(self):
        """Load fine-tuned model if exists, otherwise load pre-trained FinBERT."""
        if os.path.exists(self.FINE_TUNED_PATH):
            logger.info(f"Loading fine-tuned model from {self.FINE_TUNED_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.FINE_TUNED_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.FINE_TUNED_PATH)
        else:
            logger.info(f"Loading pre-trained FinBERT from {self.MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        
        # Set up inference pipeline
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.DEVICE
        )
    
    def getSentiment(self, headline: str) -> Dict[str, Any]:
        """
        Classify sentiment of a single headline.
        
        Args:
            headline: Text of the headline to analyze
            
        Returns:
            Dict with keys:
            - sentiment: 'positive', 'neutral', or 'negative'
            - score: float in range [-1.0, 1.0] (-1=negative, 0=neutral, 1=positive)
            - confidence: float in range [0.0, 1.0] (model confidence)
        """
        try:
            result = self.pipeline(headline, truncation=True, max_length=512)
            
            # FinBERT outputs: {"label": "positive|neutral|negative", "score": float}
            label = result[0]['label']
            model_score = float(result[0]['score'])
            
            # Normalize to [-1, 1] range
            if label == 'positive':
                score = model_score
            elif label == 'negative':
                score = -model_score
            else:  # neutral
                score = 0.0
            
            return {
                'sentiment': label,
                'score': score,
                'confidence': model_score
            }
        except Exception as e:
            logger.error(f"Error classifying headline '{headline[:50]}...': {e}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            }
    
    def fineTune(self, headlines_df: Optional[pd.DataFrame] = None, num_samples: Optional[int] = None):
        """
        Unsupervised fine-tune FinBERT on financial headlines from Kaggle dataset.
        
        If fine-tuned model already exists, skips fine-tuning and loads cached model.
        
        Args:
            headlines_df: DataFrame with 'headline' and 'label' columns. If None, loads from raw_partner_headlines.csv
            num_samples: Max number of samples to fine-tune on (useful for testing). If None, uses all.
        """
        # Check if already fine-tuned
        if os.path.exists(self.FINE_TUNED_PATH):
            logger.info(f"Fine-tuned model already exists at {self.FINE_TUNED_PATH}. Skipping fine-tuning.")
            return
        
        # Load data if not provided
        if headlines_df is None:
            if not os.path.exists(self.RAW_HEADLINES_FILE):
                logger.error(f"{self.RAW_HEADLINES_FILE} not found. Run download_and_extract_kaggle_data() first.")
                return
            
            logger.info(f"Loading headlines from {self.RAW_HEADLINES_FILE}")
            headlines_df = pd.read_csv(self.RAW_HEADLINES_FILE)
        
        # Limit samples if specified (for testing)
        if num_samples is not None:
            headlines_df = headlines_df.head(num_samples)
        
        logger.info(f"Fine-tuning on {len(headlines_df)} headlines...")
        
        # Prepare data for fine-tuning
        # Here we use unsupervised fine-tuning (MLM) on the headlines themselves.
        try:
            # Convert to HuggingFace Dataset
            dataset = Dataset.from_pandas(headlines_df[['headline']].reset_index(drop=True))
            
            # Tokenize
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['headline'],
                    padding='max_length',
                    truncation=True,
                    max_length=512
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.FINE_TUNED_PATH,
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                save_steps=100,
                save_total_limit=2,
                logging_steps=50,
            )
            
            # Train
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )
            
            logger.info(f"Starting fine-tuning for {self.num_epochs} epochs with batch size {self.batch_size}")
            trainer.train()
            
            # Save fine-tuned model
            os.makedirs(self.FINE_TUNED_PATH, exist_ok=True)
            self.model.save_pretrained(self.FINE_TUNED_PATH)
            self.tokenizer.save_pretrained(self.FINE_TUNED_PATH)
            logger.info(f"Fine-tuned model saved to {self.FINE_TUNED_PATH}")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            logger.info("Continuing with pre-trained model...")
    
    def generateSentimentCache(self, force: bool = False) -> bool:
        """
        Generate sentiments.csv cache by classifying all headlines in raw_partner_headlines.csv.
        
        Args:
            force: If True, regenerate even if cache exists
            
        Returns:
            True if cache was generated, False otherwise
        """
        # Check if cache already exists
        if os.path.exists(self.SENTIMENTS_CACHE) and not force:
            logger.info(f"Sentiment cache already exists at {self.SENTIMENTS_CACHE}. Use force=True to regenerate.")
            return False
        
        # Check if raw headlines file exists
        if not os.path.exists(self.RAW_HEADLINES_FILE):
            logger.error(f"{self.RAW_HEADLINES_FILE} not found. Run download_and_extract_kaggle_data() first.")
            return False
        
        logger.info(f"Generating sentiment cache from {self.RAW_HEADLINES_FILE}")
        
        # Load raw headlines
        headlines_df = pd.read_csv(self.RAW_HEADLINES_FILE)
        logger.info(f"Loaded {len(headlines_df)} headlines")
        
        # Classify each headline
        sentiments = []
        for idx, row in headlines_df.iterrows():
            headline = row['headline']
            sentiment_result = self.getSentiment(headline)
            
            sentiments.append({
                'index': row['index'],
                'sentiment': sentiment_result['sentiment'],
                'score': round(sentiment_result['score'], 4),
                'confidence': round(sentiment_result['confidence'], 4)
            })
            
            # Progress logging
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(headlines_df)} headlines")
        
        # Write cache
        sentiments_df = pd.DataFrame(sentiments)
        sentiments_df.to_csv(self.SENTIMENTS_CACHE, index=False)
        logger.info(f"Sentiment cache saved to {self.SENTIMENTS_CACHE}")
        
        return True
    
    def loadSentimentCache(self) -> Optional[pd.DataFrame]:
        """Load cached sentiments from CSV."""
        if not os.path.exists(self.SENTIMENTS_CACHE):
            logger.warning(f"Sentiment cache not found at {self.SENTIMENTS_CACHE}")
            return None
        
        return pd.read_csv(self.SENTIMENTS_CACHE)


def download_and_extract_kaggle_data() -> bool:
    """
    Download stock news dataset from Kaggle and extract raw headlines.
    
    Creates raw_partner_headlines.csv with columns:
    index, headline, URL, publisher, date, stock_ticker
    
    Returns:
        True if successful, False otherwise
    """
    output_file = NewsAnalyzer.RAW_HEADLINES_FILE
    
    # Check if already extracted
    if os.path.exists(output_file):
        logger.info(f"{output_file} already exists. Skipping download.")
        return True
    
    try:
        logger.info("Downloading Kaggle stock news dataset...")
        dataset_path = kagglehub.dataset_download(
            "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
        )
        logger.info(f"Dataset downloaded to: {dataset_path}")
        
        # Parse the dataset
        # The Kaggle dataset contains multiple CSV files. We'll parse them to extract headlines.
        extracted_count = 0
        headlines_data = []
        
        # Search for CSV files in the dataset
        for csv_file in Path(dataset_path).glob("**/*.csv"):
            logger.info(f"Processing {csv_file.name}...")
            
            try:
                df = pd.read_csv(csv_file)
                
                # Look for common headline/news columns
                headline_cols = [col for col in df.columns if 'headline' in col.lower() or 'title' in col.lower() or 'news' in col.lower()]
                ticker_cols = [col for col in df.columns if 'ticker' in col.lower() or 'symbol' in col.lower()]
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                url_cols = [col for col in df.columns if 'url' in col.lower() or 'link' in col.lower()]
                
                if headline_cols and ticker_cols:
                    for idx, row in df.iterrows():
                        headline = str(row[headline_cols[0]]).strip() if headline_cols else ""
                        ticker = str(row[ticker_cols[0]]).strip() if ticker_cols else ""
                        date_str = str(row[date_cols[0]]).strip() if date_cols else ""
                        url = str(row[url_cols[0]]).strip() if url_cols else ""
                        publisher = str(row.get('publisher', '')).strip() if 'publisher' in row else ""
                        
                        # Skip Benzinga entries
                        if publisher.lower() == 'benzinga':
                            continue
                        
                        # Normalize date to YYYY-MM-DD
                        try:
                            date_obj = pd.to_datetime(date_str)
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except:
                            date_str = ""
                        
                        if headline and ticker and date_str:
                            headlines_data.append({
                                'index': extracted_count,
                                'headline': headline,
                                'URL': url,
                                'publisher': publisher,
                                'date': date_str,
                                'stock_ticker': ticker
                            })
                            extracted_count += 1
            
            except Exception as e:
                logger.warning(f"Could not parse {csv_file.name}: {e}")
                continue
        
        if headlines_data:
            # Save to CSV
            df_output = pd.DataFrame(headlines_data)
            df_output.to_csv(output_file, index=False)
            logger.info(f"Extracted {len(df_output)} headlines to {output_file}")
            return True
        else:
            logger.error("No headlines extracted from Kaggle dataset")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading/extracting Kaggle data: {e}")
        logger.info("Ensure you have Kaggle API credentials at ~/.kaggle/kaggle.json")
        return False


def initializeNewsDatabase(stockExchange, news_analyzer: NewsAnalyzer) -> bool:
    """
    Populate MySQL news_headlines table with extracted headlines and sentiment scores.
    
    Args:
        stockExchange: StockExchange instance with MySQL connection
        news_analyzer: NewsAnalyzer instance with sentiment cache
        
    Returns:
        True if successful, False otherwise
    """
    # Check if raw headlines and sentiments exist
    if not os.path.exists(NewsAnalyzer.RAW_HEADLINES_FILE):
        logger.error(f"{NewsAnalyzer.RAW_HEADLINES_FILE} not found. Run download_and_extract_kaggle_data() first.")
        return False
    
    if not os.path.exists(NewsAnalyzer.SENTIMENTS_CACHE):
        logger.error(f"{NewsAnalyzer.SENTIMENTS_CACHE} not found. Run news_analyzer.generateSentimentCache() first.")
        return False
    
    # Load data
    headlines_df = pd.read_csv(NewsAnalyzer.RAW_HEADLINES_FILE)
    sentiments_df = pd.read_csv(NewsAnalyzer.SENTIMENTS_CACHE)
    
    # Merge on index
    merged_df = headlines_df.merge(sentiments_df, on='index', how='inner')
    
    if len(merged_df) == 0:
        logger.error("No matching headlines and sentiments found")
        return False
    
    logger.info(f"Populating MySQL with {len(merged_df)} headlines...")
    
    # Insert into database
    cursor = stockExchange.connection.cursor()
    
    insert_query = """
        INSERT INTO news_headlines 
        (ticker, headline, url, publisher, date, sentiment, sentiment_score, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
        sentiment = VALUES(sentiment), 
        sentiment_score = VALUES(sentiment_score),
        confidence = VALUES(confidence)
    """
    
    inserted = 0
    for idx, row in merged_df.iterrows():
        try:
            cursor.execute(insert_query, (
                row['stock_ticker'],
                row['headline'],
                row['URL'],
                row['publisher'],
                row['date'],
                row['sentiment'],
                float(row['score']),
                float(row['confidence'])
            ))
            inserted += 1
            
            if (idx + 1) % 500 == 0:
                stockExchange.connection.commit()
                logger.info(f"Inserted {inserted}/{len(merged_df)} records")
        
        except Exception as e:
            logger.error(f"Error inserting record {idx}: {e}")
            continue
    
    # Final commit
    stockExchange.connection.commit()
    cursor.close()
    logger.info(f"Successfully populated news_headlines with {inserted} records")
    
    return True


if __name__ == "__main__":
    """
    Main execution block: Download data, fine-tune model, generate cache, and populate database.
    """
    logger.info("Starting stock news sentiment analysis pipeline...")
    
    # Step 1: Download and extract Kaggle data
    logger.info("\n=== STEP 1: Download and Extract Kaggle Data ===")
    if not download_and_extract_kaggle_data():
        logger.error("Failed to download/extract Kaggle data. Exiting.")
        exit(1)
    
    # Step 2: Initialize NewsAnalyzer and fine-tune (or load cached)
    logger.info("\n=== STEP 2: Fine-tune FinBERT ===")
    analyzer = NewsAnalyzer(batch_size=16, num_epochs=3)
    analyzer.fineTune()
    
    # Step 3: Generate sentiment cache
    logger.info("\n=== STEP 3: Generate Sentiment Cache ===")
    if analyzer.generateSentimentCache():
        logger.info("Sentiment cache generated successfully")
    else:
        logger.info("Sentiment cache already exists or error occurred")
    
    # Step 4: Populate MySQL (requires StockExchange instance)
    logger.info("\n=== STEP 4: Populate MySQL news_headlines Table ===")
    try:
        from stockExchange import StockExchange
        exchange = StockExchange()
        if initializeNewsDatabase(exchange, analyzer):
            logger.info("MySQL database populated successfully")
        else:
            logger.error("Failed to populate MySQL database")
    except Exception as e:
        logger.warning(f"Could not initialize database: {e}")
        logger.info("You can call initializeNewsDatabase(exchange, analyzer) manually later")
    
    logger.info("\n=== Pipeline Complete ===")