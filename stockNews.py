"""
Stock News Sentiment Analysis Initialisation.

Downloads the Kaggle stock news dataset, extracts raw headlines, and fine-tunes FinBERT
for financial sentiment classification.

Architecture:
1. Download Kaggle dataset via kagglehub
2. Extract raw_partner_headlines.csv (6 columns: index, headline, URL, publisher, date, stock_ticker)
3. Fine-tune ProsusAI/finBERT on the dataset headlines
4. Populate MySQL news_headlines table in StockExchange
"""
import os
import sys
import pandas as pd
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import kagglehub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
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
    - Mapping headlines back to tickers and dates for query efficiency
    """
    MODEL_NAME = 'ProsusAI/finBERT'
    FINE_TUNED_PATH = './models/finbert_finetuned'
    RAW_HEADLINES_FILE = 'raw_partner_headlines.csv'
    DEVICE = 0 if torch.cuda.is_available() else -1

    def __init__(self, batchSize=16, numEpochs=3):
        """
        Initialise NewsAnalyser.
        
        batchSize: Batch size for fine-tuning to mitigate memory full issues.
        numEpochs: number of epochs for fine-tuning. 
        """
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'NewsAnalyser initialised. Using device: {self.device}')
        logger.info(f'Batch size: {batchSize}, Epochs: {numEpochs}')
        self.loadOrInitModel()

    def loadOrInitModel(self):
        """Load model and tokenizer."""
        modelPath = self.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelPath)

    def tokenizeFunction(self, examples):
        return self.tokenizer(examples['headline'], padding='max_length', truncation=True, max_length=512)

    def fineTune(self, headlinesDf=None, numSamples=None, batchChunkSize=50000, enableFineTuning=False):
        """
        Batch-based fine-tuning of FinBERT on financial headlines.
        
        Processes headlines in chunks to fit within memory.
        Uses gradient accumulation and smaller per-device batch sizes for efficiency.
        
        headlinesDf: Optional DataFrame with headlines. If None, loads from RAW_HEADLINES_FILE
        numSamples: Limit to first N samples (optional)
        batchChunkSize: Number of headlines to process per batch.
        enableFineTuning: Set to True to actually fine-tune. Default False (use pre-trained)
        
        Returns true if fine-tuning completed, False if skipped
        """
        if not enableFineTuning:
            logger.info('Fine-tuning DISABLED by default: Using pre-trained FinBERT (already trained on financial data)')
            logger.info('To enable batched fine-tuning, call: analyser.fineTune(enableFineTuning=True)')
            logger.info('This will process headlines in 50k batches with gradient accumulation')
            return False
        if headlinesDf is None:
            if not os.path.exists(self.RAW_HEADLINES_FILE):
                logger.error(f'{self.RAW_HEADLINES_FILE} not found')
                return False
            logger.info(f'Loading headlines from {self.RAW_HEADLINES_FILE}')
            headlinesDf = pd.read_csv(self.RAW_HEADLINES_FILE)
        if numSamples is not None:
            headlinesDf = headlinesDf.head(numSamples)
        totalHeadlines = len(headlinesDf)
        logger.info(f'Fine-tuning on {totalHeadlines} headlines in batches of {batchChunkSize}')
        try:
            for chunkIdx, startIdx in enumerate(range(0, totalHeadlines, batchChunkSize)):
                endIdx = min(startIdx + batchChunkSize, totalHeadlines)
                chunkDf = headlinesDf.iloc[startIdx:endIdx]
                chunkSize = len(chunkDf)
                logger.info(f'\n[Batch {chunkIdx + 1}] Processing headlines {startIdx}-{endIdx} ({chunkSize} headlines)')
                dataset = Dataset.from_pandas(chunkDf[['headline']].reset_index(drop=True))
                logger.info(f'Tokenizing batch {chunkIdx + 1}...')
                tokenizedDataset = dataset.map(self.tokenizeFunction, batched=True, batch_size=1000)
                trainingArgs = TrainingArguments(output_dir=f'{self.FINE_TUNED_PATH}/checkpoint-batch-{chunkIdx}', num_train_epochs=1, per_device_train_batch_size=8, gradient_accumulation_steps=8, save_steps=500, save_total_limit=1, logging_steps=100, log_level='warning', use_cpu=False if torch.cuda.is_available() else True)
                logger.info(f'Training on batch {chunkIdx + 1}...')
                trainer = Trainer(model=self.model, args=trainingArgs, train_dataset=tokenizedDataset)
                trainer.train()
                logger.info(f'Batch {chunkIdx + 1} complete')
            os.makedirs(self.FINE_TUNED_PATH, exist_ok=True)
            logger.info(f'\nSaving fine-tuned model to {self.FINE_TUNED_PATH}')
            self.model.save_pretrained(self.FINE_TUNED_PATH)
            self.tokenizer.save_pretrained(self.FINE_TUNED_PATH)
            logger.info('Fine-tuning completed successfully!')
            return True
        except Exception as e:
            logger.error(f'Error during fine-tuning: {e}')
            logger.info('Continuing with pre-trained model...')
            return False

def downloadAndExtractKaggleData():
    """
    Download stock news dataset from Kaggle and extract raw headlines.
    
    Creates raw_partner_headlines.csv with columns:
    index, headline, URL, publisher, date, stock_ticker
    
    Returns true if successful, False otherwise
    """
    outputFile = NewsAnalyser.RAW_HEADLINES_FILE
    if os.path.exists(outputFile):
        logger.info(f'{outputFile} already exists. Skipping download.')
        return True
    try:
        logger.info('Downloading Kaggle stock news dataset...')
        datasetPath = kagglehub.dataset_download('miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests')
        logger.info(f'Dataset downloaded to: {datasetPath}')
        extractedCount = 0
        headlinesData = []
        for csvFile in Path(datasetPath).glob('**/*.csv'):
            logger.info(f'Processing {csvFile.name}...')
            try:
                df = pd.read_csv(csvFile)
                # Builds lists of candidate columns by scanning for fields which match the right category.
                # As the dataset has multiple CSVs this collates all the relevant data into one large dataframe.
                headlineCols = [col for col in df.columns if 'headline' in col.lower() or 'title' in col.lower() or 'news' in col.lower()]
                tickerCols = [col for col in df.columns if 'ticker' in col.lower() or 'symbol' in col.lower() or 'stock' in col.lower()]
                dateCols = [col for col in df.columns if 'date' in col.lower()]
                urlCols = [col for col in df.columns if 'url' in col.lower() or 'link' in col.lower()]
                if headlineCols and tickerCols:
                    for idx, row in df.iterrows():
                        headline = str(row[headlineCols[0]]).strip() if headlineCols else ''
                        ticker = str(row[tickerCols[0]]).strip() if tickerCols else ''
                        dateStr = str(row[dateCols[0]]).strip() if dateCols else ''
                        url = str(row[urlCols[0]]).strip() if urlCols else ''
                        publisher = str(row.get('publisher', '')).strip() if 'publisher' in row else ''
                        if publisher.lower() == 'benzinga':
                            continue
                        try:
                            date_obj = pd.to_datetime(dateStr)
                            dateStr = date_obj.strftime('%Y-%m-%d')
                        except:
                            dateStr = ''
                        if headline and ticker and dateStr:
                            headlinesData.append({'index': extractedCount, 'headline': headline, 'URL': url, 'publisher': publisher, 'date': dateStr, 'stock_ticker': ticker})
                            extractedCount += 1
            except Exception as e:
                logger.warning(f'Could not parse {csvFile.name}: {e}')
                continue
        if headlinesData:
            dfOutput = pd.DataFrame(headlinesData)
            dfOutput.to_csv(outputFile, index=False)
            logger.info(f'Extracted {len(dfOutput)} headlines to {outputFile}')
            return True
        else:
            logger.error('No headlines extracted from Kaggle dataset')
            return False
    except Exception as e:
        logger.error(f'Error downloading/extracting Kaggle data: {e}')
        logger.info('Ensure you have Kaggle API credentials at ~/.kaggle/kaggle.json')
        return False

def initialiseNewsDatabase(stockExchange):
    """
    Populate MySQL news_headlines table with extracted headlines.
    Important for caching.
    
    stockExchange: StockExchange instance with MySQL connection
        
    Returns True if successful, False otherwise
    """
    if not os.path.exists(NewsAnalyser.RAW_HEADLINES_FILE):
        logger.error(f'{NewsAnalyser.RAW_HEADLINES_FILE} not found. Run downloadAndExtractKaggleData() first.')
        return False
    headlinesDf = pd.read_csv(NewsAnalyser.RAW_HEADLINES_FILE)
    logger.info(f'Populating MySQL with {len(headlinesDf)} raw headlines...')
    cursor = stockExchange.connection.cursor()
    insertQuery = '''
        INSERT INTO news_headlines
        (id, ticker, headline, url, publisher, date, sentiment, sentiment_score, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        sentiment = VALUES(sentiment),
        sentiment_score = VALUES(sentiment_score),
        confidence = VALUES(confidence)
    '''
    inserted = 0
    for idx, row in headlinesDf.iterrows():
        try:
            sentiment = None
            score = None
            confidence = None
            cursor.execute(insertQuery, (int(row['index']), row['stock_ticker'], row['headline'], row['URL'], row['publisher'], row['date'], sentiment, score, confidence))
            inserted += 1
            if (idx + 1) % 5000 == 0:
                stockExchange.connection.commit()
                logger.info(f'Inserted {inserted}/{len(headlinesDf)} records ({100 * inserted / len(headlinesDf):.1f}%)')
        except Exception as e:
            logger.error(f'Error inserting record {idx}: {e}')
            continue
    stockExchange.connection.commit()
    cursor.close()
    logger.info(f'Successfully populated news_headlines with {inserted} records')
    return True

if __name__ == '__main__':
    validArgs = {'--finetune', '--help', '-h'}
    providedArgs = sys.argv[1:]
    unknownArgs = [arg for arg in providedArgs if arg not in validArgs]
    if '--help' in providedArgs or '-h' in providedArgs:
        print('Usage is: python3 stockNews.py [--finetune]')
        print('  --finetune   Enable fine-tuning on historical headlines')
        sys.exit(0)
    if unknownArgs:
        logger.error(f'Unknown arguments: {unknownArgs}')
        print('Usage: python3 stockNews.py [--finetune]')
        sys.exit(2)
    enableFineTuning = '--finetune' in providedArgs
    logger.info('Starting stock news sentiment analysis pipeline...')
    if enableFineTuning:
        logger.info('Mode: FINE-TUNING ENABLED')
    else:
        logger.info('Mode: PRE-TRAINED MODEL')
        logger.info('Note: Use --finetune arg to enable fine-tuning.')
    logger.info('Download and Extract Kaggle Data')
    if not downloadAndExtractKaggleData():
        logger.error('Failed to download/extract Kaggle data.')
        exit(1)
    analyser = NewsAnalyser(batchSize=16, numEpochs=3)
    analyser.fineTune(enableFineTuning=enableFineTuning)
    logger.info('Populating MySQL with Raw Headlines...')
    try:
        from stockExchange import StockExchange
        exchange = StockExchange()
        if initialiseNewsDatabase(exchange):
            logger.info('MySQL database populated successfully with raw headlines')
        else:
            logger.error('Failed to populate MySQL database')
    except Exception as e:
        logger.warning(f'Could not initialise database: {e}')
    logger.info('\n Complete!')