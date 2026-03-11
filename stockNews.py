import kagglehub

# Download latest version of the stock news dataset.
# https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?resource=download
# Note the data is entirely historical and ranges from 2009-2020.
# It covers only United States companies.
# Rights: public domain.
path = kagglehub.dataset_download("miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")

print("Path to dataset files:", path)