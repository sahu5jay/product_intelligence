import os
import sys
import pandas as pd
from dataclasses import dataclass
import torchvision
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    pricing_raw_path: str = os.path.join('artifacts', "pricing_raw.csv")
    pricing_train_path: str = os.path.join('artifacts', "pricing_train.csv")
    pricing_test_path: str = os.path.join('artifacts', "pricing_test.csv")
    review_raw_path: str = os.path.join('artifacts', "reviews_raw.csv")
    image_dir: str = os.path.join('artifacts', "fashion_data")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("--- Data Ingestion Started ---")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.pricing_raw_path), exist_ok=True)

            # 1. Pricing Data
            df_pricing = pd.read_csv('notebook/data/train.csv')
            df_pricing.to_csv(self.ingestion_config.pricing_raw_path, index=False)
            train_set, test_set = train_test_split(df_pricing, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.pricing_train_path, index=False)
            test_set.to_csv(self.ingestion_config.pricing_test_path, index=False)

            # 2. Image Data (Fashion-MNIST)
            torchvision.datasets.FashionMNIST(root=self.ingestion_config.image_dir, train=True, download=True)

            # 3. Text Data (Reviews)
            local_jsonl_path = os.path.join('notebook', 'data', 'All_Beauty.jsonl')
            if os.path.exists(local_jsonl_path):
                df_reviews = pd.read_json(local_jsonl_path, lines=True)
                review_df = df_reviews.head(5000)[['text', 'rating']].copy()
                review_df.columns = ['review_body', 'stars']
                review_df.to_csv(self.ingestion_config.review_raw_path, index=False)
            
            logging.info("Ingestion Successful")
            return (
                self.ingestion_config.pricing_train_path,
                self.ingestion_config.pricing_test_path,
                self.ingestion_config.review_raw_path
            )

        except Exception as e:
            raise CustomException(e, sys)