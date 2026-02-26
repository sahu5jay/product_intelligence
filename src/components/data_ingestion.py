import os
import sys
import pandas as pd
from dataclasses import dataclass

# Core libraries
import torchvision
from sklearn.model_selection import train_test_split

# Local modular utilities
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    """Artifact path configurations for all three components."""
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
            # Create artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.pricing_raw_path), exist_ok=True)

            # 1. PRICING DATA (Structured Model)
            logging.info("Ingesting Pricing Intelligence data from local CSV...")
            df_pricing = pd.read_csv('notebook/data/train.csv')
            df_pricing.to_csv(self.ingestion_config.pricing_raw_path, index=False)
            
            # Creating Train-Test split for the Regression model
            train_set, test_set = train_test_split(df_pricing, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.pricing_train_path, index=False)
            test_set.to_csv(self.ingestion_config.pricing_test_path, index=False)

            # 2. IMAGE DATA (GAN - Fashion Intelligence)
            logging.info("Sourcing Fashion-MNIST via torchvision...")
            torchvision.datasets.FashionMNIST(
                root=self.ingestion_config.image_dir, 
                train=True, 
                download=True
            )

            # 3. TEXT DATA (LLM - Review Intelligence)
            logging.info("Processing manually downloaded All_Beauty.jsonl...")
            local_jsonl_path = os.path.join('notebook', 'data', 'All_Beauty.jsonl')
            
            if os.path.exists(local_jsonl_path):
                # Read JSONL (Lines=True is essential for this format)
                df_reviews = pd.read_json(local_jsonl_path, lines=True)
                
                # Selecting relevant columns: 'text' (body) and 'rating' (stars)
                # Sampling 5000 rows to keep the prototype training efficient
                review_df = df_reviews.head(5000)[['text', 'rating']].copy()
                review_df.columns = ['review_body', 'stars']
                review_df['product_category'] = 'All_Beauty'
                
                review_df.to_csv(self.ingestion_config.review_raw_path, index=False)
                logging.info(f"Review artifacts saved to {self.ingestion_config.review_raw_path}")
            else:
                logging.error(f"Review file not found at {local_jsonl_path}")
                raise FileNotFoundError("Missing All_Beauty.jsonl in notebook/data/")

            logging.info("--- Data Ingestion Successful: All Components Ready ---")
            
            return (
                self.ingestion_config.pricing_train_path,
                self.ingestion_config.review_raw_path,
                self.ingestion_config.image_dir
            )

        except Exception as e:
            logging.error("Data Ingestion failed.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()