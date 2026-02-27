import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer        # Added this
from src.components.model_trainer_gan import ModelTrainerGan
from src.components.model_trainer_llm import ModelTrainerLlm

if __name__ == "__main__":
    try:
        logging.info("Starting the Multi-Modal Training Pipeline")

        # 1. DATA INGESTION
        ingestion = DataIngestion()
        pricing_train_path, pricing_test_path, review_path = ingestion.initiate_data_ingestion()
        logging.info(f"Ingestion complete. Review path found at: {review_path}")

        # 2. DATA TRANSFORMATION (Tabular)
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            pricing_train_path, 
            pricing_test_path
        )

        # 3. PRICING MODEL TRAINING (Regression)
        # This uses the evaluate_models function from your utils.py
        logging.info("Initializing Pricing Model Trainer...")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Pricing Model Training complete. R2 Score: {r2_score}")

        # 4. GAN TRAINING (Fashion Generation)
        logging.info("Initializing GAN Trainer...")
        gan_trainer = ModelTrainerGan()
        gan_trainer.initiate_gan_training()

        # 5. LLM FINE-TUNING (Sentiment Analysis)
        logging.info("Initializing LLM Trainer...")
        llm_trainer = ModelTrainerLlm()
        llm_trainer.initiate_llm_training(review_path) 

        logging.info("--- ALL MODELS TRAINED SUCCESSFULLY ---")
        print("✅ ALL MODELS TRAINED SUCCESSFULLY")

    except Exception as e:
        raise CustomException(e, sys)