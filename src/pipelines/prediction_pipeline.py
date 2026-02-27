import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Use abspath to ensure the library treats these as local system paths
        self.preprocessor_path = os.path.abspath(os.path.join("artifacts", "preprocessor.pkl"))
        self.model_path = os.path.abspath(os.path.join("artifacts", "model.pkl"))
        self.gan_path = os.path.abspath(os.path.join("artifacts", "generator.pth"))
        self.llm_path = os.path.abspath(os.path.join("artifacts", "sentiment_model"))

    def predict_price(self, features_df):
        """Regression: Predicts House Price"""
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Transform input using the saved preprocessor
            data_scaled = preprocessor.transform(features_df)
            preds = model.predict(data_scaled)
            
            # Use expm1 if you used log-transformation during training
            return np.expm1(preds) 
        except Exception as e:
            raise CustomException(e, sys)

    def predict_sentiment(self, text):
        """Predicts Sentiment using Fine-tuned DistilBERT"""
        try:
            # Normalize path for Windows compatibility with Transformers lib
            clean_path = self.llm_path.replace("\\", "/")
            
            tokenizer = AutoTokenizer.from_pretrained(clean_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(clean_path, local_files_only=True)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            prediction = torch.argmax(outputs.logits, dim=1).item()
            mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            return mapping[prediction]
            
        except Exception as e:
            raise CustomException(e, sys)

            
    def generate_fashion_image(self):
        """Generative AI: Creates a new design using the Fashion GAN"""
        try:
            from src.components.model_trainer_gan import Generator
            
            # Initialize model architecture
            model = Generator(latent_dim=100)
            model.load_state_dict(torch.load(self.gan_path, map_location=torch.device('cpu')))
            model.eval()

            with torch.no_grad():
                noise = torch.randn(1, 100)
                generated_img = model(noise)
            
            return generated_img # Returns tensor for application.py to process
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Utility class to map web form inputs to a pandas DataFrame.
    Matches the schema expected by the Pricing Model.
    """
    def __init__(self, OverallQual, GrLivArea, GarageCars, ExterQual, KitchenQual, BsmtQual, GarageFinish):
        self.OverallQual = OverallQual
        self.GrLivArea = GrLivArea
        self.GarageCars = GarageCars
        self.ExterQual = ExterQual
        self.KitchenQual = KitchenQual
        self.BsmtQual = BsmtQual
        self.GarageFinish = GarageFinish

    def get_data_as_data_frame(self):
        try:
            input_dict = {
                "OverallQual": [self.OverallQual],
                "GrLivArea": [self.GrLivArea],
                "GarageCars": [self.GarageCars],
                "ExterQual": [self.ExterQual],
                "KitchenQual": [self.KitchenQual],
                "BsmtQual": [self.BsmtQual],
                "GarageFinish": [self.GarageFinish],
            }
            return pd.DataFrame(input_dict)
        except Exception as e:
            raise CustomException(e, sys)