import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves any Python object (Model, Preprocessor, etc.) to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Successfully saved object at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a saved artifact back into memory.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found at {file_path}")
            
        with open(file_path, "rb") as file_obj:
            # ✅ Fix: Ensure we use pickle.load, NOT pickle.dump
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Tries different models and tunes their settings (GridSearch) 
    to find the most accurate one for Pricing Intelligence.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"Tuning and Evaluating model: {model_name}")

            # GridSearchCV finds the best 'settings' for the model
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Update the model with the best found settings
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Performance Scoring (R2 Score)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"Model: {model_name} | R2 Score: {test_model_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)