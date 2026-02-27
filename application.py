import os
import sys
import numpy as np
import torch
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, render_template

from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

application = Flask(__name__)
app = application

# --- ROUTES ---

@app.route('/')
def index():
    """Renders the Landing Page"""
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Handles Multi-Modal Inference for Pricing, Reviews, and GANs"""
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # 1. CAPTURE HOUSE PRICING DATA (Tabular)
            data = CustomData(
                OverallQual=int(request.form.get('OverallQual')),
                GrLivArea=int(request.form.get('GrLivArea')),
                GarageCars=int(request.form.get('GarageCars')),
                ExterQual=request.form.get('ExterQual'),
                KitchenQual=request.form.get('KitchenQual'),
                BsmtQual=request.form.get('BsmtQual'),
                GarageFinish=request.form.get('GarageFinish')
            )
            pred_df = data.get_data_as_data_frame()

            # 2. CAPTURE FUSION REVIEW (Text)
            review_text = request.form.get('review_text')

            # 3. RUN PREDICTION PIPELINE
            predict_pipeline = PredictPipeline()
            
            # Prediction A: House Price
            price_result = predict_pipeline.predict_price(pred_df)
            
            # Prediction B: Review Sentiment (Fine-tuned LLM)
            sentiment_result = predict_pipeline.predict_sentiment(review_text)

            # Prediction C: Fashion Design (GAN Generation)
            gen_image_tensor = predict_pipeline.generate_fashion_image()
            
            # --- IMAGE POST-PROCESSING ---
            # Convert GAN tensor to a viewable Base64 string for HTML
            img_array = gen_image_tensor.squeeze().detach().cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return render_template('home.html', 
                                   price_results=round(price_result[0], 2),
                                   sentiment_results=sentiment_result,
                                   fashion_image=img_str)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Ensure artifacts exist before running
    app.run(host="0.0.0.0", port=5000, debug=True)