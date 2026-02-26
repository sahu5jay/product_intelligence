Product Intelligence Platform

AI-Powered Synthetic Media & Customer Insight System

1> Cold-start Visuals: Generating synthetic product imagery using Generative Adversarial Networks (GANs) to augment catalogs where real data is scarce.

2> Scalable Insights: Fine-tuning Large Language Models (LLMs) to automate the analysis of millions of unstructured customer reviews, providing instant sentiment and summarization.

Project Architecture
PRODUCT_INTELLIGENCE/
 artifacts/               # Trained models, checkpoints, and synthetic samples
 logs/                    # Execution and training logs
 notebook/                # Exploratory Data Analysis (EDA) and Prototyping
 src/                     # Production-grade source code
    components/          # Core modules (Ingestion, GAN Trainer, LLM Trainer)
    pipelines/           # Orchestration (Training & Prediction)
    logger.py            # Standardized logging
    exception.py         # Custom error handling
    utils.py             # Common helper functions
 application.py           # API layer (FastAPI) for model inference
 requirements.txt         # Dependency management
 setup.py

Environment Configuration
The system is optimized for Python 3.11 within an Anaconda environment to ensure compatibility between PyTorch (GANs) and HuggingFace (LLMs).
# Create and activate environment
conda create -n product_intel python=3.11 -y
conda activate product_intel

# Install dependencies
pip install -r requirements.txt

Packaging (setup.py)
The setup.py file allows the src directory to be treated as a local package. This enables clean imports across the project


Structured Dataset (Contextual Understanding)
The House Prices Dataset "Pricing Intelligence": > "I'm using the House Prices dataset as a proxy for complex product valuation. Just as a retailer evaluates a product's value based on its attributes (material, brand, age), we are analyzing how house 'features' (quality, location, size) dictate market price. This forms the Contextual Understanding layer of our platform."


Image Dataset (GAN Training)
Fashion MNIST "Virtual Design Lab": > "New fashion lines often lack photography. By training a GAN on Fashion MNIST, we are creating a 'Synthetic Design Lab.' The AI learns the 'grammar' of clothing-what a shirt or a boot looks like-allowing the marketing team to generate infinite design variations for testing before a single physical sample is even sewn."


Text Dataset (LLM Fine-Tuning)
Amazon Reviews  "Voice of the Customer":
"To solve the scale problem of millions of reviews, we are fine-tuning an LLM to act as an automated 'Sentiment Analyst.' It doesn't just read words; it understands the specific language of retail-like the difference between 'heavy' being a positive for a winter coat but a negative for a running shoe."

Data Ingestion Component
The Data Ingestion stage is the "entry point" of our project. Its main job is to gather data from three different places and organize them into a single folder called artifacts so the rest of the pipeline can use them easily.

Handling the Pricing Data (Numbers)

    What it does: It reads the file and immediately splits it into two parts: a Training set (80%) and a Testing set (20%).
    Why: This ensures that when we build our price-prediction model later, we have a "secret" set of data to test if the model actually learned or just memorized the answers.

Handling the Fashion Data (Images)
For the Fashion Intelligence (GAN) part of the assignment, we need images.

    What it does: The script uses a library called torchvision to automatically download the Fashion-MNIST dataset.
    Result: It saves these images directly into artifacts/fashion_data/. This is much more reliable than trying to find and upload thousands of individual image files manually.


Handling the Review Data (Text)
    The Problem: - originally tried to stream the Amazon reviews directly from the internet, but the "Hugging Face" library blocked the request for security reasons

    The Solution - We pivoted to a Local Manual Ingestion strategy. You manually downloaded the All_Beauty.jsonl file, and the script now reads it directly from your computer.