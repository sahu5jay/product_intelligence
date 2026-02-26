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

Handling the (Numbers)

    What it does: It reads the file and immediately splits it into two parts: a Training set (80%) and a Testing set (20%).
    Why: This ensures that when we build our price-prediction model later, we have a "secret" set of data to test if the model actually learned or just memorized the answers.

Handling the Fashion Data (Images)
For the Fashion Intelligence (GAN) part of the assignment, we need images.

    What it does: The script uses a library called torchvision to automatically download the Fashion-MNIST dataset.
    Result: It saves these images directly into artifacts/fashion_data/. This is much more reliable than trying to find and upload thousands of individual image files manually.


Handling the Review Data (Text)
    The Problem: - originally tried to stream the Amazon reviews directly from the internet, but the "Hugging Face" library blocked the request for security reasons

    The Solution - We pivoted to a Local Manual Ingestion strategy. You manually downloaded the All_Beauty.jsonl file, and the script now reads it directly from your computer.

Summary of the datasets

House Prices Dataset
--- Pricing Statistics ---
count      1168.000000
mean     181441.541952
std       77263.583862
min       34900.000000
25%      130000.000000
50%      165000.000000
75%      214925.000000
max      745000.000000
Name: SalePrice, dtype: float64
Skewness: 1.74 (High skew > 1 requires Log Transform)
Kurtosis: 5.48 (High kurtosis indicates heavy outliers)
Number of statistical outliers: 48

Fashion MNIST "Virtual Design Lab"
--- Fashion Image Stats ---
Dataset Mean: 0.2829
Dataset Std Dev: 0.3531
Class Distribution: Counter({np.int64(9): 6000, np.int64(0): 6000, np.int64(3): 6000, np.int64(2): 6000, np.int64(7): 6000, np.int64(5): 6000, np.int64(1): 6000, np.int64(6): 6000, np.int64(4): 6000, np.int64(8): 6000})

Amazon Reviews  "Voice of the Customer"
--- Review Sentiment Distribution ---
stars
5    57.28
4    20.00
3    10.00
1     7.20
2     5.52
Name: proportion, dtype: float64

--- Review Length Stats ---
count    5000.000000
mean       71.428600
std        88.217083
min         0.000000
25%        17.000000
50%        45.000000
75%        92.000000
max      1511.000000
Name: word_count, dtype: float64
Missing Review Texts: 0

Summary of House Price Data
Target Fix: SalePrice is right-skewed; apply Log Transformation.

Missing Data: PoolQC, MiscFeature, and Alley are >90% empty; treat "NaN" as a specific category called "None".

Predictive Power: Neighborhood, OverallQual, and GrLivArea are your strongest features.

Redundancy: Street and Utilities should be dropped as they provide no statistical variance.


---------------------------------
Encoding for categorical Data of Pricing Data Summary
---------------------------------

Feature     	        Encoding Type	                    Why
Neighborhood	    Target Mean Encoding	         High cardinality + price influence
ExterQual	        Ordinal Encoding	             Natural quality order
KitchenQual	        Ordinal Encoding	             Quality scale
BsmtQual	        Ordinal Encoding	             Quality scale
Foundation	        Domain-based Ordinal	         Structural durability ranking


Skewness Rule
Skewness	Meaning
0	Perfect normal
0 – 0.5	Approximately normal
0.5 – 1	Moderately skewed
> 1	Highly skewed


GrLivArea      1.366560
TotalBsmtSF    1.524255
SalePrice      0.121347

GrLivArea → Needs transformation
TotalBsmtSF → Needs transformation
SalePrice → Already good

Review Intelligence Dataset Summary
The system utilizes a subset of the Amazon "All_Beauty" reviews dataset to perform domain-specific text intelligence tasks. This data powers the automated sentiment analysis and summarization features of the platform

Dataset Overview: The processed dataset contains 5,000 unique product reviews.

Data Structure: Each entry includes three core features:
review_body: The raw text of the customer feedback.
stars: The numerical rating (1–5) serving as the sentiment label.
product_category: Fixed as "All_Beauty" for domain-specific context.

Text Statistics:
Average Review Length: ~71 words, providing sufficient context for sentiment extraction.
95th Percentile Length: 227 words, which informs the tokenization strategy by setting the max_length parameter for the LLM.

Business Application: This data is used to fine-tune a pre-trained Large Language Model (e.g., DistilBERT or GPT-2) to transform unstructured feedback into actionable marketing insights.