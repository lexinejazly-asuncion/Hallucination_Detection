# Hallucination_Detection
## Topic of Interest

  The project investigates hallucination detection in Retrieval-Augmented Generation (RAG) systems using ML and NLP techniques. By analyzing the relationship between generated answers and their underlying evidence, this project aims to improve the evaluation and reliability of RAG-based AI systems.

## Research Question
  How effectively can NLP and machine learning models classify AI-generated responses as grounded or hallucinated in a RAG system?

## Dataset
Name: LLM-AggreFact (Large Language Model Aggregated Fact Verification Benchmark)

Source: Curated by Liyan Tang, Philippe Laban, and Greg Durrett

Link: https://huggingface.co/datasets/lytang/LLM-AggreFact

## Summary
  Artificial intelligence (AI) has grown into an ubiquitous tool. Across industries, AI systems are being adopted to automate systems, analyze and synthesize information, and most importantly, influence decision making, which affects both communities and individuals. This is especially important to consider when building systems because they have the potential to make mistakes, namely produce hallucinations.
  
  To answer our research question, we will analyze the LLM-AggreFact dataset, which contains document–claim pairs collected from multiple factual consistency benchmarks. Each sample has a binary label indicating whether the claim is supported by the source document (1) or hallucinated/not supported (0). These text pairs will be processed using Natural Language Processing (NLP) techniques and trained using Logistic Regression and Random Forest models. Analyzing this data will allow the models to learn patterns that distinguish grounded from hallucinated claims, helping us evaluate how effectively machine learning can detect hallucinations in AI-generated responses.
  
  Our findings will help individuals and organizations make more informed decisions when deploying or interacting with RAG systems. Since they are becoming widely trusted and often guide decision making, building a tool that can clearly classify outputs can improve the collective understanding of language models and help track error trends that lead to hallucinations and eventually misinformation. This tool, paired with human validation, can serve as a guardrail layer, ultimately preventing costly errors in high stakes domains ensuring that user facing AI tools remain safe and reliable.

______________________________________________________________________________________________________________________


## First Time Set Up  
### Prerequisites  
1. Install Python  

### Downloading dependencies 
Create a local copy of the repository:  
```
git clone https://github.com/lexinejazly-asuncion/Hallucination_Detection.git
```

Make sure you are in the correct folder: 'Hallucination_Detection', if not run:
```
cd Hallucination_Detection
```

This application runs on a virtual environment.   
To create a virtual environment:  
```
python3 -m venv venv
```
*Note: Creating a virtual environment only needs to be done the first time the project is set up.*  

Activate the virtual environment:  
```
. venv/bin/activate
```
Install application dependencies and libraries:  
```
pip install -r requirements.txt
```


## How to run this application  
### Step 0a: Check that the virtual environment is activated  
If your virtual environment is activated, it should say (venv) in your command line (Go to Step 0b)  
If it's not, activate the virtual environment:  
```
. venv/bin/activate
```

### Step 0b: Check if the Models and Data are initialized
If they are initialized, you will see 2 files in the ./Artifacts directory: logistic_regression_model_joblib, tfidf_vectorizer.joblib  (SKIP to Step 1)  
If any of these 3 files are missing, delete the Artifacts folder (if it exists) 

**IMPORTANT**: This step must be completed when starting the web application for the first time!   

Run the main.py script:  
```
python3 main.py
```

This process will:  
- Pre-process the dataset 
- Train the Lexical (TF-IDF) and Semantic (Sentence Transformer) models   
- Save all models to the ./Artifacts directory  

### Step 0c: Authorize use for dataset

Run the main.py script:  
```
huggingface-cli login
hf auth login
```
NOTE: You need to have access to the dataset to train the model

### Step 1: Run the Streamlit app

Run the app.py script:
```
streamlit run app.py
```

Deactivate virtual environment: 
```
deactivate
```