# Hallucination_Detection
## Topic of Interest

  The project investigates hallucination detection in Retrieval-Augmented Generation (RAG) systems using ML and NLP techniques. By analyzing the relationship between generated answers and their underlying evidence, this project aims to improve the evaluation and reliability of RAG-based AI systems.

## Research Question
  How effectively can NLP and machine learning models classify AI-generated responses as grounded or hallucinated in a RAG system?

## Summary
  Artificial intelligence (AI) has grown into an ubiquitous tool. Across industries, AI systems are being adopted to automate systems, analyze and synthesize information, and most importantly, influence decision making, which affects both communities and individuals. This is especially important to consider when building systems because they have the potential to make mistakes, namely produce hallucinations.
  
  To answer our research question, we will analyze the HaluBench dataset, which contains samples structured as context, question, and answer triplets. Each sample has a binary label indicating whether the AI-generated answer is grounded in the provided context or contains a hallucination. These textual triplets will be processed using Natural Language Processing (NLP) techniques and trained using Logistic Regression and Random Forest models. Analyzing this data will allow the models to learn to classify between grounded and hallucinated outputs, directly helping us determine exactly how effectively machine learning can audit RAG based AI outputs.
  
  Our findings will help individuals and organizations make more informed decisions when deploying or interacting with RAG systems. Since they are becoming widely trusted and often guide decision making, building a tool that can clearly classify outputs can improve the collective understanding of language models and help track error trends that lead to hallucinations and eventually misinformation. This tool, paired with human validation, can serve as a guardrail layer, ultimately preventing costly errors in high stakes domains ensuring that user facing AI tools remain safe and reliable.

____________________________________________________________________________________________________________________



