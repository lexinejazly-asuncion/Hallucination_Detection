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

