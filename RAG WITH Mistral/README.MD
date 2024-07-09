# Transformer-based Question Answering System

This project implements a question answering system using transformer-based models and document retrieval techniques.

## Installation

Install the required packages:

```bash
pip install --upgrade langchain langchain-community langchainhub langchain-google-genai langchain-chroma bs4 faiss-cpu
pip install --upgrade langchain-huggingface text-generation transformers google-search-results numexpr langchainhub sentencepiece jinja2
pip install transformers[torch] -U
pip install BitsAndBytesConfig


Usage

 1. Set up your Google API Key:



2. Load and process documents:




3. Set up the language model:



4.Create the question answering chain:



5.Get answers to questions:

