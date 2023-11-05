***
MODEL DEMO : http://54.209.186.163:8000/
***
# Sentiment Classification Models for App Reviews

This repository contains a collection of machine learning models designed to perform sentiment classification on app reviews. Specifically, it focuses on reviews from the IRCTC app, which were collected using the Google Play Store API.

## Project Overview

The project aims to classify the sentiment of user reviews as positive or negative. Reviews with a rating of 3 stars or higher are labeled as positive, while those with fewer than 3 stars are considered negative. We have experimented with several machine learning models, including Random Forest, LSTM, Multinomial Naive Bayes, and GPT-2, with the latter showing the best performance after fine-tuning.

![Sentiment Analysis Process](path/to/your/sentiment-analysis-process-image.png)

## Dataset

The dataset consists of reviews collected from the IRCTC app via the Google Play Store API. It has been preprocessed and labeled according to the star rating associated with each review.

## Models

The following models were trained:
- Random Forest Classifier
- LSTM (Long Short-Term Memory) network
- Multinomial Naive Bayes Classifier
- Fine-tuned GPT-2 (Generative Pretrained Transformer 2)
