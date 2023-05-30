# Assignment 5 (self-assigned) - Sentiment Analysis

## Contribution
The code in this assignment was developed by myself using packages from HuggingFace and VADER Sentiment Analysis.

## Description of the Project
For this project, I wanted to explore the concept of sentiment analysis for finance and economics. A real-world application for sentiment analysis can be found in financial markets, where there is a huge monetary incentive to be the first to react to financial news. We are at a stage where computers can digest text much faster than humans, so an entity that has entity that has access to the fastest and most accurate model for sentiment analysis can reap huge profits by front running the rest of market participants when market moving news are published.

The project focuses on performing simple sentiment analysis on statements made by the Federal Reserve FOMC (Federal Open Market Committee), which is one of the most followed entities in financial markets. 

## Data
The dataset used in this project consists of text from Federal Reserve FOMC meeting minutes and statements. The data was downloaded from Kaggle [here](https://www.kaggle.com/datasets/drlexus/fed-statements-and-minutes), where it was originally collected by scraping the Federal Reserve's website using a custom Python scraper built with BeautifulSoup. The dataset spans the period between 2015 and 2023 and provides insights into the central bank's monetary policy decisions and discussions.

## Methods
Two scripts were used in this project: one utilizing the HuggingFace pipeline and the other using VADER Sentiment Analysis.

### HuggingFace Script
- The `hf.py` script loads the dataset and filters out meeting minutes, keeping the statements (as they are shorter in length).
- The statements are grouped by date.
- The sentiment analysis classifier from HuggingFace is initialized.
- Sentiment analysis is performed for each statement by splitting the text into chunks (considering the maximum sequence length supported by the model).
- The most frequent sentiment is assigned to each statement.
- The sentiment results are added as a new column to the grouped dataframe.
- The grouped dataframe is saved to a CSV file in the "out" folder.

### VADER Script
- The `vader.py` script follows a similar process to the HuggingFace script, but instead uses the VADER SentimentIntensityAnalyzer.
- Sentiment analysis is performed for each group using VADER.
- Sentiment labels (Positive, Negative, Neutral) are assigned based on the compound score, which is described as a normalized, weighted composite score. The labels are assigned using this recommended logic:
  - positive sentiment: compound score >= 0.05
  - neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
  - negative sentiment: compound score <= -0.05
- The sentiment results are added as a new column to the grouped dataframe.
- The grouped dataframe is saved to a CSV file in the "out" folder.

## Usage and Reproducibility
To use this code on your own device, follow these steps:
1. Clone this GitHub repository to your local device.
2. Install the required packages by navigating to the root folder and running `pip install -r requirements.txt` in your terminal.
3. Run the script by executing either `python src/hf.py` (for the HuggingFace pipeline) or `python src/vader.py` (for VADER Sentiment Analysis) in your terminal.
4. Find the results in the "out" folder as .csv files.

*Note: This code was successfully executed in Coder Python 1.76.1 on uCloud. Your terminal commands may need to vary slightly depending on your device.*

## Discussion of Results
The sentiment analysis results provided a basic overview of the sentiment expressed in the Federal Reserve FOMC statements. Of course, the approach taken in this assignment is very much proof-of-concept. In reality, sentiment analysis models for financial markets need to consider countless variables to be more accurate and reliable.

An interesting note was that the HuggingFace pipeline took several minutes to produce the results, while VADER Sentiment Analysis finished almost instantly. This brings to attention the question of speed vs accuracy - a question where the answer in my opinion leans towards that speed is more important than accuracy in the context of financial markets' sentiment analysis. A trader who uses the fastest model with decent accuracy will probably beat the trader who uses the most accurate model with decent speed, simply because he is able to execute his trades first, before the prices have had time to adjust.

In conclusion, the results offer an interesting starting point for further analysis into the role of sentiment analysis in financial markets. A more comprehensive project could combine the sentiment data with price data to investigate if the market has historically moved down on news classified as negative, or up on positive news. This could lead to a deeper understanding of how sentiment affects financial markets and potentially provide insights for market participants.
