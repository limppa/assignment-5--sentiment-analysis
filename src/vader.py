import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv('in/Fed_Scrape-2015-2023.csv')

# Delete rows where "Type" is "1"
df = df[df['Type'] != 1]

# Group paragraphs by date
grouped_df = df.groupby('Date')['Text'].apply(' '.join).reset_index()

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis using VADER for each group
sentiments = []
for _, group in grouped_df.iterrows():
    text = group['Text']
    sentiment_scores = analyzer.polarity_scores(text)
    
    # Assign sentiment label based on the compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    sentiments.append(sentiment)

# Add the sentiment column to the grouped dataframe
grouped_df['Sentiment'] = sentiments

# Create the "out" folder if it doesn't exist
os.makedirs("out", exist_ok=True)

# Save the grouped dataframe to a CSV file
output_path = "out/vader_sentiment_analysis_results.csv"
grouped_df.to_csv(output_path, index=False)

print("Sentiment analysis results saved to:", output_path)