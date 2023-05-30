import os
import pandas as pd
from transformers import pipeline

# Load the dataset
df = pd.read_csv('in/Fed_Scrape-2015-2023.csv')

# Delete rows where "Type" is "1"
df = df[df['Type'] != 1]

# Group paragraphs by date
grouped_df = df.groupby('Date')['Text'].apply(' '.join).reset_index()

# Initialize the sentiment analysis classifier
classifier = pipeline("sentiment-analysis")

# Analyze sentiment for each group
sentiments = []
for _, group in grouped_df.iterrows():
    text = group['Text']
    max_length = 512  # Maximum sequence length supported by the model
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    chunk_sentiments = []
    for chunk in chunks:
        result = classifier(chunk)
        sentiment = result[0]['label']
        chunk_sentiments.append(sentiment)
    # Assign the most frequent sentiment to the group
    sentiment = max(set(chunk_sentiments), key=chunk_sentiments.count)
    sentiments.append(sentiment)

# Add the sentiment column to the grouped dataframe
grouped_df['Sentiment'] = sentiments

# Create the "out" folder if it doesn't exist
os.makedirs("out", exist_ok=True)

# Save the grouped dataframe to a CSV file
output_path = "out/hf_sentiment_analysis_results.csv"
grouped_df.to_csv(output_path, index=False)

print("Sentiment analysis results saved to:", output_path)