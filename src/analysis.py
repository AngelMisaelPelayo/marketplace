import pandas as pd
import os
import matplotlib.pyplot as plt

def use_processed_data():
    # Read the CSV file into a DataFrame
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define the path to the 'data' folder, which is one level above 'src'
    data_folder = os.path.join(script_dir, '..', 'data')
    # Ensure the 'data' folder exists
    os.makedirs(data_folder, exist_ok=True)
    # Define the path for the CSV file
    csv_file_path = os.path.join(data_folder, 'processed_data.csv')
    processed_df = pd.read_csv(csv_file_path)
    
    # Now we can use the DataFrame in this script
    print("Processed DataFrame from CSV file:")
    print(processed_df)
    
    # Perform sentiment analysis and plot pie chart
    plot_sentiment_pie(processed_df)
    
    # Perform category analysis and plot bar chart
    plot_category_bar(processed_df)

def plot_sentiment_pie(df):
    """
    Plot a pie chart for sentiment distribution.
    """
    # Count the occurrences of each sentiment
    sentiment_counts = df['sentiment'].value_counts()
    
    # Map sentiment values to labels
    sentiment_labels = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    sentiments = [sentiment_labels[sentiment] for sentiment in sentiment_counts.index]
    
    # Plot the pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiments, autopct='%1.1f%%', colors=['green', 'yellow', 'red'])
    plt.title('Sentiment Distribution')
    plt.show()

def plot_category_bar(df):
    """
    Plot a bar chart for categories, where each bar is divided based on sentiment.
    """
    # Group by category and sentiment
    category_sentiment_counts = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
    
    # Map sentiment values to colors
    sentiment_colors = {1: 'green', 0: 'yellow', -1: 'red'}
    
    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize the bottom for stacking
    bottom = None
    
    # Iterate over each sentiment and add to the bar chart
    for sentiment, color in sentiment_colors.items():
        if sentiment in category_sentiment_counts.columns:
            ax.bar(category_sentiment_counts.index, category_sentiment_counts[sentiment], color=color, label=sentiment_colors[sentiment], bottom=bottom)
            if bottom is None:
                bottom = category_sentiment_counts[sentiment]
            else:
                bottom += category_sentiment_counts[sentiment]
    
    # Add labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Category Distribution with Sentiment')
    ax.legend(title='Sentiment')
    
    plt.show()

if __name__ == "__main__":
    use_processed_data()

