import pandas as pd
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.cloud.aiplatform_v1.types import PredictRequest
import os
import re

# Set your Google Cloud Project ID and Location
PROJECT_ID = 'your-project-id'
LOCATION = 'us-central1'

# Initialize Vertex AI client
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Initialize Prediction Service Client
client = PredictionServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
)

def classify_text(text):
    # Define the prompt
    prompt = f"""
    Your task is reviewing customer questions and complaints then classify them in one of the following categories: 
    a: affordability-price, b: clearness in the product description, c: product quality, d: other. 
    Then determint the sentiment of the comment as: 1:positive, 0:neutral, or -1:negative. Return only the category followed
    by the sentiment separated by comma, the valid answers are like: 
    a,1
    b,0
    c,-1
    Now classify the following text: {text}
    """
    
    # Prepare the request
    request = PredictRequest(
        endpoint=f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-1.5-flash",
        instances=[{"content": prompt}]
    )
    
    # Send the request to the model
    response = client.predict(request=request)
    
    # Parse the response
    predictions = response.predictions
    if predictions:
        return predictions[0]['content']
    else:
        return "No response from model"

def parse_model_response(response):
    """
    Parse the model's response to extract the category (a, b, c, d) and sentiment (1, 0, -1).
    The model's response is expected to be in the format: "a, 1" or "b, -1", etc.
    """
    # Use regex to extract the category and sentiment
    match = re.match(r'^\s*([a-d])\s*,\s*(-1|0|1)', response)
    if match:
        category = match.group(1)  # Extract the category letter
        sentiment = int(match.group(2))  # Extract the sentiment as an integer
        return category, sentiment
    else:
        return None, None  # Return None if the response format is unexpected

def process_dataframe(df, text_column):
    # Iterate over each row in the DataFrame
    df['classification'] = df[text_column].apply(classify_text)
    
    # Parse the classification column to extract category and sentiment
    df['category'] = df['classification'].apply(lambda x: parse_model_response(x)[0])
    df['sentiment'] = df['classification'].apply(lambda x: parse_model_response(x)[1])
    
    return df

if __name__ == "__main__":
    # Load your DataFrame, this would load a data from mercado libre or whatever the marketplace is
    # as an example let us use:
    df = pd.DataFrame({
        'customer_text': [
            "The price is too high for this product.",
            "The product description was very clear and detailed.",
            "The quality of the product is poor.",
            "I love this product, it's worth every penny."
        ]
    })
    
    # Process the DataFrame
    processed_df = process_dataframe(df, 'customer_text')

    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define the path to the 'data' folder, which is one level above 'src'
    data_folder = os.path.join(script_dir, '..', 'data')
    # Ensure the 'data' folder exist
    os.makedirs(data_folder, exist_ok=True)
    # Define the path for the CSV file
    csv_file_path = os.path.join(data_folder, 'processed_data.csv')
    # Save the DataFrame to the 'data' folder
    processed_df.to_csv(csv_file_path, index=False) #we will analyse this process dataframe in others python scripts
    print(f"File saved to: {csv_file_path}")
