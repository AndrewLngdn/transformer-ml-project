from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the sentiment analysis pipeline from Hugging Face Transformers
# This will download the model the first time it's run
sentiment_analyzer = pipeline("sentiment-analysis")

# Define a Pydantic model for input validation
class TextInput(BaseModel):
    text: str

# Define a Pydantic model for output
class SentimentOutput(BaseModel):
    label: str
    score: float

# Create an API endpoint for sentiment analysis
@app.post("/analyze-sentiment", response_model=SentimentOutput)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyzes the sentiment of the provided text using a pre-trained Hugging Face model.
    """
    result = sentiment_analyzer(input_data.text)[0]
    return SentimentOutput(label=result['label'], score=result['score'])

# To run this application:
# 1. Save the code as a Python file (e.g., main.py).
# 2. Install necessary libraries: pip install fastapi uvicorn transformers pydantic
# 3. Run the application using uvicorn: uvicorn main:app --reload