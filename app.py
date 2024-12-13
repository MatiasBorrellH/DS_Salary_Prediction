from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
from feature_creation import create_features
from pre_processing import preprocess_data


# Load the trained model
try:
    model = joblib.load('lightgbm_tunned_model.joblib')  # Load your model
except FileNotFoundError:
    raise Exception("Model file 'lightgbm_tunned_model.joblib' not found. Make sure it's in the correct path.")

# Initialize FastAPI
app = FastAPI()

# Define the structure of the expected JSON
class PredictionRequest(BaseModel):
    data: list  # List of dictionaries containing the input data

@app.post("/predict")
async def predict(payload: PredictionRequest):
    """
    Endpoint for making predictions using a JSON file.

    Parameters:
    - payload: JSON containing the input data.

    Returns:
    - JSON with the predictions.
    """
    try:
        # Convert the JSON into a DataFrame
        input_data = pd.DataFrame(payload.data)
        input_data_pre = preprocess_data(input_data)
        input_data_feat = create_features(input_data_pre)
        
        catcol = ["experience_level", "employment_type", "job_title", 
                "employee_residence", "remote_ratio", "company_location", "company_size"]

        # Apply same Label Encoding from train to each categorical column
        for col in catcol:
            le = joblib.load("data_encoder.pkl")  
            input_data_feat[col] = le.fit_transform(input_data_feat[col])


        # Make predictions
        predictions = model.predict(input_data_feat)

        # Return the predictions
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))