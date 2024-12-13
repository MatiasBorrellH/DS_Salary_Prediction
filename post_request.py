import requests
import json  
import joblib 


url = "http://127.0.0.1:8000/predict"

# Load the JSON data from the file
with open('test_json', 'r') as file:
    data = json.load(file)  # Load the JSON content

# Make the POST request
response = requests.post(url, json=data)

# Print the response
if response.status_code == 200:
    print("Predictions:", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")