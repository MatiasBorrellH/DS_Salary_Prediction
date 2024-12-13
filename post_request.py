import requests

url = "http://127.0.0.1:8000/predict"

with open("test_json", "rb") as file:
    files = {"file": file}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()  
        print("Prediction:", response.json()["predict"])

    except requests.exceptions.HTTPError as err:
        
        if response.status_code == 400:
            print("Error 400: Bad request. Check the data sent.")
        elif response.status_code == 401:
            print("Error 401: Unauthorized. Check your credentials.")
        elif response.status_code == 403:
            print("Error 403: Forbidden. You do not have permission to access this resource.")
        elif response.status_code == 404:
            print("Error 404: Resource not found. Verify the URL.")
        elif response.status_code == 500:
            print("Error 500: Internal server error. Please try again later.")
        elif response.status_code == 503:
            print("Error 503: Service unavailable. Please try again later.")
        else:
            print(f"HTTP error {response.status_code}: {err}")

    except requests.exceptions.RequestException as err:
        print(f"Connection or network error: {err}")