import requests

# API health check endpoint URL
url = "http://127.0.0.1:8000/health"

try:
    response = requests.get(url)
    
    if response.status_code == 200 and response.json().get("status") == "ok":
        print("API is up and running!")
    else:
        print(f"API health check failed. Status code: {response.status_code}")
        print("Response:", response.text)

except requests.exceptions.ConnectionError as e:
    print(f"Connection Error: {e}")
    print("Please ensure the API server is running by executing 'python api.py'")