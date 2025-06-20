import requests
import json

# API endpoint URL
url = "http://127.0.0.1:8000/predict/"

# Path to the image file
file_path = "samples/InfominaBerhad2024/statements/page_14_statements_of_comprehensive_income.png"

# Open the file in binary mode
with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/png")}
    
    # Send the POST request
    try:
        response = requests.post(url, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Successfully received response from API:")
            # Pretty print the JSON response
            response_json = response.json()
            print(json.dumps(response_json, indent=2))

            # Save the response to a file
            with open("api_test_output.json", "w", encoding="utf-8") as f:
                json.dump(response_json, f, indent=2, ensure_ascii=False)
            print("\nResponse saved to api_test_output.json")
        else:
            print(f"Error: Received status code {response.status_code}")
            print("Response content:", response.text)
            
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
        print("Please ensure the API server is running by executing 'python api.py'")