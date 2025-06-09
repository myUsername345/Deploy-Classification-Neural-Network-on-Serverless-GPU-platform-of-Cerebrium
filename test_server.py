import requests
import argparse

def test_deployed_model(image_path, api_url, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(api_url, headers=headers, files=files)

    if response.status_code == 200:
        print(f"Predicted Class ID: {response.json()['class_id']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__"::
    example_image_path = "images/n01440764_tench.jpg"
    example_api_url = "https://api.cerebrium.ai/my-model"  
    example_api_key = "9371206831lruemck"
    test_deployed_model(example_image_path, example_api_url, example_api_key)
