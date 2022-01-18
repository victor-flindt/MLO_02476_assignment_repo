import time
import requests
url = 'https://europe-west1-useful-circle-337909.cloudfunctions.net/cloud_deployment_fucntion'
payload = {'message': 'Hello, General Kenobi'}

for _ in range(10):
   r = requests.get(url, params=payload)