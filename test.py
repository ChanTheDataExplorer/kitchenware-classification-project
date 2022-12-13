import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://raw.githubusercontent.com/ChanTheDataExplorer/kitchenware-classification-project/main/testing_images/0000.jpg'}

result = requests.post(url, json=data).json()
print(result)