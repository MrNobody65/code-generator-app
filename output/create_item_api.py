```
import requests

data = {"name": "John Doe", "age": 30}
response = requests.post("http://localhost:5000/items", json=data)
print(response.text)
```