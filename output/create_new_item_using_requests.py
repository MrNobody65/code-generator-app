import requests

create_new_item = requests.post("http://127.0.0.1:5000/api/items", json={"name": "new_item", "description": "This is a new item."})