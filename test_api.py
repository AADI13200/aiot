import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "area": "Market",
    "hour": 19,
    "day": "Weekday",
    "vehicles": 70,
    "pedestrians": 25,
    "lux": 15,
    "motion": 1,
    "traffic": "High"
}

response = requests.post(
    url,
    json=data
)

print("Server response:")
print(response.json())