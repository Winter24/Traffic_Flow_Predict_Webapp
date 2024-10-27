import requests

def get_geocode(address):
    # Geocoding API endpoint
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': address,
        'key': ''
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f'Failed to retrieve data: status code {response.status_code}')

# Replace '1600 Amphitheatre Parkway, Mountain View, CA' with your address
get_geocode('1600 Amphitheatre Parkway, Mountain View, CA')
