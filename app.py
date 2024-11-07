import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import openrouteservice
from openrouteservice import convert

from sklearn.preprocessing import LabelEncoder
import base64
from geopy.geocoders import Nominatim
from io import BytesIO
from sklearn.model_selection import GridSearchCV
import requests
from flask import Flask, render_template, request 
import plotly.express as px
from plotly.io import to_html
from plotly.subplots import make_subplots
import pandas as pd
import json
import plotly.graph_objects as go
from joblib import load
import numpy as np
from sklearn.preprocessing import QuantileTransformer

#==================================================================================================================================================

# Define class and color of object
classes = [
    "Bus",
    "XeKhach",
    "Bike",
    "Car",
    "Truck",
    "XeBaGac",
    "XeChuyenDung",
    "XeDap",
    "XeContainer"
]

colors = {
    "Bus":(255,255,0),
    "XeKhach":(255,0,255),
    "Bike":(0,255,255),
    "Car":(0,0,255),
    "Truck":(0,255,0),
    "XeBaGac":(255,0,0),
    "XeChuyenDung":(125,125,0),
    "XeDap":(125,0,125),
    "XeContainer":(0,125,125)
}

#==================================================================================================================================================

app = Flask(__name__)

# Load yolov3-tiny weight and config of dataset Vietnam's traffic
weightsPath = './yolov3.weights'
configPath = './yolov3_training.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get class object in model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Select model
# grid = './svm_grid_model_no_date_time.joblib'
grid = './svm_grid_model.joblib'
def load_model():
    return load(grid)

model = load_model()

def predict_traffic(features):
    prediction = model.predict(features.reshape(1, -1))[0]
    print(prediction)
    color_mapping = {0: 'red', 1: 'purple', 2: 'green', 3: 'blue'}
    return color_mapping.get(prediction, color_mapping)

def create_encoder():
    times = ['12:00:00 AM', '12:15:00 AM', '12:30:00 AM', '12:45:00 AM',
             '1:00:00 AM', '1:15:00 AM', '1:30:00 AM', '1:45:00 AM',
             '2:00:00 AM', '2:15:00 AM', '2:30:00 AM', '2:45:00 AM',
             '3:00:00 AM', '3:15:00 AM', '3:30:00 AM', '3:45:00 AM',
             '4:00:00 AM', '4:15:00 AM', '4:30:00 AM', '4:45:00 AM',
             '5:00:00 AM', '5:15:00 AM', '5:30:00 AM', '5:45:00 AM',
             '6:00:00 AM', '6:15:00 AM', '6:30:00 AM', '6:45:00 AM',
             '7:00:00 AM', '7:15:00 AM', '7:30:00 AM', '7:45:00 AM',
             '8:00:00 AM', '8:15:00 AM', '8:30:00 AM', '8:45:00 AM',
             '9:00:00 AM', '9:15:00 AM', '9:30:00 AM', '9:45:00 AM',
             '10:00:00 AM', '10:15:00 AM', '10:30:00 AM', '10:45:00 AM',
             '11:00:00 AM', '11:15:00 AM', '11:30:00 AM', '11:45:00 AM',
             '12:00:00 PM', '12:15:00 PM', '12:30:00 PM', '12:45:00 PM',
             '1:00:00 PM', '1:15:00 PM', '1:30:00 PM', '1:45:00 PM',
             '2:00:00 PM', '2:15:00 PM', '2:30:00 PM', '2:45:00 PM',
             '3:00:00 PM', '3:15:00 PM', '3:30:00 PM', '3:45:00 PM',
             '4:00:00 PM', '4:15:00 PM', '4:30:00 PM', '4:45:00 PM',
             '5:00:00 PM', '5:15:00 PM', '5:30:00 PM', '5:45:00 PM',
             '6:00:00 PM', '6:15:00 PM', '6:30:00 PM', '6:45:00 PM',
             '7:00:00 PM', '7:15:00 PM', '7:30:00 PM', '7:45:00 PM',
             '8:00:00 PM', '8:15:00 PM', '8:30:00 PM', '8:45:00 PM',
             '9:00:00 PM', '9:15:00 PM', '9:30:00 PM', '9:45:00 PM',
             '10:00:00 PM', '10:15:00 PM', '10:30:00 PM', '10:45:00 PM',
             '11:00:00 PM', '11:15:00 PM', '11:30:00 PM', '11:45:00 PM']
    
    le = LabelEncoder()
    le.fit(times)
    return le

def encode_time(time_str, encoder):
    return encoder.transform([time_str])[0]

# Create the encoder
le = create_encoder()

def get_encoded_day(day_name):
    day_to_code = {
        'Monday': 1,
        'Tuesday': 5,
        'Wednesday': 6,
        'Thursday': 4,
        'Friday': 0,
        'Saturday': 2,
        'Sunday': 3
    }
    return day_to_code.get(day_name, -1)  # Returns -1 if day_name is not found

def pred(image):
    
    height,width = image.shape[:2]

    # Detect object
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    # extract output 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.25:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Bbox coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    object_count = {"Bus": 0, "Bike": 0, "Car": 0, "Truck": 0}
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color=colors[label]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(image, label, (x, y), font, 0.5, color, 1)

            # Counting the objects
            if label in object_count:
                object_count[label] += 1
            
    total = sum(object_count.values())

    print("Detected objects:", object_count, total)
    return object_count, total, image

id_file = "./id.txt"
def get_image_realtime(street_name):
    id = ""
    with open(id_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if street_name in line:
                id = line.split(': ')[1].strip()
                break
    print('id:', id)
    if not id:
        raise Exception("No ID for this camera :(")

    # Traffic information website of Ho Chi Minh city, Viet Nam
    image_url = f'http://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id={id}&bg=black&w=300&h=230&t=1730895202943'

    # Define headers with cookies
    headers = {
        'User-Agent': '', # Contains information about the browser and operating system
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Cookie': '', # Put your web's cookie here 
    }

    # Send a GET request with the headers (including cookies) and get the image of camera's id
    response = requests.get(image_url, headers=headers)
    image = Image.open(BytesIO(response.content))
    # image = image.resize((224, 224))  # Resize if needed
    image = np.array(image)
    if image.size == 0:
        raise Exception("Couldn't get to the camera right now :(")
    else:
        return image

client = openrouteservice.Client(key='')   # Replace with your actual API key from Open Route Service

# Get latitude and longitude of the street
def get_lat_long(street_name):
    geolocator = Nominatim(user_agent="myGeolocator", timeout=10)
    location = geolocator.geocode(street_name)
    lat, long = location.raw['lat'], location.raw['lon']
    return long, lat 

@app.route('/', methods=['GET', 'POST'])
def index():
    post_request_made = False
    detected = ''
    total_vhc = ''
    image = None

    # I'm not used with this scatter_map_box so not really know what happening here
    data = {
        'name': ['Heavy', 'Low', 'Normal'],
        'value': [0,1,2]
    }
    df = pd.DataFrame(data)

    # Plot the choropleth map
    fig = px.scatter_mapbox(
        df,
        color = "name",         # Colors the points according to the traffic situation
        text = "name",          # Displays text labels next to each point
        hover_name = "name",    # Shows the name on hover
        mapbox_style = "open-street-map",
        zoom = 15, 
        center = {"lat": 10.7752637, "lon": 106.7017981},
        opacity = 0.5,
        width = 1600,
        height = 1000
    )

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            bearing=0,
            pitch=0
        ),
        showlegend=True
    )

    if request.method == 'POST':
        post_request_made = True
        # Extract date components
        day = int(request.form['day'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        date_str = f"{year}-{month:02d}-{day:02d}"
        date_object = datetime.strptime(date_str, '%Y-%m-%d')
        day_of_week = date_object.strftime('%A')  # Ensure this matches training process
    
        # Extract time components
        hour = int(request.form['hour'])
        am_pm = request.form['am_pm']
        minute = int(request.form['minute'])
    
        # Ensure this match the training preprocessing part
        time_encoded = encode_time(f'{hour}:{minute:02d}:00 {am_pm}', le)
        day_of_week = get_encoded_day(day_of_week)

        street_name_start = request.form.get('street_name_start', '')
        street_name_end = request.form.get('street_name_end', '')
        street_name = street_name_start + " - " + street_name_end

        # Get image from camera and predict
        cam = get_image_realtime(street_name) 
        object_count, total, image = pred(cam)
        detected = "Detected vehicle: " + str(object_count) 
        total_vhc = "Total: " + str(total)
        
        # Prepare input data
        features = np.array([
            time_encoded, 
            day, 
            day_of_week, 
            object_count['Car'], 
            object_count['Bike'], 
            object_count['Bus'], 
            object_count['Truck'], 
            total
        ])  

        # predict traffic flow
        color = predict_traffic(features) 
        
        # Extract lat and long of the street path and start path planning with api
        start_coords = get_lat_long(street_name_start + ", Hồ Chí Minh")
        end_coords = get_lat_long(street_name_end + ", Hồ Chí Minh")
        coords = (start_coords, end_coords)

        route = client.directions(coords)
        geometry = route['routes'][0]['geometry']
        decoded = convert.decode_polyline(geometry)

        path_long = [coord[0] for coord in decoded['coordinates']]
        path_lat = [coord[1] for coord in decoded['coordinates']]

        # print(path_long)
        # print(path_lat)

        # Reload the map 
        fig = px.scatter_mapbox(
            df,
            color = "name",         # Colors the points according to the traffic situation
            text = "name",          # Displays text labels next to each point
            hover_name = "name",    # Shows the name on hover
            mapbox_style = "open-street-map",
            zoom = 15, 
            center = {"lat": float(start_coords[1]), "lon": float(start_coords[0])},
            opacity = 0.5,
            width = 1600,
            height = 1000
        )

        # Draw the path on the map
        line_trace = go.Scattermapbox(
            lat = path_lat,
            lon = path_long,
            mode = 'lines', #+markers
            line = dict(width = 5, color = color),
            name = street_name
        )
        fig.add_trace(line_trace)
        
    # Plot the image to website
    camera_image_html = ""
    if image is not None:
        _, buffer = cv2.imencode('.png', image)  # Encode the image to PNG format
        camera_image_html = base64.b64encode(buffer).decode("utf-8") 

    # Update the map with the new trace
    graph_html = fig.to_html(full_html=False)

    return render_template(
        'index.html', 
        graph_html=graph_html,
        camera_image_html = camera_image_html,
        post_request_made = post_request_made,
        vehicle_detected = detected,
        total = total_vhc
    )

if __name__ == '__main__':
    app.run(port = 5001)
