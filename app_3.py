from flask import Flask, render_template_string     # Import Flask and a function to render HTML templates from strings
import plotly.express as px                         # Import Plotly Express for quick plotting
import pandas as pd                                 # Import Pandas for data manipulation

# Initialize the Flask application
app = Flask(__name__)  


# Step 1: Create sample data
data = {
    'latitude': [10.7769, 10.7802, 10.7823, 10.7850],  # Latitude coordinates
    'longitude': [106.7009, 106.6998, 106.6959, 106.6900],  # Longitude coordinates
    'traffic_density': [50, 30, 80, 100]  # Traffic density values
}

# Create a DataFrame from the data dictionary
df = pd.DataFrame(data)  


@app.route('/')  # Define the route for the main page (index page)
def index():
    # Step 2: Create a Plotly mapbox scatter plot
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', size='traffic_density',
                            color='traffic_density', zoom=14, title="Traffic Density",
                            mapbox_style="open-street-map", hover_name='traffic_density')

    # Configure the map layout, centering it based on the average latitude and longitude
    fig.update_layout(mapbox=dict(center= {"lat": df['latitude'].mean(), "lon": df['longitude'].mean()}, zoom=14))
    
    # Remove margins from the figure layout for a cleaner look
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Convert the Plotly figure to an HTML snippet
    graph_html = fig.to_html(full_html=False)

    # Return an HTML page, with the embedded Plotly chart (graph_html)
    return render_template_string('''
        <html>
            <head>
                <title>Flask Plotly Map</title>  <!-- Page title -->
            </head>
            <body>
                <h1>Traffic Density</h1>  <!-- Header for the page -->
                {{ graph_html|safe }}  <!-- Render the Plotly graph HTML safely -->
            </body>
        </html>
    ''', graph_html=graph_html)  # Pass the graph_html as a variable to the template

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
