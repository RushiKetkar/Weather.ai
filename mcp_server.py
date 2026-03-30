import json
import math
import requests
import plotly.graph_objects as go
from mcp.server import FastMCP

mcp = FastMCP("Weather Data Provider")

API_KEY  = "9ea58055bc8f80cb3fc0dc0c4ccbf84f"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

@mcp.tool(description=(
    "Check whether a city name is listed in location_data.txt. "
    "Returns JSON with 'found' (bool) and 'city'. "
    "If not found, also returns 'available' with all known cities."
))
def get_location(city: str) -> str:
    try:
        with open("location_data.txt", "r") as fh:
            locations = [line.strip() for line in fh if line.strip()]
        normalised = [loc.lower() for loc in locations]
        if city.lower() in normalised:
            canonical = locations[normalised.index(city.lower())]
            return json.dumps({"found": True, "city": canonical})
        return json.dumps({"found": False, "city": city, "available": locations})
    except FileNotFoundError:
        return json.dumps({"error": "location_data.txt not found"})

@mcp.tool(description=(
    "Fetch current weather for a city from OpenWeatherMap. "
    "Returns JSON with: city, temp (°C), humidity (%), "
    "wind_speed (m/s), dew_point (°C), and description."
))
def get_weather(city: str) -> str:
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    resp = requests.get(BASE_URL, params=params, timeout=10)

    if resp.status_code != 200:
        return json.dumps({"error": "API call failed", "status": resp.status_code,
                           "message": resp.text})

    raw = resp.json()
    temp     = float(raw["main"]["temp"])
    humidity = float(raw["main"]["humidity"])
    wind_speed = float(raw["wind"]["speed"])

    # Magnus formula for dew point (°C)
    a, b  = 17.625, 243.04
    if humidity > 0:
        alpha     = math.log(humidity / 100.0) + (a * temp) / (b + temp)
        dew_point = round((b * alpha) / (a - alpha), 2)
    else:
        dew_point = temp  # edge case: 0 % humidity

    result = {
        "city":        city,
        "temp":        round(temp, 2),
        "humidity":    round(humidity, 2),
        "wind_speed":  round(wind_speed, 2),
        "dew_point":   dew_point,
        "description": raw["weather"][0]["description"],
    }
    return json.dumps(result)

@mcp.tool(description=(
    "Create a Plotly bar chart for weather metrics "
    "(temp, humidity, wind_speed, dew_point) and save it as an HTML file. "
    "Returns the output filename."
))
def plot_weather(
    city:       str,
    temp:       float,
    humidity:   float,
    wind_speed: float,
    dew_point:  float,
) -> str:
    labels = [
        "Temperature (°C)",
        "Humidity (%)",
        "Wind Speed (m/s)",
        "Dew Point (°C)",
    ]
    values = [float(temp), float(humidity), float(wind_speed), float(dew_point)]
    colours = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    fig = go.Figure(data=[
        go.Bar(x=labels, y=values, marker_color=colours, text=values,
               textposition="outside")
    ])
    fig.update_layout(
        title=f"Current Weather — {city}",
        xaxis_title="Weather Fields",
        yaxis_title="Values",
        template="plotly_dark",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    filename = f"weather_{city.lower().replace(' ', '_')}.html"
    fig.write_html(filename)
    return json.dumps({"success": True, "file": filename})

if __name__ == "__main__":
    print("Starting MCP Weather Server on http://localhost:8000/sse …")
    mcp.run(transport="sse")
