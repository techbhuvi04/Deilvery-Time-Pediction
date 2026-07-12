from datetime import datetime

import folium
import polyline
import requests
import streamlit as st


def get_time_of_day():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


@st.cache_data(ttl=300)
def get_place_suggestions(query, api_key):
    if len(query.strip()) < 3:
        return []
    url = "https://places.googleapis.com/v1/places:autocomplete"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
    }
    payload = {
        "input": query.strip(),
        "includedRegionCodes": ["in"],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=8)
    response.raise_for_status()
    suggestions = []
    for item in response.json().get("suggestions", []):
        prediction = item.get("placePrediction")
        if prediction:
            suggestions.append({
                "name": prediction["text"]["text"],
                "place_id": prediction["placeId"],
            })
    return suggestions[:5]


@st.cache_data(ttl=3600)
def get_place_coordinates(place_id, api_key):
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "location",
    }
    response = requests.get(url, headers=headers, timeout=8)
    response.raise_for_status()
    location = response.json()["location"]
    return location["latitude"], location["longitude"]


@st.cache_data(ttl=600)
def get_weather(latitude, longitude, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
    }
    response = requests.get(url, params=params, timeout=8)
    response.raise_for_status()
    condition = response.json()["weather"][0]["main"]
    weather_map = {
        "Clear": "Clear",
        "Rain": "Rainy",
        "Drizzle": "Rainy",
        "Thunderstorm": "Rainy",
        "Clouds": "Cloudy",
        "Snow": "Snowy",
    }
    return weather_map.get(condition, "Clear")


@st.cache_data(ttl=300)
def get_route_data(origin_place_id, destination_place_id, api_key):
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "routes.distanceMeters,"
            "routes.duration,"
            "routes.staticDuration,"
            "routes.polyline.encodedPolyline"
        ),
    }
    payload = {
        "origin": {"placeId": origin_place_id},
        "destination": {"placeId": destination_place_id},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    routes = response.json().get("routes", [])
    if not routes:
        raise ValueError("No delivery route found.")
    route = routes[0]
    distance = route["distanceMeters"] / 1000
    live_time = float(route["duration"].removesuffix("s")) / 60
    static_time = float(route["staticDuration"].removesuffix("s")) / 60
    encoded_polyline = route["polyline"]["encodedPolyline"]
    route_coordinates = polyline.decode(encoded_polyline)
    return distance, live_time, static_time, route_coordinates


def calculate_traffic(live_time, static_time):
    if static_time <= 0:
        return "Low"
    traffic_ratio = live_time / static_time
    if traffic_ratio < 1.15:
        return "Low"
    if traffic_ratio < 1.35:
        return "Medium"
    if traffic_ratio < 1.60:
        return "High"
    return "Very High"


def calculate_risk(prediction):
    if prediction <= 30:
        return "LOW", "ON TIME"
    if prediction <= 45:
        return "MEDIUM", "AT RISK"
    return "HIGH", "DELAYED"


def build_eta_breakdown(prediction, live_time, prep_time):
    rule_based_eta = prep_time + live_time
    ml_context_buffer = prediction - rule_based_eta
    return {
        "ml_eta": prediction,
        "live_travel": live_time,
        "prep_time": prep_time,
        "rule_based_eta": rule_based_eta,
        "ml_context_buffer": ml_context_buffer,
    }


def create_delivery_map(route_coordinates, restaurant_name, customer_name):
    if not route_coordinates:
        return None
    start_point = route_coordinates[0]
    end_point = route_coordinates[-1]
    route_map = folium.Map(
        location=start_point,
        zoom_start=13,
        control_scale=True,
        tiles="CartoDB dark_matter",
    )
    folium.PolyLine(
        locations=route_coordinates,
        weight=6,
        opacity=0.9,
        tooltip="Delivery Route",
    ).add_to(route_map)
    folium.Marker(
        location=start_point,
        tooltip="Restaurant",
        popup=restaurant_name,
        icon=folium.Icon(icon="cutlery", prefix="fa"),
    ).add_to(route_map)
    folium.Marker(
        location=end_point,
        tooltip="Customer",
        popup=customer_name,
        icon=folium.Icon(icon="home", prefix="fa"),
    ).add_to(route_map)
    rider_index = len(route_coordinates) // 2
    rider_position = route_coordinates[rider_index]
    folium.Marker(
        location=rider_position,
        tooltip="Delivery Partner",
        icon=folium.DivIcon(
            html="""
            <div style="
                font-size: 30px;
                transform: translate(-15px, -15px);
            ">
                🛵
            </div>
            """
        ),
    ).add_to(route_map)
    route_map.fit_bounds(route_coordinates)
    return route_map
