import streamlit as st
import pandas as pd
import pickle
import json
import sqlite3
import requests
import folium
from folium.plugins import AntPath, Fullscreen
import polyline
import math
from streamlit_folium import st_folium
from datetime import datetime

# Avoid a pandas/pyarrow segfault seen when building DataFrames with
# string columns on some pandas/pyarrow version pairs.
try:
    pd.options.future.infer_string = False
except Exception:
    pass

st.set_page_config(layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_data():
    with open("model/model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("model/model_info.json", "r") as file:
        model_info = json.load(file)
    return model, model_info

# ---------------- DATABASE ----------------
def init_db():
    with sqlite3.connect("predictions.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                distance REAL,
                weather TEXT,
                traffic TEXT,
                time_of_day TEXT,
                vehicle TEXT,
                prep_time REAL,
                experience REAL,
                predicted_time REAL
            )
        """)

def save_prediction(distance, weather, traffic, time_of_day, vehicle,
                     prep_time, experience, predicted_time):
    with sqlite3.connect("predictions.db") as conn:
        conn.execute("""
            INSERT INTO predictions (
                timestamp, distance, weather, traffic, time_of_day,
                vehicle, prep_time, experience, predicted_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            distance, weather, traffic, time_of_day, vehicle,
            prep_time, experience, float(predicted_time)
        ))

def get_recent_predictions(limit=20):
    with sqlite3.connect("predictions.db") as conn:
        return pd.read_sql(
            "SELECT timestamp, distance, traffic, predicted_time "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            conn, params=(limit,)
        )

# ---------------- HELPERS ----------------
def get_time_of_day():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"

def _is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def _sanitize_route_coordinates(route_coordinates):
    clean = []
    for point in route_coordinates or []:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            lat, lon = point
            if _is_finite(lat) and _is_finite(lon):
                clean.append((float(lat), float(lon)))
    return clean

# ---------------- GOOGLE / WEATHER APIS ----------------
@st.cache_data(ttl=300)
def get_place_suggestions(query, api_key):
    query = query.strip()
    if len(query) < 3:
        return []

    url = "https://places.googleapis.com/v1/places:autocomplete"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key}
    payload = {"input": query, "includedRegionCodes": ["in"]}

    response = requests.post(url, headers=headers, json=payload, timeout=8)
    if response.status_code != 200:
        st.error(f"Google Places error {response.status_code}: {response.text}")
        return []

    suggestions = []
    for item in response.json().get("suggestions", []):
        prediction = item.get("placePrediction")
        if prediction:
            suggestions.append({
                "name": prediction["text"]["text"],
                "place_id": prediction["placeId"]
            })
    return suggestions[:5]

@st.cache_data(ttl=3600)
def get_place_coordinates(place_id, api_key):
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "location"}
    response = requests.get(url, headers=headers, timeout=8)
    response.raise_for_status()
    location = response.json()["location"]
    return location["latitude"], location["longitude"]

@st.cache_data(ttl=600)
def get_weather(latitude, longitude, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": latitude, "lon": longitude, "appid": api_key}
    response = requests.get(url, params=params, timeout=8)
    response.raise_for_status()
    condition = response.json()["weather"][0]["main"]
    weather_map = {
        "Clear": "Clear", "Rain": "Rainy", "Drizzle": "Rainy",
        "Thunderstorm": "Rainy", "Clouds": "Cloudy", "Snow": "Snowy"
    }
    return weather_map.get(condition, "Clear")

@st.cache_data(ttl=300)
def get_route_data(origin_place_id, destination_place_id, api_key):
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "routes.distanceMeters,routes.duration,"
            "routes.staticDuration,routes.polyline.encodedPolyline"
        )
    }
    payload = {
        "origin": {"placeId": origin_place_id},
        "destination": {"placeId": destination_place_id},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE"
    }
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()

    routes = response.json().get("routes", [])
    if not routes:
        raise ValueError("No delivery route found.")

    route = routes[0]
    distance = route["distanceMeters"] / 1000
    live_time = float(route["duration"].removesuffix("s")) / 60
    static_time = float(route["staticDuration"].removesuffix("s")) / 60

    route_coordinates = _sanitize_route_coordinates(
        polyline.decode(route["polyline"]["encodedPolyline"])
    )
    if len(route_coordinates) < 2:
        raise ValueError("Decoded route has too few valid points to display.")

    return distance, live_time, static_time, route_coordinates

def calculate_traffic(live_time, static_time):
    if static_time <= 0:
        return "Low"
    ratio = live_time / static_time
    if ratio < 1.15:
        return "Low"
    if ratio < 1.35:
        return "Medium"
    if ratio < 1.60:
        return "High"
    return "Very High"

def get_traffic_color(traffic):
    return {
        "Low": "#22c55e",
        "Medium": "#eab308",
        "High": "#f97316",
        "Very High": "#ef4444",
    }.get(traffic, "#22c55e")

def calculate_risk(prediction):
    if prediction <= 30:
        return "LOW", "ON TIME"
    if prediction <= 45:
        return "MEDIUM", "AT RISK"
    return "HIGH", "DELAYED"

# ---------------- MAP ----------------
def create_delivery_map(route_coordinates, restaurant_name, customer_name, traffic="Low"):
    start_point = route_coordinates[0]
    end_point = route_coordinates[-1]
    traffic_color = get_traffic_color(traffic)

    route_map = folium.Map(
        location=start_point, zoom_start=13,
        control_scale=True, tiles="CartoDB dark_matter",
        prefer_canvas=True, zoom_control=True
    )

    # One-time CSS for the pulsing rider marker.
    route_map.get_root().html.add_child(folium.Element("""
        <style>
        @keyframes pulseRing {
            0%   { transform: scale(0.7); opacity: 0.7; }
            100% { transform: scale(2.4); opacity: 0; }
        }
        .rider-pulse {
            position: absolute; top: 50%; left: 50%;
            width: 34px; height: 34px; border-radius: 50%;
            background: rgba(255, 196, 0, 0.55);
            transform: translate(-50%, -50%);
            animation: pulseRing 1.8s ease-out infinite;
        }
        .rider-emoji {
            position: relative; font-size: 26px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));
        }
        </style>
    """))

    # Soft glow beneath the route for a more polished look.
    folium.PolyLine(
        locations=route_coordinates, color=traffic_color,
        weight=14, opacity=0.15
    ).add_to(route_map)

    # Animated, direction-aware route line colored by live traffic.
    AntPath(
        locations=route_coordinates, color=traffic_color,
        weight=5, opacity=0.95, delay=800,
        dash_array=[12, 18], pulse_color="#ffffff",
        tooltip=f"Delivery Route · {traffic} Traffic"
    ).add_to(route_map)

    def badge_icon(emoji, border_color):
        html = f"""
        <div style="
            background:#1a1f2b; width:38px; height:38px; border-radius:50%;
            border:3px solid {border_color}; display:flex; align-items:center;
            justify-content:center; font-size:18px;
            box-shadow:0 4px 14px {border_color}66;
        ">{emoji}</div>
        """
        return folium.DivIcon(html=html, icon_size=(38, 38), icon_anchor=(19, 19))

    folium.Marker(
        location=start_point, tooltip=f"Restaurant · {restaurant_name}",
        popup=restaurant_name, icon=badge_icon("🍴", "#ff4b4b")
    ).add_to(route_map)

    folium.Marker(
        location=end_point, tooltip=f"Customer · {customer_name}",
        popup=customer_name, icon=badge_icon("📍", "#3b82f6")
    ).add_to(route_map)

    rider_position = route_coordinates[len(route_coordinates) // 2]
    folium.Marker(
        location=rider_position, tooltip="Delivery Partner - En Route",
        icon=folium.DivIcon(html="""
            <div style="position:relative; width:34px; height:34px;">
                <div class="rider-pulse"></div>
                <div class="rider-emoji">🛵</div>
            </div>
        """, icon_size=(34, 34), icon_anchor=(17, 17))
    ).add_to(route_map)

    Fullscreen(position="topright", title="Expand map", title_cancel="Exit fullscreen").add_to(route_map)

    route_map.fit_bounds(route_coordinates, padding=(40, 40))
    st_folium(route_map, height=440, width="100%", returned_objects=[])

# ---------------- STYLING ----------------
def apply_dashboard_style():
    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 15% 0%, rgba(255,75,75,0.08), transparent 25%), #090c12;
    }
    .block-container { max-width: 1300px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { font-size: 2.8rem !important; font-weight: 800 !important; letter-spacing: -1.5px; }
    h2, h3 { letter-spacing: -0.5px; }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(18, 22, 30, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 18px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(24,29,39,0.95), rgba(15,19,27,0.95));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #9ca3af; }
    div[data-testid="stMetricValue"] { font-weight: 700; }

    .section-caption { color: #8b93a1; font-size: 14px; margin-top: -8px; margin-bottom: 14px; }

    .eta-card {
        background: linear-gradient(135deg, rgba(255,75,75,0.20), rgba(255,75,75,0.05));
        border: 1px solid rgba(255, 75, 75, 0.40);
        border-radius: 16px;
        padding: 24px 28px;
        margin-top: 20px;
    }
    .eta-label { color: #ff7b7b; font-size: 13px; font-weight: 700; letter-spacing: 0.8px; margin-bottom: 10px; }
    .eta-value { font-size: 52px; font-weight: 800; line-height: 1; margin-bottom: 14px; }
    .status-chip {
        display: inline-block;
        background: rgba(255, 75, 75, 0.18);
        border: 1px solid rgba(255, 75, 75, 0.45);
        border-radius: 999px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 700;
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }

    iframe { border-radius: 16px; box-shadow: 0 12px 32px rgba(0,0,0,0.35); }
    header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- APP SETUP ----------------
apply_dashboard_style()
init_db()

try:
    model, model_info = load_data()
    google_key = st.secrets["GOOGLE_MAPS_API_KEY"]
    weather_key = st.secrets["OPENWEATHER_API_KEY"]
except FileNotFoundError:
    st.error("model.pkl or model_info.json is missing.")
    st.stop()
except KeyError:
    st.error("Google Maps or OpenWeather API key is missing.")
    st.stop()

if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<h1>Delivery Intelligence <span style="color:#ff4b4b;">Dashboard</span> 🍔</h1>
<p style="color:#8b93a1;font-size:16px;margin-top:-12px;">
Traffic-aware routing, live weather and ML-powered ETA prediction.
</p>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.2], gap="medium")

# ---------------- LEFT COLUMN: INPUT FORM ----------------
with left_col:
    with st.container(border=True):
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📦 Delivery Order")

            restaurant_query = st.text_input(
                "Restaurant Location", placeholder="Type at least 3 characters...",
                key="restaurant_query"
            )
            restaurant_options = {
                item["name"]: item["place_id"]
                for item in get_place_suggestions(restaurant_query, google_key)
            }
            origin_place_id = None
            restaurant_name = None
            if restaurant_options:
                restaurant_name = st.selectbox(
                    "Select Restaurant", list(restaurant_options.keys()), key="restaurant_selection"
                )
                origin_place_id = restaurant_options[restaurant_name]
            elif len(restaurant_query.strip()) >= 3:
                st.caption("No restaurant suggestions found.")

            customer_query = st.text_input(
                "Customer Location", placeholder="Type at least 3 characters...",
                key="customer_query"
            )
            customer_options = {
                item["name"]: item["place_id"]
                for item in get_place_suggestions(customer_query, google_key)
            }
            destination_place_id = None
            customer_name = None
            if customer_options:
                customer_name = st.selectbox(
                    "Select Delivery Location", list(customer_options.keys()), key="customer_selection"
                )
                destination_place_id = customer_options[customer_name]
            elif len(customer_query.strip()) >= 3:
                st.caption("No delivery location suggestions found.")

            prep_time = st.number_input("Preparation Time (min)", min_value=0, max_value=120, value=20)

        with c2:
            st.subheader("🛵 Delivery Partner")
            vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car"])
            experience = st.number_input("Courier Experience (yrs)", min_value=0, max_value=20, value=2)
            time_of_day = get_time_of_day()
            st.text_input("Current Time Context", value=time_of_day, disabled=True)

        submit_button = st.button("🚀 Predict Delivery ETA", type="primary", use_container_width=True)

    if submit_button:
        if not origin_place_id or not destination_place_id:
            st.warning("Select a restaurant and customer location from suggestions.")
        else:
            try:
                with st.spinner("Analyzing live delivery conditions..."):
                    latitude, longitude = get_place_coordinates(destination_place_id, google_key)
                    weather = get_weather(latitude, longitude, weather_key)
                    distance, live_time, static_time, route_coordinates = get_route_data(
                        origin_place_id, destination_place_id, google_key
                    )
                    traffic = calculate_traffic(live_time, static_time)

                    # Dict-of-lists (not list-of-dicts) avoids a pandas/pyarrow
                    # segfault that can occur building DataFrames from records.
                    input_df = pd.DataFrame({
                        "Distance_km": [distance],
                        "Weather": [weather],
                        "Traffic_Level": [traffic],
                        "Time_of_Day": [time_of_day],
                        "Vehicle_Type": [vehicle],
                        "Preparation_Time_min": [prep_time],
                        "Courier_Experience_yrs": [experience]
                    })

                    prediction = float(model.predict(input_df)[0])
                    risk, status = calculate_risk(prediction)

                    save_prediction(
                        distance, weather, traffic, time_of_day, vehicle,
                        prep_time, experience, prediction
                    )

                    st.session_state.result = {
                        "restaurant_name": restaurant_name,
                        "customer_name": customer_name,
                        "distance": distance,
                        "live_time": live_time,
                        "weather": weather,
                        "traffic": traffic,
                        "route_coordinates": route_coordinates,
                        "prediction": prediction,
                        "risk": risk,
                        "status": status,
                    }

            except requests.Timeout:
                st.error("Google Maps or weather service timed out.")
            except requests.HTTPError as error:
                status_code = error.response.status_code if error.response else "Unknown"
                st.error(f"External API request failed. HTTP status: {status_code}")
            except Exception as error:
                st.error(f"Prediction failed: {type(error).__name__}: {error}")

    if st.session_state.result:
        r = st.session_state.result
        st.markdown(f"""
        <div class="eta-card">
            <div class="eta-label">🎯 ML PREDICTED DELIVERY ETA</div>
            <div class="eta-value">{r['prediction']:.1f}<span style="font-size:20px;color:#aeb4bf;"> min</span></div>
            <div class="status-chip">{r['risk']} RISK · {r['status']}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- RIGHT COLUMN: LIVE INTELLIGENCE ----------------
with right_col:
    with st.container(border=True):
        if st.session_state.result:
            r = st.session_state.result

            st.markdown("### 🗺️ Traffic-Aware Delivery Route")
            st.markdown(f"""
            <div class="section-caption">
                🍴 {r['restaurant_name']} &nbsp;&nbsp;→&nbsp;&nbsp; 📍 {r['customer_name']}
            </div>
            """, unsafe_allow_html=True)
            create_delivery_map(r["route_coordinates"], r["restaurant_name"], r["customer_name"], r["traffic"])

            st.markdown("### 🛰️ Live Delivery Intelligence")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📍 Route Distance", f"{r['distance']:.1f} km")
            m2.metric("⏱️ Live Travel", f"{r['live_time']:.1f} min")
            m3.metric("🚦 Traffic", r["traffic"])
            m4.metric("🌦️ Weather", r["weather"])

            with st.expander("📊 Prediction Analytics"):
                history = get_recent_predictions()
                if history.empty:
                    st.caption("No prediction history yet.")
                else:
                    st.line_chart(history.set_index("timestamp")["predicted_time"])
                    st.dataframe(history, use_container_width=True, hide_index=True)
        else:
            st.info("Fill in the delivery details and click **Predict Delivery ETA** to see the live route and intelligence here.")