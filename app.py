import requests
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from api_services import (
    build_eta_breakdown,
    calculate_risk,
    calculate_traffic,
    create_delivery_map,
    get_place_coordinates,
    get_place_suggestions,
    get_route_data,
    get_time_of_day,
    get_weather,
)
from database import get_prediction_history, init_db, save_prediction
from model_utils import explain_prediction, load_model, predict_delivery_time

st.set_page_config(
    page_title="Delivery Intelligence Dashboard",
    page_icon="🍔",
    layout="wide",
)


def apply_dashboard_style():
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(255,75,75,0.08), transparent 25%),
            #090c12;
    }

    .block-container {
        max-width: 100%;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(18, 22, 30, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 18px;
        padding: 8px 12px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
    }

    h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px;
    }

    h2, h3 {
        letter-spacing: -0.5px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(24, 29, 39, 0.95),
            rgba(15, 19, 27, 0.95)
        );
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    .eta-card {
        background: linear-gradient(
            135deg,
            rgba(255, 75, 75, 0.20),
            rgba(255, 75, 75, 0.05)
        );
        border: 1px solid rgba(255, 75, 75, 0.40);
        border-radius: 16px;
        padding: 24px 28px;
        margin-top: 20px;
        margin-bottom: 16px;
    }

    .eta-label {
        color: #ff7b7b;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }

    .eta-value {
        font-size: 52px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 14px;
    }

    .status-chip {
        display: inline-block;
        background: rgba(255, 75, 75, 0.18);
        border: 1px solid rgba(255, 75, 75, 0.45);
        border-radius: 999px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 700;
    }

    .insight-card {
        background: rgba(18, 22, 30, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 18px 20px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .insight-title {
        color: #ff7b7b;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin-bottom: 12px;
    }

    .factor-item {
        color: #d1d5db;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .section-caption {
        color: #8b93a1;
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 14px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }

    iframe {
        border-radius: 16px;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    st.session_state.setdefault("prediction_result", None)


def run_prediction(
    origin_place_id,
    destination_place_id,
    restaurant_name,
    customer_name,
    prep_time,
    vehicle,
    experience,
    time_of_day,
    model,
    google_key,
    weather_key,
):
    latitude, longitude = get_place_coordinates(
        destination_place_id,
        google_key,
    )
    weather = get_weather(latitude, longitude, weather_key)
    distance, live_time, static_time, route_coordinates = get_route_data(
        origin_place_id,
        destination_place_id,
        google_key,
    )
    traffic = calculate_traffic(live_time, static_time)

    input_data = {
        "Distance_km": distance,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience,
    }

    prediction = predict_delivery_time(model, input_data)
    risk, status = calculate_risk(prediction)
    explanation = explain_prediction(model, input_data)
    eta_breakdown = build_eta_breakdown(prediction, live_time, prep_time)

    save_prediction(
        distance,
        weather,
        traffic,
        time_of_day,
        vehicle,
        prep_time,
        experience,
        prediction,
    )

    st.session_state.prediction_result = {
        "prediction": prediction,
        "risk": risk,
        "status": status,
        "distance": distance,
        "live_time": live_time,
        "traffic": traffic,
        "weather": weather,
        "prep_time": prep_time,
        "restaurant_name": restaurant_name,
        "customer_name": customer_name,
        "route_coordinates": route_coordinates,
        "explanation": explanation,
        "eta_breakdown": eta_breakdown,
    }


def render_eta_card(result):
    prediction = result["prediction"]
    risk = result["risk"]
    status = result["status"]
    st.markdown(
        f"""
        <div class="eta-card">
            <div class="eta-label">🚀 ML PREDICTED DELIVERY ETA</div>
            <div class="eta-value">
                {prediction:.1f}
                <span style="font-size:20px;color:#aeb4bf;">min</span>
            </div>
            <div class="status-chip">
                {risk} RISK - {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explainability(result):
    explanation = result["explanation"]
    top_factors = explanation["top_factors"]

    factor_lines = []
    for factor in top_factors:
        sign = "+" if factor["contribution"] >= 0 else ""
        factor_lines.append(
            f'<div class="factor-item">'
            f'• <b>{factor["feature"]}</b> {factor["direction"]} ETA by '
            f'<span style="color:#ff7b7b;">{sign}{factor["contribution"]:.1f} min</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">🔍 WHY THIS ETA?</div>
            {''.join(factor_lines)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eta_comparison(result):
    breakdown = result["eta_breakdown"]
    st.markdown("#### ⚖️ ETA Breakdown Comparison")
    c1, c2, c3 = st.columns(3)
    c1.metric("ML Predicted ETA", f"{breakdown['ml_eta']:.1f} min")
    c2.metric("Google Live Travel", f"{breakdown['live_travel']:.1f} min")
    c3.metric("Food Preparation", f"{breakdown['prep_time']:.1f} min")

    comparison_df = pd.DataFrame({
        "Component": [
            "Food Preparation",
            "Google Live Travel",
            "Rule-Based Total (Prep + Travel)",
            "ML Context Buffer (traffic, weather, etc.)",
            "ML Predicted Total",
        ],
        "Minutes": [
            breakdown["prep_time"],
            breakdown["live_travel"],
            breakdown["rule_based_eta"],
            breakdown["ml_context_buffer"],
            breakdown["ml_eta"],
        ],
    })
    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Rule-based ETA = preparation + live travel. "
        "ML adds context from traffic, weather, vehicle, and experience."
    )


def render_map_and_metrics(result):
    st.markdown("## 🗺️ Traffic-Aware Delivery Route")
    st.markdown(
        f"""
        <div class="section-caption">
            🍴 {result["restaurant_name"]}
            &nbsp;&nbsp; → &nbsp;&nbsp;
            📍 {result["customer_name"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    route_map = create_delivery_map(
        result["route_coordinates"],
        result["restaurant_name"],
        result["customer_name"],
    )
    if route_map:
        st_folium(
            route_map,
            height=400,
            width="100%",
            returned_objects=[],
        )

    st.markdown("## 🛰️ Live Delivery Intelligence")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📍 Route Distance", f"{result['distance']:.1f} km")
    m2.metric("⏱️ Live Travel", f"{result['live_time']:.1f} min")
    m3.metric("🚦 Traffic", result["traffic"])
    m4.metric("☁️ Weather", result["weather"])


def render_feature_chart(result):
    st.markdown("#### 📊 Feature Impact on ETA")
    chart_data = result["explanation"]["chart_data"].set_index("Feature")
    st.bar_chart(chart_data, height=280)
    st.caption(
        f"Model base time: {result['explanation']['intercept']:.1f} min. "
        "Bars show how each factor pushes ETA up or down."
    )


def render_prediction_analytics(model_info):
    with st.expander("📈 Prediction Analytics"):
        history = get_prediction_history()
        if history.empty:
            st.info("No prediction history available.")
            return

        average_eta = history["predicted_time"].mean()
        high_risk_percentage = (history["predicted_time"] > 45).mean() * 100
        model_metrics = model_info["metrics"][model_info["selected_model"]]

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Average ETA", f"{average_eta:.1f} min")
        a2.metric("Predictions", len(history))
        a3.metric("High Risk", f"{high_risk_percentage:.1f}%")
        a4.metric("Model MAE", f"{model_metrics['MAE']:.1f} min")

        st.dataframe(history, hide_index=True, use_container_width=True)


apply_dashboard_style()
init_db()
init_session_state()

try:
    model, model_info = load_model()
    google_key = st.secrets["GOOGLE_MAPS_API_KEY"]
    weather_key = st.secrets["OPENWEATHER_API_KEY"]
except FileNotFoundError:
    st.error("model.pkl or model_info.json is missing.")
    st.stop()
except KeyError:
    st.error("Google Maps or OpenWeather API key is missing.")
    st.stop()

time_of_day = get_time_of_day()
left_col, right_col = st.columns([2, 3], gap="large")

with left_col:
    st.markdown("# Delivery Intelligence Dashboard 🍔")
    st.markdown(
        '<p class="section-caption">'
        "Traffic-aware routing, live weather and ML-powered ETA prediction."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        order_col, partner_col = st.columns(2)

        with order_col:
            st.markdown("#### 📦 Delivery Order")
            restaurant_query = st.text_input(
                "Restaurant Location",
                placeholder="Search restaurant area...",
            )
            origin_suggestions = get_place_suggestions(restaurant_query, google_key)
            origin_map = {
                item["name"]: item["place_id"]
                for item in origin_suggestions
            }
            restaurant_name = st.selectbox(
                "Select Restaurant",
                list(origin_map.keys()) or ["Type 3+ characters above"],
                disabled=not origin_map,
            )
            customer_query = st.text_input(
                "Customer Location",
                placeholder="Search delivery area...",
            )
            destination_suggestions = get_place_suggestions(
                customer_query,
                google_key,
            )
            destination_map = {
                item["name"]: item["place_id"]
                for item in destination_suggestions
            }
            customer_name = st.selectbox(
                "Select Delivery Location",
                list(destination_map.keys()) or ["Type 3+ characters above"],
                disabled=not destination_map,
            )
            prep_time = st.number_input(
                "Preparation Time (min)",
                min_value=0,
                max_value=120,
                value=20,
            )

        with partner_col:
            st.markdown("#### 🛵 Delivery Partner")
            vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car"])
            experience = st.number_input(
                "Courier Experience (yrs)",
                min_value=0,
                max_value=20,
                value=2,
            )
            st.text_input(
                "Current Time Context",
                value=time_of_day,
                disabled=True,
            )

        submit_button = st.button(
            "🚀 Predict Delivery ETA",
            type="primary",
            use_container_width=True,
        )

origin_place_id = origin_map.get(restaurant_name)
destination_place_id = destination_map.get(customer_name)

if submit_button:
    if not origin_place_id or not destination_place_id:
        with left_col:
            st.warning(
                "Select a restaurant and customer location from suggestions."
            )
    else:
        try:
            with st.spinner("Analyzing live delivery conditions..."):
                run_prediction(
                    origin_place_id,
                    destination_place_id,
                    restaurant_name,
                    customer_name,
                    prep_time,
                    vehicle,
                    experience,
                    time_of_day,
                    model,
                    google_key,
                    weather_key,
                )
        except requests.Timeout:
            with left_col:
                st.error("Google Maps or weather service timed out.")
        except requests.HTTPError as error:
            status_code = (
                error.response.status_code if error.response else "Unknown"
            )
            with left_col:
                st.error(
                    f"External API request failed. HTTP status: {status_code}"
                )
        except Exception as error:
            with left_col:
                st.error(
                    f"Prediction failed: {type(error).__name__}: {error}"
                )

result = st.session_state.prediction_result

with right_col:
    if result:
        render_map_and_metrics(result)
        render_feature_chart(result)
    else:
        st.markdown(
            '<p class="section-caption">'
            "Run a prediction to view the live route, intelligence metrics, "
            "and feature impact chart."
            "</p>",
            unsafe_allow_html=True,
        )
    render_prediction_analytics(model_info)

if result:
    with left_col:
        render_eta_card(result)
        render_explainability(result)
        render_eta_comparison(result)
