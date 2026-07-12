import json
import pickle
import re

import pandas as pd
import streamlit as st

FEATURE_LABELS = {
    "Distance_km": "Route Distance",
    "Preparation_Time_min": "Food Preparation",
    "Courier_Experience_yrs": "Courier Experience",
    "Weather": "Weather",
    "Traffic_Level": "Traffic Level",
    "Time_of_Day": "Time of Day",
    "Vehicle_Type": "Vehicle Type",
}


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("model_info.json", "r") as file:
        model_info = json.load(file)
    return model, model_info


def predict_delivery_time(model, input_data):
    input_df = pd.DataFrame([input_data])
    return float(model.predict(input_df)[0])


def _parse_feature_name(feature_name):
    if feature_name.startswith("numeric__"):
        return feature_name.replace("numeric__", ""), None

    match = re.match(r"categorical__(.+?)_(.+)", feature_name)
    if match:
        return match.group(1), match.group(2)
    return feature_name, None


def explain_prediction(model, input_data):
    input_df = pd.DataFrame([input_data])
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    coefficients = regressor.coef_
    intercept = float(regressor.intercept_)

    grouped = {}
    for index, raw_name in enumerate(feature_names):
        base_feature, category = _parse_feature_name(raw_name)
        contribution = float(transformed[0, index] * coefficients[index])
        if abs(contribution) < 1e-9:
            continue

        label = FEATURE_LABELS.get(base_feature, base_feature)
        if category:
            label = f"{label} ({category})"

        grouped[label] = grouped.get(label, 0.0) + contribution

    contributions = [
        {
            "feature": feature,
            "contribution": round(value, 2),
            "direction": "increases" if value >= 0 else "reduces",
        }
        for feature, value in grouped.items()
    ]
    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    chart_df = pd.DataFrame(
        {
            "Feature": [item["feature"] for item in contributions],
            "Minutes": [item["contribution"] for item in contributions],
        }
    )

    return {
        "intercept": round(intercept, 2),
        "top_factors": contributions[:3],
        "chart_data": chart_df,
    }
