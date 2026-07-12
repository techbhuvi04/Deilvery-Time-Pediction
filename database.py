import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = "predictions.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
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


def save_prediction(
    distance,
    weather,
    traffic,
    time_of_day,
    vehicle,
    prep_time,
    experience,
    predicted_time,
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                timestamp, distance, weather, traffic, time_of_day,
                vehicle, prep_time, experience, predicted_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            distance,
            weather,
            traffic,
            time_of_day,
            vehicle,
            prep_time,
            experience,
            float(predicted_time),
        ))


def get_prediction_history(limit=20):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT timestamp, distance, weather, traffic, time_of_day,
                   vehicle, prep_time, experience, predicted_time
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
