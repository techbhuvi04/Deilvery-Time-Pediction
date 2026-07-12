import json
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = "finaldata.csv"
TARGET = "Delivery_Time_min"
ID_COLUMN = "Order_ID"


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero = y_true != 0

    return np.mean(
        np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
    ) * 100


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "MAPE (%)": safe_mape(y_test, predictions),
        "R2 Score": r2_score(y_test, predictions)
    }


# Load and clean dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna(how="all").dropna(axis=1, how="all")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])

# Features and target
X = df.drop(columns=[TARGET, ID_COLUMN])
y = df[TARGET]

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    col for col in X.columns if col not in numeric_features
]

# Save cleaned dataset for inspection during training
cleaned_df = X.copy()
cleaned_df[TARGET] = y

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    )
}

results = {}
trained_models = {}

# Train and evaluate models
for name, algorithm in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    results[name] = evaluate_model(
        pipeline, X_test, y_test
    )

    trained_models[name] = pipeline

# Model comparison
metrics_df = pd.DataFrame(results).T

print("\nModel Performance")
print(metrics_df.round(3))

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in trained_models.items():
    mae_scores = -cross_val_score(
        model, X, y, cv=cv,
        scoring="neg_mean_absolute_error"
    )

    r2_scores = cross_val_score(
        model, X, y, cv=cv,
        scoring="r2"
    )

    cv_results[name] = {
        "CV MAE Mean": mae_scores.mean(),
        "CV MAE Std": mae_scores.std(),
        "CV R2 Mean": r2_scores.mean(),
        "CV R2 Std": r2_scores.std()
    }

cv_df = pd.DataFrame(cv_results).T

print("\nCross Validation")
print(cv_df.round(3))

# Select model with lowest MAE
best_model_name = metrics_df["MAE"].idxmin()
best_model = trained_models[best_model_name]

print("\nSelected Model:", best_model_name)

# Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save model information
model_info = {
    "selected_model": best_model_name,
    "target_column": TARGET,
    "feature_columns": list(X.columns),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "metrics": {
        name: {
            metric: float(value)
            for metric, value in values.items()
        }
        for name, values in results.items()
    },
    "cross_validation": {
        name: {
            metric: float(value)
            for metric, value in values.items()
        }
        for name, values in cv_results.items()
    }
}

with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print("\nTraining completed successfully.")