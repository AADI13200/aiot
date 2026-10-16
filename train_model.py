"""Train and evaluate the smart street-light brightness model."""

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = "traffic_data.csv"
MODEL_PATH = "models/smart_street_light_model.pkl"
METRICS_PATH = "outputs/model_metrics.json"
CONFUSION_PATH = "outputs/confusion_matrix.csv"
FEATURES = [
    "Area_Type",
    "Hour",
    "Day_Type",
    "Vehicle_Count",
    "Pedestrian_Count",
    "Ambient_Lux",
    "Motion_Detected",
    "Traffic_Level",
]
CATEGORICAL_FEATURES = ["Area_Type", "Day_Type", "Traffic_Level"]
NUMERICAL_FEATURES = [
    "Hour",
    "Vehicle_Count",
    "Pedestrian_Count",
    "Ambient_Lux",
    "Motion_Detected",
]


def build_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", estimator)]
    )


def main() -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    missing = sorted(set(FEATURES + ["Brightness_Level"]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if data[FEATURES + ["Brightness_Level"]].isnull().any().any():
        raise ValueError("Dataset contains missing values in model columns")

    X = data[FEATURES]
    y = data["Brightness_Level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    evaluations = {}
    trained = {}
    predictions = {}
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cross_validator,
            scoring="balanced_accuracy",
        )
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        evaluations[name] = {
            "accuracy": accuracy_score(y_test, predicted),
            "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
            "macro_f1": f1_score(y_test, predicted, average="macro"),
            "cv_balanced_accuracy": cv_scores.mean(),
            "cv_balanced_accuracy_std": cv_scores.std(),
        }
        trained[name] = pipeline
        predictions[name] = predicted

    best_name = max(
        evaluations,
        key=lambda name: (
            evaluations[name]["cv_balanced_accuracy"],
            evaluations[name]["macro_f1"],
        ),
    )
    best_model = trained[best_name]
    best_prediction = predictions[best_name]
    labels = sorted(y.unique())
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    importances = best_model.named_steps["model"].feature_importances_
    feature_importance = [
        {
            "feature": str(name),
            "importance": float(value),
        }
        for name, value in sorted(
            zip(feature_names, importances),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]

    joblib.dump(best_model, MODEL_PATH)
    matrix = confusion_matrix(y_test, best_prediction, labels=labels)
    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(CONFUSION_PATH)

    metadata = {
        "model": best_name,
        "dataset_rows": len(data),
        "features": FEATURES,
        "classes": [int(label) for label in labels],
        "evaluations": evaluations,
        "feature_importance": feature_importance,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test, best_prediction, output_dict=True, zero_division=0
        ),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metadata, metrics_file, indent=2)

    print(f"Dataset rows: {len(data)}")
    print(f"Selected model: {best_name}")
    for name, metrics in evaluations.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.3f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}, "
            f"cv_balanced_accuracy={metrics['cv_balanced_accuracy']:.3f}"
        )
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
