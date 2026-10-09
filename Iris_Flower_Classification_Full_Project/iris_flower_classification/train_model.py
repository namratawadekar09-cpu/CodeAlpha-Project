from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASET_PATH = Path(r"C:\Users\sdvad\Downloads\archive\Iris.csv")


def main():
    dataset = pd.read_csv(DATASET_PATH)

    feature_columns = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
    target_column = "Species"

    X = dataset[feature_columns].values
    y_labels = dataset[target_column].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    target_names = label_encoder.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=target_names
    ))

    model_folder = Path(__file__).resolve().parent / "model"
    model_folder.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "target_names": target_names,
        "feature_names": feature_columns
    }

    model_path = model_folder / "iris_model.joblib"
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
