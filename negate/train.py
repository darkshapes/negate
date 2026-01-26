import asyncio
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from negate import ResidualExtractor


def train_model(
    model_name,
    X,
    y,
):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=3)
    try:
        model.fit(X_train, y_train)
    except ValueError as exc:
        print(f"Model not fitted due to error: {exc}")
        return

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(model.feature_importances_)

    joblib.dump(model, model_name)
    print(f"Model saved to {model_name}")


def main():
    """Entry point for training model with residual data."""

    residual_extractor = ResidualExtractor(input=Path("assets/synthetic_v2"), verbose=True)

    async def async_main() -> pd.DataFrame:
        """
        Asynchronously processes residual data.
        """
        residuals = await residual_extractor.process_residuals()
        return residuals

    residuals = asyncio.run(async_main())

    synthetic_fractal = pd.DataFrame(residuals["fractal_complexity"])
    synthetic_texture = pd.DataFrame(residuals["texture_complexity"])

    residual_extractor = ResidualExtractor(input=Path("assets/real"), verbose=True)

    residuals = asyncio.run(async_main())
    human_origin_fractal = pd.DataFrame(residuals["fractal_complexity"])
    human_origin_texture = pd.DataFrame(residuals["texture_complexity"])
    X = pd.concat([human_origin_fractal, synthetic_fractal], axis=1)

    y = human_origin_fractal["fractal_complexity"].astype("category").cat.codes

    train_model("fractal_classifier.joblib", X, y)
    X = pd.concat([human_origin_texture, synthetic_texture], axis=1)

    y = human_origin_texture["texture_complexity"].astype("category").cat.codes

    train_model("texture_classifier.joblib", X, y)


if __name__ == "__main__":
    main()
