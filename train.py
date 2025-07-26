from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import os

def main():
    # Load California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    print("Hello")
    print(X.shape)
    print(y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    score = r2_score(y_test, y_pred)
    print(f"R² Score: {score:.4f}")

    # Create output directory if not exists
    os.makedirs("C:\\Users\\rsoni\\PycharmProjects\\Assignment3_MLOps\\models", exist_ok=True)

    # Save the model and scaler
    joblib.dump(model, "models/lr_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("Model and scaler saved to 'models/'")

if __name__ == "__main__":
    main()