"""University-level student grade prediction project (synthetic data)."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "students.csv"
OUT = ROOT / "reports"
FEATURES = ["study_hours", "attendance_percent", "previous_grade", "assignments_completed", "sleep_hours"]

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    print("Shape:", df.shape)
    print("Missing values:\n", df.isna().sum())
    print("Summary:\n", df.describe().round(2))
    df = df.dropna(subset=FEATURES + ["final_grade"])
    X, y = df[FEATURES], df["final_grade"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1),
    }
    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        rows.append({"model": name, "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                     "R2": r2_score(y_test, pred)})
    results = pd.DataFrame(rows).sort_values("MAE")
    results.to_csv(OUT / "model_comparison.csv", index=False)
    print("\nModel comparison:\n", results.to_string(index=False))
    best_name = results.iloc[0]["model"]
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions[best_name], alpha=0.65)
    plt.plot([0, 100], [0, 100], linestyle="--", color="black")
    plt.xlabel("Actual final grade")
    plt.ylabel("Predicted final grade")
    plt.title(f"Actual vs Predicted - {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "actual_vs_predicted.png", dpi=150)
    plt.close()
    df[FEATURES + ["final_grade"]].hist(figsize=(11, 7), bins=18)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "feature_distributions.png", dpi=150)
    plt.close()
    print("\nSaved results to reports/.")

if __name__ == "__main__":
    main()
