# Student Performance Prediction

A university-level regression project predicting a student's **final grade (0–100)** from study hours, attendance, previous grade, assignments completed, and sleep hours.

## Tools
Python, Pandas, Matplotlib, Scikit-learn, Jupyter Notebook.

## Dataset
`data/students.csv` contains **500 synthetic records**, generated for demonstration. **Not real student data**. The relationships are simulated; results should not be presented as real-world evidence.

## Workflow
1. Load and inspect CSV data; check missing values and descriptive statistics.
2. Explore feature distributions.
3. Split data 80% training / 20% testing (random_state=42).
4. Train Linear Regression, Decision Tree Regressor, Random Forest Regressor.
5. Compare MAE, RMSE, R² on the held-out test set.
6. Export a model comparison table and charts.

## Run on Windows (VS Code / PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m jupyter notebook
```
Open `notebooks/student_performance.ipynb` in Jupyter and run the cells in order.

## Output
- `reports/model_comparison.csv`
- `reports/figures/actual_vs_predicted.png`
- `reports/figures/feature_distributions.png`

## Limitations
Synthetic training/test data share the same generation process. Results do not generalize to actual schools without a suitable consented dataset, validation, and checks for bias. Do not use this model to make educational decisions about students.

## GitHub
Commit source, sample dataset, README, and notebooks. Do not commit `.venv` or personal student records.
