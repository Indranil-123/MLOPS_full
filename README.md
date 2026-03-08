# MLOPS_full
# 🚀 MLOps Full Project — Random Forest with MLflow Tracking

A complete MLOps pipeline using **Scikit-learn**, **MLflow**, and **RandomizedSearchCV** for training, hyperparameter tuning, experiment tracking, and model registration.

---

## 📁 Project Structure

```
MLOPS_full/
│
├── MLOps_Project_32/
│   └── notebook.ipynb          # Main Jupyter notebook
│
├── mlruns/                     # MLflow experiment metadata (auto-generated)
├── .gitignore
└── README.md
```

---

## ⚙️ Prerequisites

Make sure you have the following installed:

- Python 3.8+
- pip
- git

---

## 🛠️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Indranil-123/MLOPS_full.git
cd MLOPS_full
```

### 2. Create a Virtual Environment

```bash
python -m venv mlflowvenv
```

Activate it:

- **Windows:**
  ```bash
  mlflowvenv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source mlflowvenv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install mlflow scikit-learn pandas numpy matplotlib seaborn jupyter
```

---

## ▶️ Running the Project

### Step 1: Start the MLflow Tracking Server

Open a **separate terminal** and run:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

> Keep this terminal open while running the notebook.

### Step 2: Launch Jupyter Notebook

In your main terminal (with venv activated):

```bash
jupyter notebook
```

Open `MLOps_Project_32/notebook.ipynb` in the browser.

### Step 3: Run All Cells

Execute all cells in order. The notebook will:

1. Load and preprocess the dataset
2. Split data into train/test sets
3. Run `RandomizedSearchCV` for hyperparameter tuning
4. Log parameters, metrics, and the best model to MLflow
5. Register the model as `"Best Random Search Model"`

---

## 📊 Viewing MLflow Experiments

Once the notebook has run, open your browser and go to:

```
http://127.0.0.1:5000
```

You will see:
- All experiment runs
- Logged parameters (n_estimators, max_depth, etc.)
- Metrics (MSE)
- Registered models

---

## 🧠 Hyperparameters Tuned

| Parameter | Values Searched |
|---|---|
| `n_estimators` | [100, 200, 300, 400, 500] |
| `max_depth` | [4, 6, 8, 10, None] |
| `min_samples_split` | [2, 5, 10] |
| `min_samples_leaf` | [1, 2, 4] |
| `random_state` | [42] |

---

## 📌 Key Code Snippet

```python
# Set tracking URI before starting a run
mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():
    random_search = hyperparameter_tuning(X_train, y_train, params)
    best_model = random_search.best_estimator_

    mlflow.log_param("best_n_estimators", random_search.best_params_["n_estimators"])
    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(best_model, "model",
                             registered_model_name="Best Random Search Model",
                             signature=signature1)
```

---

## 🚫 .gitignore (Important!)

The following are excluded from version control to avoid large file errors:

```gitignore
mlruns/
mlartifacts/
mlflowvenv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
```

> ⚠️ Never push `mlartifacts/` to GitHub — MLflow model files can exceed GitHub's 100MB limit.

---

## 🐛 Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `PermissionError: [WinError 5]` | Space in folder path (`Edunet Foundation`) | Move project to a path without spaces e.g. `C:\MLOPS\` |
| `TypeError: parameter grid not iterable` | Passing single values in `params` dict | Use lists: `"n_estimators": [100, 200, 300]` |
| `remote rejected` on git push | MLflow artifact > 100MB pushed to GitHub | Add `mlartifacts/` to `.gitignore` and remove from tracking |
| `AttributeError: .Scheme` | Wrong case on urlparse attribute | Use `.scheme` (lowercase) |

---

## 📬 Author

**Indranil** — [GitHub](https://github.com/Indranil-123)

---

## 📄 License

This project is for educational purposes as part of an MLOps training program.