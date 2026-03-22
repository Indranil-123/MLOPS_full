# MLOps Projects Repository

A collection of end-to-end MLOps projects demonstrating best practices for building, tracking, versioning, and deploying machine learning models using modern MLOps tooling.

---

## Repository Structure

```
.
├── DVC/                    # Data Version Control experiments and pipelines
├── MLOps_Project_32/       # MLOps project – sprint 32
├── MLOps_Project_33/       # MLOps project – sprint 33
├── MLOps_Project_34/       # MLOps project – sprint 34
├── dagshub/                # DagsHub integration and remote tracking configs
├── mlflow_test.ipynb       # MLflow experiment tracking notebook
├── mlflow.db               # Local MLflow tracking database
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **MLflow** | Experiment tracking, model registry, artifact logging |
| **DVC** | Data versioning, pipeline management, remote storage |
| **DagsHub** | Remote MLflow & DVC server, collaboration |
| **Python** | Core language |
| **Jupyter** | Experimentation and prototyping |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Indranil-123/neel.git
cd neel

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## MLflow Experiment Tracking

MLflow is used for logging parameters, metrics, and model artifacts across all projects.

### Running the MLflow UI locally

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### Using DagsHub as a Remote MLflow Server

```python
import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"]      = "https://dagshub.com/<username>/<repo>.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "<your_dagshub_username>"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<your_dagshub_token>"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
```

> **Note:** Never commit credentials to the repository. Use environment variables or a `.env` file (already listed in `.gitignore`).

---

## DVC — Data & Pipeline Versioning

DVC tracks large datasets and model files without storing them in Git.

### Common DVC Commands

```bash
# Pull tracked data from remote
dvc pull

# Run the full pipeline
dvc repro

# Push data changes to remote
dvc push

# Check pipeline DAG
dvc dag
```

---

## Projects Overview

### MLOps_Project_32
> _Brief description of what this project covers — e.g., training a classification model with MLflow logging and DVC-tracked datasets._

### MLOps_Project_33
> _Brief description — e.g., building a model pipeline with DVC stages and remote artifact storage on DagsHub._

### MLOps_Project_34
> _Brief description — e.g., model evaluation, registration in MLflow Model Registry, and serving with MLflow._

### DVC/
Contains standalone DVC pipeline experiments, including `dvc.yaml` pipeline definitions and `params.yaml` configuration files.

### dagshub/
Configuration and scripts for syncing experiments and data with [DagsHub](https://dagshub.com).

---

## Workflow

```
Data (DVC) ──► Feature Engineering ──► Model Training
                                              │
                                         MLflow Logging
                                         (params, metrics, artifacts)
                                              │
                                       Model Registry (MLflow / DagsHub)
                                              │
                                         Deployment / Serving
```

---

## Environment Variables

Create a `.env` file in the root directory (already in `.gitignore`):

```env
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your_username>
MLFLOW_TRACKING_PASSWORD=<your_token>
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.

---

## Author

**Indranil** — [@Indranil-123](https://github.com/Indranil-123)