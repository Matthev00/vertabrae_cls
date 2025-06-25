# 🛠️ Makefile Commands

This document provides a complete overview of available `make` commands in the project. These commands streamline key development and deployment tasks, from data preparation and training to running the app and building documentation.

---

## 📦 Dependency Management

### ✅ Install Python Dependencies

Installs all required Python packages defined in `requirements.txt`.

```bash
make requirements
```

### 🧹 Delete Compiled Python Files

Removes all compiled Python files (`*.pyc`, `*.pyo`) and `__pycache__` directories.

```bash
make clean
```

---

## 🧼 Code Quality Tools

### 🔍 Lint the Code

Runs `flake8`, `isort`, and `black` in check mode to validate code style.

```bash
make lint
```

### 🎨 Format the Code

Applies `black` formatting to the codebase.

```bash
make format
```

---

## 📂 Data Workflow

### 🧪 Prepare Interim Data

Extracts and cleans metadata from XLS into intermediate format.

```bash
make prepare_interim_data
```

### 📊 Create Dataset

Finalizes dataset by cropping and normalizing vertebrae volumes.

```bash
make create_dataset
```

---

## 🧠 Training & Experiments

### 🚀 Run Experiments

Launches model training and logs all results to Weights & Biases.

```bash
make run_experiments
```

> ⚠️ Training can be computationally expensive and time-consuming.  
> Progress is automatically tracked via Weights & Biases.  
> 📊 Dashboard: `[ W&B link here]`

---

## 🌐 API & App

### 🔌 Start API

Launches the FastAPI server to expose model predictions.

```bash
make start_api
```

> ℹ️ **Configuration**: Set the model path and inference device in `src/config.py` based on your environment (`cuda`, `cpu`, `mps`).

### 🖼️ Run App

Runs the local Streamlit app for prediction visualization.

```bash
make run_app
```

---

## 📚 Documentation

### 🧪 Serve Docs

Serves documentation locally for development preview.

```bash
make serve_docs
```

### 🚀 Deploy Docs

Publishes documentation to the configured hosting provider (e.g. GitHub Pages).

```bash
make deploy_docs
```

---

## 📜 Utilities

### 🆘 Display Help

Lists all available `make` commands and short descriptions.

```bash
make help
```
