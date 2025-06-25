# 🚀 Getting Started

A step-by-step guide to setting up the project from scratch: installing dependencies, preparing data, and running training or inference.

---

## 🔧 Prerequisites

Before you begin, make sure you have the following installed:

- Python **3.13**
- `make` utility (Linux/macOS or via WSL on Windows)

---

## 📦 Setup Instructions

### 🧬 1. Clone the Repository

```bash
git clone <repository-url>
cd vertebrae_cls
```

### 🧪 2. Install Python Dependencies

```bash
make requirements
```

Installs all required packages from `pyproject.toml`.

---

### 📂 3. Prepare the Data

#### Step 1: Process XLS Metadata → Interim Data

```bash
make prepare_interim_data
```

#### Step 2: Finalize Dataset

```bash
make create_dataset
```

Generates the training-ready dataset by cropping and normalizing vertebrae volumes.

---

## 🧠 Running Experiments

To train the model and log results to Weights & Biases:

```bash
make run_experiments
```

> ⚠️ **Note**: Training may take a long time on full datasets.  
> All experiments are automatically logged to **Weights & Biases**.  
> 📊 View logs at:  
> `👉 [W&B Dashboard Link Here]`  
> `📄 [Experiment Summary Report Here]`

---

## 🌐 Starting the API

To run the prediction REST API server:

```bash
make start_api
```

> ⚙️ **Important**: Edit `src/config.py` to configure:
> - which device to use (`cpu`, `cuda`, `mps`, etc.).
>
> Adjust settings based on your system capabilities and available hardware.

---

## 🖼️ Launching the UI App

To start the web interface for inference and heatmap visualization:

```bash
make run_app
```

Runs the local dashboard interface where you can select patients, generate heatmaps, and inspect prediction results.

---
