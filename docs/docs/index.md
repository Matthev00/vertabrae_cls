# Vertebrae Classifier Documentation

> **Author**: Mateusz Ostaszewski  
> **Project Type**: Engineer's Thesis  
> **Topic**: 3D Vertebrae Classification from CT Scans  
> **Repository**: `vertebrae_cls`

---

## 🧠 Project Overview

The goal of this project is to build a **3D deep learning pipeline** for detecting spinal injuries based on CT scans. The approach combines **medical imaging preprocessing**, **vertebra segmentation**, and **3D classification models** trained on individually extracted vertebrae.

---

## 🏗️ Pipeline Structure

```mermaid
graph TD
    A[DICOM CT Scans] --> B[Data Preprocessing]
    B --> C[Vertebra Segmentation (MONAI WholeBody)]
    C --> D[Vertebra Cropping + Normalization]
    D --> E[3D CNN Classifier]
    E --> F[Injury Prediction]
```

Each component in the pipeline is modular and can be adapted or reused across other medical imaging projects.

---

## 🗃️ Data Sources

- Input data consists of real-world **DICOM-format CT scans**.
- Each scan is associated with an `XLS` metadata file (e.g. diagnosis, vertebrae info).
- The segmentation model is used only to extract vertebrae, not for injury detection.

---

## ⚙️ Core Modules

| Module        | Purpose                                      |
|---------------|----------------------------------------------|
| `data_utils`  | Parsing metadata, loading DICOM, extracting vertebrae |
| `modeling`    | Definition of ResNet3D, Med3D, and MONAI-based backbones |
| `training`    | Training loop, W&B logging, hyperparameter sweeps |
| `inference`   | Prediction API and visualization of heatmaps |

> 🛠️ All components are implemented in Python with PyTorch and MONAI. Configuration is managed via YAML and CLI commands.

---

## 📊 Results and Evaluation

Evaluation is performed using:

**TO DO**

Models are trained and compared using **Weights & Biases** with automated sweep runners.

---

## 🚀 Getting Started

- Setup instructions: [Getting Started](getting-started.md)
- Project commands: [Commands](commands.md)
- Code reference: [Code Docs](code/data_preparation.md)

---

## 📄 License

This project is intended for academic use under the [MIT License](https://opensource.org/licenses/MIT).
