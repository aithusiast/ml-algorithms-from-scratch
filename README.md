# 🧠 ML Algorithms From Scratch

> Building machine learning algorithms from the ground up — no scikit-learn, no black boxes, just math and NumPy.

![Python](https://img.shields.io/badge/Python-3.14%2B-blue?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4%2B-013243?logo=numpy&logoColor=white)

---

## 📌 Overview

This repository contains clean, well-documented implementations of core machine learning algorithms built **from scratch using Python and NumPy**. The goal is to build deep intuition for how these algorithms work under the hood — math, gradients, and all.

Each algorithm lives in its own folder with a dedicated **Python File** covering the derivation, implementation, and a **Jupyter Notebook** covering visual exploration and model evaluation.

---

## 🗂️ Algorithms Implemented

| Algorithm | Folder |
|---|---|
| Linear Regression | [`linear_regression/`](./linear_regression) |
| Logistic Regression | [`logistic_regression/`](./logistic_regression) |

> More algorithms coming soon — see the roadmap below.

---

## 📂 Repository Structure

```
ml-algorithms-from-scratch/
│
├── linear_regression/
│   └── algorithm.py
│   └── training.ipynb
│
├── logistic_regression/
│   └── algorithm.py
│   └── training.ipynb
│
...
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🚀 Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management.

### Prerequisites

- Python 3.14+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed

### Installation

```bash
# Clone the repo
git clone https://github.com/aithusiast/ml-algorithms-from-scratch.git
cd ml-algorithms-from-scratch

# Install dependencies
uv sync
```
---

## 📦 Dependencies

Managed via `pyproject.toml`:

| Package | Purpose |
|---|---|
| `numpy` | Core math and array operations |
| `matplotlib` | Plots and visualizations |
| `seaborn` | Statistical plots |
| `pandas` | Data loading and manipulation |
| `ipykernel` | Jupyter notebook kernel |

---

## 🗺️ Roadmap

Algorithms planned for implementation:

- [ ] Decision Tree (CART)
- [ ] Random Forest
- [ ] K-Nearest Neighbors (KNN)
- [ ] Support Vector Machine (SVM)
- [ ] K-Means Clustering
- [ ] Principal Component Analysis (PCA)
- [ ] Naive Bayes
- [ ] Neural Network / Backpropagation

---

## 💡 Design Philosophy

- **Clarity over cleverness** — code is written to be read and understood
- **Math-first** — each notebook derives the algorithm from first principles before a single line of code
- **No ML shortcuts** — only NumPy for the core implementations; no scikit-learn under the hood
- **Visual** — every notebook includes plots to build geometric intuition

---

## 🤝 Contributing

Found a bug or want to add an algorithm? Contributions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/algorithm-name`
3. Add your notebook and implementation
4. Open a pull request

---