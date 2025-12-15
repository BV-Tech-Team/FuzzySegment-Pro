# FuzzySegment Pro

**Intelligent Customer Profiling System using Fuzzy C-Means Clustering**

---

## Repository Structure

```
├── project/           # Main implementation code
│   ├── src/          # Core Python modules
│   ├── streamlit_app/# Web dashboard
│   ├── notebooks/    # Analysis scripts
│   ├── data/         # Dataset files
│   └── README.md     # Project documentation
│
└── report/           # (Coming soon) Technical report and presentation
```

---

## Getting Started

Navigate to the [`project/`](./project) folder for full documentation and setup instructions.

```powershell
cd project
conda create -n fuzzysegment python=3.10 -y
conda activate fuzzysegment
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

---

## What is FuzzySegment Pro?

A customer segmentation tool that uses **Fuzzy C-Means clustering** to capture multi-dimensional customer behavior, going beyond traditional K-Means' single-category assignments.

**Key Features:**

- Soft clustering with membership degrees
- Multi-dimensional customer profiling
- Interactive Streamlit dashboard
- Comprehensive fuzzy validation metrics
- K-Means comparison analysis

---

## Results Preview

- **793 customers** analyzed from Superstore dataset
- **20-40%** identified as multi-dimensional (missed by K-Means)
- **5 fuzzy metrics** for cluster quality validation
- **Real-time visualization** of membership degrees

---

## Links

- **GitHub**: https://github.com/BV-Tech-Team/FuzzySegment-Pro
- **Full Documentation**: [`project/README.md`](./project/README.md)

---

## Team

**BV Tech Team**

---

_Built with Fuzzy C-Means & Granular Computing_
