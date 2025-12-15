# FuzzySegment Pro

**Intelligent Customer Profiling using Fuzzy C-Means Clustering and Granular Computing**

> _Where Every Customer Belongs Everywhere_

---

## 🎯 Problem Statement

Traditional K-Means clustering forces customers into single, rigid categories, missing the multi-dimensional nature of customer behavior. A customer who purchases tech (25%), fashion (45%), and beauty (20%) products gets labeled as just a "Fashion Shopper" (100%), losing 55% of their behavioral complexity.

**Business Impact:**

- Lost cross-sell opportunities
- Poor targeting accuracy
- Missed revenue potential

---

## 💡 Solution

**FuzzySegment Pro** uses **Fuzzy C-Means clustering** to assign membership degrees across multiple segments simultaneously, capturing the full spectrum of customer interests.

### Key Innovation

- **Soft Clustering**: Customers belong to multiple segments with membership scores (0-1)
- **Granular Computing**: Multi-level data abstraction for hierarchical analysis
- **Comprehensive Metrics**: 5 fuzzy-specific validation indices (PC, MPC, PE, XBI, FSI)

---

## 🏗️ Project Structure

```
project/
├── src/                          # Core modules
│   ├── fuzzy_clustering.py       # Fuzzy C-Means wrapper
│   ├── feature_engineering.py    # RFM + category affinity extraction
│   ├── metrics.py                # Fuzzy validation metrics
│   └── data_pipeline.py          # Data loading & preprocessing
├── streamlit_app/
│   └── app.py                    # Interactive web dashboard
├── notebooks/
│   ├── preprocess_data.py        # Transaction → customer features
│   ├── 01_eda.py                 # Exploratory data analysis
│   └── 02_fuzzy_vs_kmeans.py     # Comparative analysis
├── data/
│   ├── train.csv                 # Raw transaction data
│   └── preprocessed_customers.csv # Customer-level features
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Create conda environment
conda create -n fuzzysegment python=3.10 -y
conda activate fuzzysegment

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess Data

```powershell
cd notebooks
python preprocess_data.py
```

### 3. Run Streamlit App

```powershell
cd ..
streamlit run streamlit_app/app.py
```

Visit: **http://localhost:8501**

---

## 📊 Features

### Data Pipeline

- ✅ RFM (Recency, Frequency, Monetary) analysis
- ✅ Category affinity percentages (Furniture, Office Supplies, Technology)
- ✅ Normalized features (0-1 scale)

### Clustering Engine

- ✅ Fuzzy C-Means implementation with `scikit-fuzzy`
- ✅ Configurable parameters (n_clusters, fuzzifier)
- ✅ Hard label conversion for comparison

### Validation Metrics

| Metric  | Range       | Better | Description                                      |
| ------- | ----------- | ------ | ------------------------------------------------ |
| **PC**  | [1/c, 1]    | Higher | Partition Coefficient - measures crispness       |
| **MPC** | [0, 1]      | Higher | Modified PC - normalized version                 |
| **PE**  | [0, log(c)] | Lower  | Partition Entropy - measures disorder            |
| **XBI** | [0, ∞)      | Lower  | Xie-Beni Index - compactness/separation ratio    |
| **FSI** | [-1, 1]     | Higher | Fuzzy Silhouette - fuzzy extension of silhouette |

### Interactive Dashboard

- ✅ CSV upload or use default dataset
- ✅ Feature selection
- ✅ Real-time clustering with parameter tuning
- ✅ Membership heatmap visualization
- ✅ Side-by-side FCM vs K-Means comparison

---

## 📈 Results

**Dataset:** 793 customers from Superstore transaction data

**Fuzzy C-Means Performance:**

- PC: ~0.65 (moderate fuzziness)
- XBI: ~0.15 (good compactness)
- FSI: ~0.40 (reasonable separation)
- **Multi-dimensional customers detected:** 20-40% of dataset

**vs K-Means:**

- K-Means: 100% membership (all customers)
- FCM: Variable memberships revealing overlapping interests
- **Result:** FCM captures customer complexity that K-Means misses

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-fuzzy** — Fuzzy C-Means implementation
- **scikit-learn** — K-Means baseline, metrics, preprocessing
- **pandas & numpy** — Data manipulation
- **matplotlib & seaborn** — Visualization
- **streamlit** — Interactive web dashboard

---

## 📚 Research Foundation

Based on peer-reviewed research:

1. **Sivaguru & Punniyamoorthy (2020)** — Dynamic Fuzzy C-Means for customer segmentation
2. **Sivaguru (2023)** — Granular Computing + Fuzzy validation metrics
3. **Yuliari et al. (2015)** — Fuzzy RFM methodology
4. **Kuo et al. (2023)** — GA-based FCM optimization

---

## 🎯 Business Applications

1. **Personalized Marketing**: Target customers based on multi-segment interests
2. **Cross-sell Optimization**: Recommend products from secondary/tertiary segments
3. **Dynamic Pricing**: Price based on segment affinity scores
4. **Retention Strategy**: Identify customers shifting between segments

**ROI (from literature):**

- E-commerce: +158% revenue
- Banking: +2,504% ROI

---

## 📝 License

MIT License

---

## 👥 Contributors

**FuzzySegment Pro Team**  
GitHub: [BV-Tech-Team/FuzzySegment-Pro](https://github.com/BV-Tech-Team/FuzzySegment-Pro)

---

## 🔗 Links

- **GitHub Repository**: https://github.com/BV-Tech-Team/FuzzySegment-Pro
- **Documentation**: See `.learn/` folder (development only, not in production)

---

_Built with ❤️ using Fuzzy C-Means and Granular Computing_
