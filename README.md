# FuzzySegment Pro

Intelligent Customer Profiling using Fuzzy C-Means clustering and Granular Computing.

## Quick start

1. Create conda environment (Python 3.10+ recommended).

```powershell
conda create -n fuzzysegment python=3.10 -y
conda activate fuzzysegment
pip install -r requirements.txt
```

2. Preprocess data (generates customer features from transactions)

```powershell
cd notebooks
python preprocess_data.py
cd ..
```

3. Run the Streamlit app

```powershell
streamlit run streamlit_app/app.py
```

Project layout

- `src/` — core modules (data pipeline, clustering wrappers)
- `data/` — sample and preprocessed datasets
- `streamlit_app/` — interactive demo
- `notebooks/` — learning notebooks and experiments
- `.learn/` — guided lessons to learn FuzzySegment Pro

Next steps: implement models, tune parameters, and run examples.
