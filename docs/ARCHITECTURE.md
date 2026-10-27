# Architecture

```text
UCI Online Shoppers CSV
        |
        v
src.data: validate / clean
        |
        v
src.modeling: preprocessing pipeline + six classifiers
        |                         |
        v                         v
artifacts/model.joblib     metrics / model card / importance
        |
        v
FastAPI /predict  <----  Streamlit dashboard
        |
        v
Purchase probability + threshold-based prediction
```

Preprocessing is inside the fitted scikit-learn Pipeline so the same transformations are applied during training and inference. The API loads the artifact lazily and returns HTTP 503 until training has produced it.
