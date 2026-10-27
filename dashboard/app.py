from pathlib import Path
import sys, json
import pandas as pd
import streamlit as st
import requests
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_data

st.set_page_config(page_title="SWYNEX Purchase Intent", page_icon="🛍️", layout="wide")
st.title("🛍️ E-Commerce Purchase Intent")
st.caption("SWYNEX Final Data Science Project · Machine learning evaluation and session scoring")
page = st.sidebar.radio("Navigate", ["Overview", "Explore data", "Model evaluation", "Predict a session", "About & limitations"])

@st.cache_data(show_spinner=False)
def get_data(): return load_data()

if page == "Overview":
    st.subheader("Project overview")
    st.write("Estimate whether an online shopping session may result in a purchase using browsing behaviour and session context.")
    try:
        df = get_data()
        c1,c2,c3 = st.columns(3)
        c1.metric("Sessions", f"{len(df):,}")
        c2.metric("Features", len(df.columns)-1)
        c3.metric("Purchase rate", f"{df['Revenue'].astype(int).mean():.1%}")
        if df.attrs.get("synthetic", False): st.warning("Synthetic demo data is active because the UCI dataset could not be downloaded. Do not use these metrics as real results.")
        st.bar_chart(df["Revenue"].astype(int).value_counts().rename(index={0:"No purchase",1:"Purchase"}))
    except Exception as e: st.error(str(e))
elif page == "Explore data":
    st.subheader("Dataset explorer")
    try:
        df=get_data()
        st.write("Rows and columns", df.shape)
        st.dataframe(df.head(100), use_container_width=True)
        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        if "Month" in df: st.subheader("Sessions by month"); st.bar_chart(df["Month"].value_counts())
    except Exception as e: st.error(str(e))
elif page == "Model evaluation":
    st.subheader("Holdout evaluation")
    p=Path("artifacts/metrics.json")
    if p.exists():
        metrics=json.loads(p.read_text())
        if metrics.get("synthetic_demo_data"): st.warning("Metrics below were produced from synthetic demo data, not the UCI dataset.")
        st.write("Selected model:", metrics["best_model"])
        st.dataframe(pd.DataFrame(metrics["all_models"]), use_container_width=True)
        st.bar_chart(pd.DataFrame(metrics["all_models"]).set_index("model")[["pr_auc","roc_auc","f1"]])
        st.caption("PR-AUC is emphasized because purchase sessions are usually the minority class. Compare precision and recall against business costs.")
    else: st.info("No model artifacts yet. Run `python scripts/train.py` in the project terminal first.")
elif page == "Predict a session":
    st.subheader("Score a hypothetical session")
    with st.form("predict_form"):
        a,b,c=st.columns(3)
        with a:
            admin=st.number_input("Administrative pages", min_value=0, value=1)
            info=st.number_input("Informational pages", min_value=0, value=0)
            product=st.number_input("Product-related pages", min_value=0, value=12)
            month=st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"], index=10)
            visitor=st.selectbox("Visitor type", ["Returning_Visitor","New_Visitor","Other"])
        with b:
            admin_d=st.number_input("Administrative duration", min_value=0.0, value=20.0)
            info_d=st.number_input("Informational duration", min_value=0.0, value=0.0)
            product_d=st.number_input("Product-related duration", min_value=0.0, value=300.0)
            bounce=st.slider("Bounce rate", 0.0, 1.0, .02)
            exit_rate=st.slider("Exit rate", 0.0, 1.0, .03)
        with c:
            pagevalues=st.number_input("PageValues", min_value=0.0, value=0.0)
            special=st.slider("Special day", 0.0, 1.0, 0.0)
            operating_systems=st.number_input("Operating systems code", min_value=1, value=2)
            browser=st.number_input("Browser code", min_value=1, value=2)
            region=st.number_input("Region code", min_value=1, value=1)
            traffic=st.number_input("Traffic type code", min_value=1, value=2)
            weekend=st.checkbox("Weekend", value=False)
        submitted=st.form_submit_button("Predict purchase intent", type="primary")
    if submitted:
        payload={"Administrative":admin,"Administrative_Duration":admin_d,"Informational":info,"Informational_Duration":info_d,"ProductRelated":product,"ProductRelated_Duration":product_d,"BounceRates":bounce,"ExitRates":exit_rate,"PageValues":pagevalues,"SpecialDay":special,"Month":month,"OperatingSystems":operating_systems ,"Browser":browser,"Region":region,"TrafficType":traffic,"VisitorType":visitor,"Weekend":weekend}
        try:
            response=requests.post(os.getenv("API_URL", "http://127.0.0.1:8000") + "/predict", json=payload, timeout=10)
            response.raise_for_status(); result=response.json()
            st.metric("Estimated purchase probability", f"{result['purchase_probability']:.1%}")
            st.success("Model predicts purchase" if result["predicted_purchase"] else "Model predicts no purchase at the selected threshold")
            st.caption(result["interpretation"])
        except Exception as e: st.error(f"Could not reach the API. Start it with `uvicorn src.api:app --reload`. Details: {e}")
elif page == "About & limitations":
    st.subheader("Methodology")
    st.markdown("""- **Business goal:** prioritize sessions that may convert.
- **Workflow:** data validation → preprocessing pipeline → model comparison → holdout evaluation → API/dashboard.
- **Primary metric:** PR-AUC, with precision, recall, F1 and ROC-AUC also reported.
- **Reproducibility:** fixed random seeds and serialized preprocessing/model pipeline.

### Limitations
- Historical sessions may not represent current user behaviour.
- `PageValues` may be unavailable early in a session and can create leakage for early predictions.
- A prediction is probabilistic, not a guarantee or causal explanation.
- Threshold selection should reflect campaign costs and conversion value.
- Check fairness, drift, privacy and security before production use.""")
