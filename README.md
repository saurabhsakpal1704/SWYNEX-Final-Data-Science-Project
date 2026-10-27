# SWYNEX Final Data Science Project
## E-Commerce Purchase Intent Prediction with Explainable AI

A reproducible portfolio project that predicts whether an online shopping session will end in a purchase. Includes data acquisition, exploratory analysis, preprocessing, six-model comparison, evaluation, FastAPI, Streamlit dashboard, tests, Docker, and GitHub Actions.

> **Data note:** the app downloads the public UCI Online Shoppers Purchasing Intention dataset on first training run. If network access is unavailable, `src/data.py` can create a clearly labelled synthetic demo dataset so the pipeline can still be tested. Do not report synthetic-demo metrics as real-world results.

## Quick start (Windows PowerShell, VS Code)
1. Install **Python 3.11** and **Git**. During Python setup, enable **Add Python to PATH**.
2. Extract this ZIP to a folder such as `C:\Users\YourName\Documents\SWYNEX-Final-Data-Science-Project`.
3. In VS Code, select **File → Open Folder** and open that extracted folder.
4. Open **Terminal → New Terminal**, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/train.py
```

If PowerShell blocks activation, run this once in that terminal and activate again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is unavailable, install Python 3.11 or try `python -m venv .venv`.

## Start the app
Open **two VS Code terminals** and activate `.venv` in each.

Terminal 1 — API:
```powershell
uvicorn src.api:app --reload
```
Open http://127.0.0.1:8000/docs to try the API.

Terminal 2 — dashboard:
```powershell
streamlit run dashboard/app.py
```
Streamlit will show a local URL, usually http://localhost:8501.

## What training produces
- `data/raw/online_shoppers.csv` — downloaded source dataset, if available.
- `artifacts/model.joblib` — fitted preprocessing + model pipeline.
- `artifacts/metrics.json` — test-set metrics for each successfully trained model.
- `artifacts/model_card.md` — summary and limitations.
- `artifacts/feature_importance.csv` — tree-based model feature importance when supported.

The first run needs internet to retrieve the UCI dataset. The synthetic fallback is for checking that the code runs, not for final evidence or claimed performance.

## Models compared
Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, and Gradient Boosting (models that fail due to an environment-specific issue are recorded and skipped). The selection rule uses PR-AUC where available, appropriate for an imbalanced target. A threshold is chosen from validation predictions using F1 as a transparent baseline; adapt it to business costs before deployment.

## GitHub: create and push repository
Create an empty public repository on GitHub named **SWYNEX-Final-Data-Science-Project**. Do not add a README/license there because this folder already has files. Then run in the project terminal:

```powershell
git init
git add .
git commit -m "Build SWYNEX e-commerce purchase intent project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/SWYNEX-Final-Data-Science-Project.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. If Git asks you to sign in, complete the browser authentication. Do not share your password or token with anyone.

If `origin` already exists, check it with `git remote -v`; update it using `git remote set-url origin https://github.com/YOUR-USERNAME/SWYNEX-Final-Data-Science-Project.git` rather than adding a second remote.

## Test before submitting
```powershell
pytest -q
```
Run training first so the API and dashboard can load the model. For a clean test, clone your GitHub repo into another folder, install requirements, train, run tests, and open both services.

## SWYNEX Task 4 submission
1. Paste your **GitHub repository URL** into “Task submission link”.
2. Create and publish a LinkedIn post describing what you actually completed. Mention **@SWYNEX Technologies**, select the official company page if available, and include a screenshot or short demo. Add relevant hashtags such as `#SWYNEX #Internship #DataScience`.
3. Paste the public LinkedIn post URL into “LinkedIn explanation post link”.
4. Open both links in a private/incognito window to confirm they are accessible, then click **Submit task**.

Do not submit a made-up LinkedIn URL or claim a training result until you have run the project. Edit the report with your actual metrics and screenshots.

## Important limitations
This is an educational decision-support project, not a production-ready system. The UCI dataset is historical; performance can change on newer traffic. `PageValues` can be unavailable at the moment an early-session prediction is needed and may introduce target leakage for some use cases. Review the feature list and intended scoring time before claiming deployment readiness. Explanations from feature importance are not causal claims.
