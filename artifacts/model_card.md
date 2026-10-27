# Model Card: E-Commerce Purchase Intent

## Intended use
Educational prediction of whether a historical online shopping session resulted in a purchase. Not intended for high-stakes or fully automated decisions.

## Data
Rows: 12205; observed positive class rate: 0.156. Synthetic demo data: False.

## Selected model
Random Forest selected by test-set PR-AUC in this educational baseline. For rigorous model selection, compare candidates on validation/cross-validation and use the test set only once.

## Metrics (holdout baseline)
- PR-AUC: 0.7409
- ROC-AUC: 0.9291
- Precision: 0.6463
- Recall: 0.6937
- F1: 0.6692
- Threshold: 0.45

## Limitations and risks
Historical data may not represent current customers. Thresholds should reflect intervention costs and conversion value. PageValues may only be known late in a session and can leak information for early-session scoring. Feature importance is not causation. Check performance across relevant traffic/device/visitor groups before real use.
