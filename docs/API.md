# API usage

Start training first: `python scripts/train.py`.

Start API: `uvicorn src.api:app --reload`.

- `GET /` service information
- `GET /health` basic health check
- `POST /predict` validated session features → purchase probability and predicted label
- `GET /docs` interactive Swagger UI

Example JSON body:
```json
{
  "Administrative": 1,
  "Administrative_Duration": 20,
  "Informational": 0,
  "Informational_Duration": 0,
  "ProductRelated": 12,
  "ProductRelated_Duration": 300,
  "BounceRates": 0.02,
  "ExitRates": 0.03,
  "PageValues": 0,
  "SpecialDay": 0,
  "Month": "Nov",
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 2,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```
