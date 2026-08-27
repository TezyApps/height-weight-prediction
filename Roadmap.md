# Roadmap

## Next Learning

- **Polynomial Regression** — extend linear regression to curved relationships (`y = a + bx + cx² + ...`).
- **Regularization (Ridge / Lasso / Elastic Net)** — control overfitting via L1/L2 penalties.
- **Decision Trees** — non-linear, rule-based models for both regression and classification.
- **Random Forests / Gradient Boosting (e.g. XGBoost)** — ensemble methods that build on decision trees.
- **K-Nearest Neighbors (KNN)** — instance-based classification/regression.
- **Support Vector Machines (SVM)** — margin-based classification, kernel trick for non-linear boundaries.
- **Naive Bayes** — probabilistic classifier, useful baseline for classification tasks.
- **Model evaluation & tuning** — cross-validation, grid/random search, precision/recall/F1, ROC-AUC.
- **Feature engineering & selection** — scaling, encoding, handling multicollinearity.
- **Unsupervised learning (stretch)** — clustering (K-Means), dimensionality reduction (PCA) as a bridge topic.

## Host the models and consume by REST API

- Wrap the existing `.pkl` models (`height_prediction.pkl`, `gender_classification.pkl`) behind a **FastAPI** app.
- Define request/response schemas with **Pydantic** (e.g. `weight_kg` in, `height_cm` out; `height_cm`/`weight_kg` in, `gender` out).
- Load models once at startup (not per-request) for performance.
- Add endpoints:
  - `POST /predict/height` — predicts height from weight.
  - `POST /predict/gender` — predicts gender from height & weight.
  - `GET /health` — basic health check.
- Run locally with `uvicorn` for development.
- Add input validation and error handling (e.g. reject negative/out-of-range values).
- Document the API automatically via FastAPI's built-in OpenAPI/Swagger UI (`/docs`).
- (Stretch) Containerize with Docker for a reproducible local deployment.

## Consume it via FastAPI in clients (iOS Native app / React web app)

- **iOS Native app (Swift/SwiftUI)**
  - Use `URLSession` (or a lightweight networking layer) to call the FastAPI endpoints.
  - Decode JSON responses with `Codable` structs matching the API schemas.
  - Build a simple form UI to input weight/height and display predictions.
- **React web app**
  - Call the API with `fetch`/`axios` from a form component.
  - Handle loading and error states around the network calls.
  - Display predicted height/gender results in the UI.
- **Cross-cutting concerns**
  - CORS configuration on the FastAPI side to allow the web client's origin.
  - Consistent error response format so both clients can handle failures uniformly.
  - (Stretch) API versioning and a shared OpenAPI client/type generation step for both clients.
