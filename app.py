
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# Load model only once
model = None

# -------------------------------
# PROMETHEUS METRICS
# -------------------------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the ML model is loaded"
)


# -------------------------------
# MODEL LOADING
# -------------------------------

def load_model():
    global model

    if model is None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "student_result_model.pkl was not found."
                )

            model = joblib.load(MODEL_PATH)

            MODEL_LOADED.set(1)

            logger.info(
                "event=model_loaded model_file=%s",
                MODEL_PATH
            )

        except Exception:
            MODEL_LOADED.set(0)

            logger.exception(
                "event=model_load_failed model_file=%s",
                MODEL_PATH
            )

            raise

    return model


# -------------------------------
# REQUEST MONITORING
# -------------------------------

@app.before_request
def before_request():
    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):

    duration_seconds = (
        time.perf_counter() - g.start_time
    )

    duration_ms = duration_seconds * 1000

    # Update Prometheus metrics
    # Exclude /metrics to avoid monitoring itself

    if request.path != "/metrics":

        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration_seconds)

    # Logging

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    response.headers["X-Request-ID"] = g.request_id

    return response


# -------------------------------
# HEALTH ENDPOINTS
# -------------------------------

@app.get("/")
def health_check():

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.get("/health")
def health():

    return jsonify({
        "status": "healthy"
    })


# -------------------------------
# PROMETHEUS METRICS ENDPOINT
# -------------------------------

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


# -------------------------------
# PREDICTION ENDPOINT
# -------------------------------

@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not data:

        logger.warning(
            "event=invalid_request reason=missing_json"
        )

        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:

        logger.warning(
            "event=validation_error missing_fields=%s",
            missing_fields
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:

        logger.info(
            "event=prediction_request feature_names=%s",
            sorted(FEATURES)
        )

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        trained_model = load_model()

        prediction_code = int(
            trained_model.predict(sample)[0]
        )

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        # Successful prediction metric
        PREDICTIONS.labels(
            result="success"
        ).inc()

        logger.info(
            "event=prediction_success prediction=%s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:

        # Failed prediction metric
        PREDICTIONS.labels(
            result="error"
        ).inc()

        logger.exception(
            "event=prediction_failed"
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# -------------------------------
# APPLICATION START
# -------------------------------

if __name__ == "__main__":

    logger.info(
        "event=application_starting"
    )

    app.run(
        host="0.0.0.0",
        port=5000
    )
