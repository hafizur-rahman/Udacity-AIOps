"""
Integration tests for the Sentiment Analysis API.
"""

import pytest
from fastapi.testclient import TestClient

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

# Test that POST /predict returns the correct response schema
# (text, sentiment, confidence, latency_ms) and a valid sentiment label.
# Hint: use @pytest.mark.parametrize to run the same assertions across multiple headlines.
@pytest.mark.parametrize("headline", [
    "Stocks rally after positive earnings report",
    "Market crashes due to global uncertainty",
    "Investors remain cautious amid mixed signals"
])
def test_predict_returns_valid_response(client: TestClient, headline):
    response = client.post("/predict", json={"text": headline})
    print(f"Response for headline '{headline}': {response.json()}")
    assert response.status_code == 200

    data = response.json()

    # Schema validation
    assert set(data.keys()) == {"text", "sentiment", "confidence", "latency_ms"}

    assert isinstance(data["text"], str)
    assert isinstance(data["sentiment"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["latency_ms"], float)

    # Sentiment label must be valid
    assert data["sentiment"] in VALID_SENTIMENTS

# Test that POST /predict/batch returns a list of results
# whose length matches the number of input texts, and each item has the correct schema.
def test_predict_batch(client: TestClient):
    texts = [
        "Tech stocks rise as AI demand grows",
        "Oil prices fall amid oversupply concerns",
        "Investors brace for inflation data"
    ]

    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200

    results = response.json()

    # Length must match input
    assert len(results) == len(texts)

    for item in results:
        assert set(item.keys()) == {"text", "sentiment", "confidence", "latency_ms"}
        assert isinstance(item["text"], str)
        assert isinstance(item["sentiment"], str)
        assert isinstance(item["confidence"], float)
        assert isinstance(item["latency_ms"], float)
        assert item["sentiment"] in VALID_SENTIMENTS

# Test that POST /predict/batch returns HTTP 422 for an empty texts list.
def test_predict_batch_empty_list_returns_422(client: TestClient):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422