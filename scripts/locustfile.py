"""
Load test for the Sentiment Analysis API.

Run: locust -f locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure and start the test.
"""

import random

from locust import HttpUser, between, task

SAMPLE_HEADLINES = [
    "The company reported record profits and raised its dividend.",
    "The firm filed for bankruptcy after massive losses.",
    "Stocks closed flat on Friday amid low trading volume.",
    "Federal Reserve signals interest rate cuts later this year.",
    "Tech giant announces major layoffs amid revenue decline.",
    "Merger talks between the two firms collapsed overnight.",
    "Quarterly earnings beat analyst expectations by wide margin.",
    "Oil prices surge on supply concerns from the Middle East.",
]


class SentimentAPIUser(HttpUser):
    # Simulate realistic user pacing (0.5–2 seconds between requests)
    wait_time = between(0.5, 2.0)

    @task(3)
    def predict_single(self):
        """POST a randomly chosen headline to /predict"""
        headline = random.choice(SAMPLE_HEADLINES)
        self.client.post("/predict", json={"text": headline})

    @task(1)
    def predict_batch(self):
        """POST a batch of 4 random headlines to /predict/batch"""
        batch = random.sample(SAMPLE_HEADLINES, 4)
        self.client.post("/predict/batch", json={"texts": batch})

    @task(1)
    def health_check(self):
        """GET /health"""
        self.client.get("/health")
