from locust import HttpUser, task, between
import random


class FeedbackLensUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://a030bc4002eaf422693f6d978ac39103-1191230641.us-east-1.elb.amazonaws.com:8000"

    swiggy_queries = [
        "Analyze Swiggy delivery issues",
        "Swiggy late delivery complaints",
        "Swiggy customer problems"
    ]

    uber_queries = [
        "What are Uber pricing problems?",
        "Uber surge pricing issues",
        "Uber complaints summary"
    ]

    zomato_queries = [
        "Analyze Zomato customer support issues",
        "Zomato refund problems",
        "Zomato delivery complaints"
    ]

    @task(3)
    def analyze_swiggy(self):
        query = random.choice(self.swiggy_queries)
        with self.client.post(
            "/analyze",
            name="Swiggy Analysis",
            json={"query": query},
            catch_response=True,
            timeout=30
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed: {response.status_code}")

    @task(2)
    def analyze_uber(self):
        query = random.choice(self.uber_queries)
        with self.client.post(
            "/analyze",
            name="Uber Analysis",
            json={"query": query},
            catch_response=True,
            timeout=30
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed: {response.status_code}")

    @task(1)
    def analyze_zomato(self):
        query = random.choice(self.zomato_queries)
        with self.client.post(
            "/analyze",
            name="Zomato Analysis",
            json={"query": query},
            catch_response=True,
            timeout=30
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed: {response.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="Health Check")