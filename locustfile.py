from locust import HttpUser, task, between
import random


class FeedbackLensUser(HttpUser):
    wait_time = between(1, 5) 
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
            try:
                if response.status_code == 200:
                    data = response.json()

                   
                    if not data.get("top_issues"):
                        response.failure("Empty top_issues in response")
                else:
                    response.failure(f"Failed: {response.status_code}")

            except Exception as e:
                response.failure(f"Exception: {str(e)}")

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
            try:
                if response.status_code == 200:
                    data = response.json()

                    if not data.get("top_issues"):
                        response.failure("Empty top_issues in response")
                else:
                    response.failure(f"Failed: {response.status_code}")

            except Exception as e:
                response.failure(f"Exception: {str(e)}")

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
            try:
                if response.status_code == 200:
                    data = response.json()

                    if not data.get("top_issues"):
                        response.failure("Empty top_issues in response")
                else:
                    response.failure(f"Failed: {response.status_code}")

            except Exception as e:
                response.failure(f"Exception: {str(e)}")

    @task(1)
    def batch_test(self):
        with self.client.post(
            "/batch",
            name="Batch Analysis",
            json={
                "company": "swiggy",
                "reviews": [
                    "Late delivery",
                    "Food arrived cold",
                    "Delivery agent was rude"
                ]
            },
            catch_response=True,
            timeout=30
        ) as response:
            try:
                if response.status_code == 200:
                    data = response.json()

                    if not data.get("recommendations"):
                        response.failure("Empty recommendations in batch response")
                else:
                    response.failure(f"Failed: {response.status_code}")

            except Exception as e:
                response.failure(f"Exception: {str(e)}")

    @task(1)
    def health_check(self):
        try:
            response = self.client.get("/health", name="Health Check")
            if response.status_code != 200:
                response.failure("Health check failed")
        except Exception as e:
            print(f"Health check exception: {e}")