from locust import HttpUser, task, between

class DashboardUser(HttpUser):
    # Simulates a user waiting 1-5 seconds between actions
    wait_time = between(1, 5)
    host = "https://heart-disease-dashboard.gentlemoss-1b6edc24.eastus.azurecontainerapps.io"

    @task(3)
    def load_homepage(self):
        """Simulate loading the dashboard homepage."""
        self.client.get("/", name="Homepage")

    @task(2)
    def health_check(self):
        """Hit the Streamlit health endpoint."""
        self.client.get("/_stcore/health", name="Health Check")

    @task(1)
    def load_static_assets(self):
        """Simulate loading Streamlit static assets."""
        self.client.get("/static", name="Static Assets")
