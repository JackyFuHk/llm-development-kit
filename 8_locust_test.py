from locust import HttpUser, task, between

class ApiUser(HttpUser):
    wait_time = between(0.5, 1.5)   # 用户间隔

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={"text": "你好 Locust"}
        )