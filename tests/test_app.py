import unittest
from app import create_app
from router import predict
from train import train

class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics = train()
        cls.client = create_app(cls.model).test_client()

    def test_probability_distribution(self):
        result = predict(self.model, "My invoice has an extra charge")
        self.assertEqual(result["category"], "billing")
        self.assertAlmostEqual(sum(r["score"] for r in result["alternatives"]), 1, places=3)

    def test_api_validation(self):
        for payload in [{}, {"text": " "}, {"text": 42}, {"text": "x" * 5001}, []]:
            self.assertEqual(self.client.post("/api/predict", json=payload).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", data="bad", content_type="application/json").status_code, 400)

    def test_pages_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertTrue(self.client.get("/health").json["model_loaded"])

    def test_api_prediction(self):
        response = self.client.post("/api/predict", json={"text": "Please reset my forgotten password"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["category"], "account")

    def test_uncertain_input_requires_review(self):
        self.assertTrue(predict(self.model, "xyzzy qqq")['needs_review'])

if __name__ == "__main__":
    unittest.main()
