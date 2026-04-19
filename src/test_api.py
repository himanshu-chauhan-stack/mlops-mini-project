
# src/test_api.py
# automated tests for the FastAPI endpoint — used in CI/CD pipeline

import requests
import sys

BASE_URL = "http://localhost:8000"
PASS = 0
FAIL = 0

def check(name, condition, got):
    global PASS, FAIL
    if condition:
        print(f"  PASS: {name}")
        PASS += 1
    else:
        print(f"  FAIL: {name} | got: {got}")
        FAIL += 1

print("\n===== Running API Tests =====")

# test 1: root endpoint
r = requests.get(f"{BASE_URL}/")
check("GET / returns 200", r.status_code == 200, r.status_code)

# test 2: health check
r = requests.get(f"{BASE_URL}/health")
check("GET /health returns healthy", r.json().get("status") == "healthy", r.json())

# test 3: predict setosa
r = requests.post(f"{BASE_URL}/predict", json={
    "sepal_length": 5.1, "sepal_width": 3.5,
    "petal_length": 1.4, "petal_width": 0.2
})
check("POST /predict setosa", r.json().get("predicted_class") == "setosa", r.json())

# test 4: predict virginica
r = requests.post(f"{BASE_URL}/predict", json={
    "sepal_length": 6.7, "sepal_width": 3.1,
    "petal_length": 5.6, "petal_width": 2.4
})
check("POST /predict virginica", r.json().get("predicted_class") == "virginica", r.json())

# test 5: invalid input returns 400
r = requests.post(f"{BASE_URL}/predict", json={
    "sepal_length": -1.0, "sepal_width": 3.5,
    "petal_length": 1.4,  "petal_width": 0.2
})
check("POST /predict invalid returns 400", r.status_code == 400, r.status_code)

# test 6: metrics endpoint
r = requests.get(f"{BASE_URL}/metrics")
check("GET /metrics returns cpu_usage", "cpu_usage_percent" in r.json(), r.json())

print(f"\n===== Results: {PASS} passed | {FAIL} failed =====")

if FAIL > 0:
    sys.exit(1)   # fail the CI/CD pipeline if any test fails
