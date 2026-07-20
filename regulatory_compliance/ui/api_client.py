import requests

BASE_URL = "http://127.0.0.1:8000"


def query_rag(question):

    response = requests.post(f"{BASE_URL}/api/v1/query", json={"question": question})

    response.raise_for_status()

    return response.json()


def upload_pdf(file):

    files = {"file": (file.name, file, "application/pdf")}

    response = requests.post(f"{BASE_URL}/upload/pdf", files=files)

    response.raise_for_status()

    return response.json()
