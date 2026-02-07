import requests

def test_chatbot(query):
    url = "http://127.0.0.1:8000/api/v1/chat"
    payload = {"message": query}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"\nUser: {query}")
            print(f"Bot: {data['response']}")
            print(f"Sources used: {', '.join(data['sources'])}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    # Test queries for your different data sources
    test_chatbot("What amphibians are found in Chitwan?")
    test_chatbot("Tell me about the birds in the park.")
    test_chatbot("How much does a Jeep Safari cost for a SAARC visitor?")