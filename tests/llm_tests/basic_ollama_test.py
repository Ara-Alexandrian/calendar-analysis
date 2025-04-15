"""
Very basic Ollama connection test using only requests.
"""
import requests
import sys

# Set the Ollama server URL
OLLAMA_URL = "http://192.168.1.5:11434"

print(f"Testing Ollama server at {OLLAMA_URL}...")
try:
    # Simple GET request with longer timeout
    response = requests.get(f"{OLLAMA_URL}/", timeout=10)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("SUCCESS: Connected to Ollama server!")
        print(f"Response: {response.text[:200]}")
    else:
        print(f"WARNING: Server returned status code {response.status_code}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)

print("Test completed.")
