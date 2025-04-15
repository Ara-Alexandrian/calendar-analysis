"""
Simple Ollama connection test script.
This script directly tests the connection to the Ollama server 
without relying on the settings module.
"""
import requests
import time

# Directly specify the Ollama server URL
OLLAMA_URL = "http://192.168.1.5:11434"

def test_ollama_connection():
    """Test basic connectivity to the Ollama server."""
    print(f"Testing connection to Ollama server at: {OLLAMA_URL}")
    start_time = time.time()
    
    try:
        # Perform a simple HTTP request to check if the server is responding
        response = requests.get(f"{OLLAMA_URL.rstrip('/')}/", timeout=5)
        
        if response.status_code == 200:
            print(f"SUCCESS: Connected to Ollama server. Status code: {response.status_code}")
            # Print any response content if available
            if response.text:
                print(f"Response: {response.text[:200]}...")
        else:
            print(f"WARNING: Received unexpected status code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
    
    except requests.exceptions.ConnectionError as e:
        print(f"FAILED: Connection Error. Could not connect to {OLLAMA_URL}")
        print(f"  Error details: {e}")
    except requests.exceptions.Timeout:
        print(f"FAILED: Connection Timeout. Request to {OLLAMA_URL} timed out")
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
    
    finally:
        end_time = time.time()
        print(f"Test duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    print("Starting simple Ollama connection test...")
    test_ollama_connection()
    print("Test completed.")
