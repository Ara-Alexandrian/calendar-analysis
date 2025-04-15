# c:\\GitHub\\calendar-analysis\\test_llm_connections.py
import os
import sys
import requests
import time

# Add project root to sys.path to allow importing config
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from config import settings
    print("Successfully imported settings.")
except ImportError as e:
    print(f"Error importing settings: {e}")
    print("Please ensure you run this script from the project root directory.")
    sys.exit(1)

# Try importing ollama library
try:
    import ollama
    OLLAMA_LIB_AVAILABLE = True
    print(f"Ollama library found and imported.")
except ImportError:
    OLLAMA_LIB_AVAILABLE = False
    print("Ollama library not found. Will use basic HTTP check.")

def test_ollama_connection():
    """Tests connection to the Ollama server."""
    print(f"--- Testing Ollama Connection ({settings.OLLAMA_BASE_URL}) ---")
    start_time = time.time()
    try:
        # Prefer using the library's list method if available
        if OLLAMA_LIB_AVAILABLE:
            print("Attempting connection using ollama library...")
            client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            # Use list() as a basic connectivity check
            models = client.list()
            print(f"SUCCESS: Connected via Ollama library.")
            print(f"Available models: {[m['name'] for m in models['models']]}")
            if settings.LLM_MODEL not in [m['name'] for m in models['models']]:
                print(f"WARNING: Default model '{settings.LLM_MODEL}' not found in available models.")
        else:
            # Fallback to basic HTTP check
            print("Attempting connection using basic HTTP GET...")
            # Use the root endpoint check
            check_url = settings.OLLAMA_BASE_URL.strip('/') + '/'
            response = requests.get(check_url, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            print(f"SUCCESS: Connected via HTTP GET. Status Code: {response.status_code}")
            # Optionally check response content if needed
            # print(f"Response Text (first 100 chars): {response.text[:100]}")

    except requests.exceptions.ConnectionError as e:
        print(f"FAILED: Connection Error. Could not connect to {settings.OLLAMA_BASE_URL}.")
        print(f"  Error details: {e}")
    except requests.exceptions.Timeout:
        print(f"FAILED: Connection Timeout. Request to {settings.OLLAMA_BASE_URL} timed out.")
    except requests.exceptions.RequestException as e:
        print(f"FAILED: Request Exception. An error occurred.")
        print(f"  URL: {e.request.url if e.request else 'N/A'}")
        print(f"  Error details: {e}")
        if e.response is not None:
            print(f"  Status Code: {e.response.status_code}")
            print(f"  Response Text: {e.response.text[:200]}...") # Show beginning of response
    except Exception as e:
        # Catch potential ollama library errors or other issues
        print(f"FAILED: An unexpected error occurred.")
        print(f"  Error details: {e}")
    finally:
        end_time = time.time()
        print(f"Ollama test duration: {end_time - start_time:.2f} seconds")
    print("-" * 40)


def test_mcp_connection():
    """Tests connection to the MCP server."""
    print(f"--- Testing MCP Connection ({settings.MCP_SERVER_URL}) ---")
    start_time = time.time()
    try:
        print("Attempting connection using basic HTTP GET to root...")
        # Check the root endpoint first, which should return {"message": "MCP Server is running"}
        check_url = settings.MCP_SERVER_URL.strip('/') + '/'
        response = requests.get(check_url, timeout=10)
        response.raise_for_status()
        print(f"SUCCESS: Connected to MCP root. Status Code: {response.status_code}")
        try:
            print(f"Response JSON: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            print(f"Response Text (not JSON): {response.text[:200]}...")

        # Optionally, try the /v1/chat endpoint (expects POST, but GET might give info)
        # print("\nAttempting GET request to /v1/chat (expects POST)...")
        # chat_url = settings.MCP_SERVER_URL.strip('/') + '/v1/chat'
        # try:
        #     chat_response = requests.get(chat_url, timeout=5)
        #     print(f"GET /v1/chat Status Code: {chat_response.status_code}")
        #     print(f"GET /v1/chat Response: {chat_response.text[:200]}...")
        # except Exception as chat_e:
        #     print(f"Error during GET /v1/chat: {chat_e}")


    except requests.exceptions.ConnectionError as e:
        print(f"FAILED: Connection Error. Could not connect to {settings.MCP_SERVER_URL}.")
        print(f"  Error details: {e}")
    except requests.exceptions.Timeout:
        print(f"FAILED: Connection Timeout. Request to {settings.MCP_SERVER_URL} timed out.")
    except requests.exceptions.RequestException as e:
        print(f"FAILED: Request Exception. An error occurred.")
        print(f"  URL: {e.request.url if e.request else 'N/A'}")
        print(f"  Error details: {e}")
        if e.response is not None:
            print(f"  Status Code: {e.response.status_code}")
            print(f"  Response Text: {e.response.text[:200]}...")
    except Exception as e:
        print(f"FAILED: An unexpected error occurred.")
        print(f"  Error details: {e}")
    finally:
        end_time = time.time()
        print(f"MCP test duration: {end_time - start_time:.2f} seconds")
    print("-" * 40)

if __name__ == "__main__":
    print("Starting LLM Connection Tests...")
    test_ollama_connection()
    print("\n") # Add a newline for spacing
    test_mcp_connection()
    print("Connection tests finished.")
