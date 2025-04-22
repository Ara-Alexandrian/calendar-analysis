"""
Ollama Test Script - Simple test to show detailed console output from Ollama

This script calls Ollama directly and prints the full responses with formatting,
helping debug why output isn't appearing correctly in the main application.
"""
import os
import sys
import time
import json
from pprint import pprint
import logging

# Configure logging to show all details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("ollama_test")

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Try to import settings from project
    from config import settings
    MODEL = getattr(settings, 'LLM_MODEL', 'mistral')
    OLLAMA_URL = getattr(settings, 'OLLAMA_BASE_URL', 'http://192.168.1.5:11434')
except ImportError:
    # Fallback settings if unable to import
    MODEL = "mistral"
    OLLAMA_URL = "http://192.168.1.5:11434"

def print_separator(text=""):
    """Print a separator line with optional text"""
    width = 80
    if text:
        print(f"\n{'=' * ((width - len(text) - 2) // 2)} {text} {'=' * ((width - len(text) - 2) // 2)}\n")
    else:
        print("\n" + "=" * width + "\n")

def test_ollama_direct():
    """Test Ollama using direct HTTP requests to see raw responses"""
    try:
        print_separator("TESTING OLLAMA WITH DIRECT HTTP REQUESTS")
        import httpx
        
        # Build request
        url = f"{OLLAMA_URL}/api/chat"
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the color of the sky?"}
            ],
            "stream": False,
            "format": "json"
        }
        
        print(f"Sending request to {url}")
        print(f"Request payload:")
        pprint(payload)
        
        # Make request with detailed logging
        print("\nMaking HTTP request...")
        start_time = time.time()
        
        # Set stream=False to get the entire response at once
        response = httpx.post(
            url, 
            json=payload, 
            timeout=30.0
        )
        
        elapsed_time = time.time() - start_time
        print(f"\nRequest completed in {elapsed_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        # Print headers for debugging
        print("\nResponse headers:")
        for key, value in response.headers.items():
            print(f"{key}: {value}")
        
        # Print raw response content
        print("\nRaw response content:")
        print(response.content.decode('utf-8')[:500])
        
        # Try to parse as JSON
        print("\nParsed response:")
        try:
            data = response.json()
            pprint(data)
            
            # Extract the content specifically
            if 'message' in data and 'content' in data['message']:
                print("\nExtracted content:")
                print(data['message']['content'])
            else:
                print("\nCouldn't find message.content in the response")
                pprint(data)
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
        
        return response
        
    except Exception as e:
        print(f"Error testing Ollama with direct HTTP: {e}")
        return None

def test_with_ollama_module():
    """Test using the ollama Python module if installed"""
    print_separator("TESTING WITH OLLAMA PYTHON MODULE")
    
    try:
        import ollama
        print("Ollama Python module is installed")
        
        print(f"\nSending request to model: {MODEL}")
        print("Prompt: What is the color of the sky?")
        
        start_time = time.time()
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the color of the sky?"}
            ],
            format="json"
        )
        elapsed_time = time.time() - start_time
        
        print(f"\nRequest completed in {elapsed_time:.2f} seconds")
        print("\nFull response object:")
        pprint(response)
        
        if 'message' in response and 'content' in response['message']:
            print("\nExtracted content:")
            print(response['message']['content'])
        
    except ImportError:
        print("Ollama Python module is not installed.")
        print("Install it with: pip install ollama")
    except Exception as e:
        print(f"Error testing with Ollama Python module: {e}")

def test_with_project_client():
    """Test using the project's own LLM client implementation"""
    print_separator("TESTING WITH PROJECT LLM CLIENT")
    
    try:
        # Import from the project's functions
        from src.infrastructure.llm.factory import LLMClientFactory # Updated import
        # from functions.llm_extraction.client import is_llm_ready # Old import

        # Get the client first
        client = LLMClientFactory.get_client() # Updated usage
        if not client:
            print("Failed to get LLM client.")
            return

        # Check if LLM is ready using the client instance
        ready = client.is_available() # Updated usage
        print(f"LLM ready: {ready}")

        if not ready:
            print("LLM is not ready. Cannot test with project client.")
            return
            
        print(f"Client obtained. Testing with model: {MODEL}")
        print("Prompt: What is the color of the sky?")
        
        start_time = time.time()
        response = client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the color of the sky?"}
            ],
            format="json"
        )
        elapsed_time = time.time() - start_time
        
        print(f"\nRequest completed in {elapsed_time:.2f} seconds")
        print("\nFull response object:")
        pprint(response)
        
        # Extract and print the message content specifically
        if 'message' in response and 'content' in response['message']:
            print("\nExtracted content:")
            print(response['message']['content'])
        
    except Exception as e:
        print(f"Error testing with project client: {e}")

def implement_ideal_ollama_call():
    """Implement an ideal way to call Ollama with full output visibility"""
    print_separator("IDEAL IMPLEMENTATION")
    
    try:
        import httpx
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        # Build request
        url = f"{OLLAMA_URL}/api/chat"
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the color of the sky?"}
            ],
            "stream": False,
            "format": "json"
        }
        
        console.print(Panel.fit(f"Sending request to Ollama ({MODEL})", style="blue"))
        console.print("Prompt: What is the color of the sky?")
        
        with console.status("[bold green]Waiting for Ollama response..."):
            start_time = time.time()
            
            response = httpx.post(url, json=payload, timeout=30.0)
            
            elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            console.print(f"[green]Request completed in {elapsed_time:.2f} seconds[/green]")
            
            data = response.json()
            
            if 'message' in data and 'content' in data['message']:
                content = data['message']['content']
                console.print(Panel.fit(f"[bold]Ollama Response:[/bold]\n\n{content}", 
                                       title="Response", border_style="green"))
            else:
                console.print("[red]Could not parse response content[/red]")
                console.print(data)
                
            return content
        else:
            console.print(f"[red]Error: Received status code {response.status_code}[/red]")
            console.print(response.text)
            return None
            
    except Exception as e:
        print(f"Error in ideal implementation: {e}")
        return None

def fix_for_extractor():
    """Provide a function that can be used to fix the extractor"""
    print_separator("EXTRACTOR FIX IMPLEMENTATION")
    
    try:
        import httpx
        
        # Build request
        url = f"{OLLAMA_URL}/api/chat"
        prompt = "What is the color of the sky?"
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "format": "json" 
        }
        
        # Print request details clearly
        print(f"\033[1m>>> Sending request to Ollama ({MODEL}): '{prompt}' <<<\033[0m")
        
        start_time = time.time()
        
        # Use direct httpx call with full output
        response = httpx.post(url, json=payload, timeout=30.0)
        
        elapsed_time = time.time() - start_time
        print(f"\033[1m>>> Request completed in {elapsed_time:.2f} seconds <<<\033[0m")
        
        # Process the response
        if response.status_code == 200:
            data = response.json()
            
            # Pretty-print the full response data first
            print("\n\033[33m--- Full Response Data ---\033[0m")
            pprint(data)
            
            # Extract and display the content specifically
            if 'message' in data and 'content' in data['message']:
                content = data['message']['content']
                print(f"\n\033[32m--- Extracted Content ---\033[0m\n{content}")
                
                # This is the format that should be used in the extractor
                print("\n\033[36m--- Code to use in extractor.py ---\033[0m")
                print("""
# Get direct httpx response with full visibility
response = httpx.post(
    f"{settings.OLLAMA_BASE_URL}/api/chat",
    json={
        "model": settings.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json" 
    },
    timeout=30.0
)

# Print full response for visibility
print(f"\\n\\033[1m>>> Ollama response for '{summary[:30]}...' <<<\\033[0m")
pprint(response.json())

# Extract content
data = response.json()
content = data.get('message', {}).get('content', '')
print(f"\\n\\033[32m--- Extracted content ---\\033[0m\\n{content}\\n")
                """)
                
                return content
            else:
                print("\n\033[31mCould not extract content from response\033[0m")
                return None
        else:
            print(f"\n\033[31mError: Received status code {response.status_code}\033[0m")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"\n\033[31mError in extractor fix: {e}\033[0m")
        return None

if __name__ == "__main__":
    print("\nOLLAMA TEST SCRIPT")
    print(f"Model: {MODEL}")
    print(f"Ollama URL: {OLLAMA_URL}")
    
    # Run tests
    direct_response = test_ollama_direct()
    print_separator()
    
    test_with_ollama_module()
    print_separator()
    
    test_with_project_client()
    print_separator()
    
    # Implement ideal approach
    implement_ideal_ollama_call()
    print_separator()
    
    # Show fix for extractor
    fix_for_extractor()
