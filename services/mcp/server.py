import os
import sys
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
import time

# Add imports for Hugging Face Transformers and related libraries
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mcp_server")

# Create FastAPI app
app = FastAPI(title="Calendar Analysis MCP Server")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model
class MCPRequest(BaseModel):
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    
# Define the response model
class MCPResponse(BaseModel):
    text: str
    
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Calendar Analysis MCP Server is running"}

@app.post("/v1/generate")
async def generate(request: MCPRequest):
    """Generate text based on the provided messages."""
    try:
        logger.info(f"Received request with {len(request.messages)} messages")
        
        # Extract the content from the last message
        last_message = request.messages[-1]["content"] if request.messages else ""
        
        # TODO: Replace this with actual model inference
        # This is just a placeholder that echoes back the input
        response = f"Calendar Analysis MCP Server received: {last_message}"
        
        logger.info(f"Generated response of length {len(response)}")
        return {"text": response}
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {"text": f"Error: {str(e)}"}

@app.post("/v1/extract_personnel")
async def extract_personnel(request: Request):
    """
    Extract personnel names from a calendar event summary.
    Endpoint specifically tailored for the calendar analysis project.
    """
    try:
        data = await request.json()
        summary = data.get("summary", "")
        canonical_names = data.get("canonical_names", [])
        
        logger.info(f"Received personnel extraction request for summary of length {len(summary)}")
        
        # If the model hasn't been loaded yet, load it now
        global model, tokenizer, generator
        if None in (model, tokenizer, generator) and HF_AVAILABLE:
            # Use the model name from environment or default to phi-2
            model_name = os.environ.get("MCP_MODEL_NAME", "microsoft/phi-2")
            model, tokenizer, generator = load_model(model_name)
            
        # Extract personnel from the summary
        extracted_names = extract_personnel_from_summary(summary, canonical_names)
        
        logger.info(f"Extracted {len(extracted_names)} personnel names: {extracted_names}")
        
        return {"extracted_personnel": extracted_names}
    
    except Exception as e:
        logger.error(f"Error in personnel extraction endpoint: {str(e)}")
        return {"error": str(e), "extracted_personnel": ["Unknown_Error"]}

# Update the chat endpoint to use the model for inference
@app.post("/v1/chat")
async def chat(request: Request):
    """Chat endpoint compatible with OpenAI format."""
    try:
        data = await request.json()
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 1000)
        
        logger.info(f"Received chat request with {len(messages)} messages")
        
        # If the model hasn't been loaded yet, load it now
        global model, tokenizer, generator
        if None in (model, tokenizer, generator) and HF_AVAILABLE:
            # Use the model name from environment or default to phi-2
            model_name = os.environ.get("MCP_MODEL_NAME", "microsoft/phi-2")
            model, tokenizer, generator = load_model(model_name)
            
        if None in (model, tokenizer, generator):
            # If model still not available, return a placeholder response
            response_content = "Model not available. Please install the necessary dependencies."
            logger.error("Model not available for chat request")
        else:
            # Format the messages for the model
            formatted_prompt = format_messages_for_model(messages)
            
            # Generate the response
            response = generator(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1
            )
            
            # Extract the generated text (remove the prompt part)
            response_content = response[0]["generated_text"][len(formatted_prompt):].strip()
            logger.info(f"Generated chat response of length {len(response_content)}")
        
        return {
            "id": "mcp-chat-response",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(formatted_prompt) // 4,  # Rough estimate
                "completion_tokens": len(response_content) // 4,  # Rough estimate
                "total_tokens": (len(formatted_prompt) + len(response_content)) // 4  # Rough estimate
            }
        }
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}

# Global variables to store the model and tokenizer
model = None
tokenizer = None
generator = None

def load_model(model_name="microsoft/phi-2"):
    """
    Load a Hugging Face model and tokenizer.
    
    Args:
        model_name: Name of the Hugging Face model to load
        
    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    global model, tokenizer, generator
    
    if not HF_AVAILABLE:
        logger.error("Hugging Face Transformers library not available. Please install with 'pip install transformers'")
        return None, None, None
    
    try:
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        # For most models, we'll want to use the BetterTransformer path for faster inference
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,  # Use float16 for better performance
            device_map="auto"  # Let the library decide on the best device mapping
        )
        
        # Create a text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        return model, tokenizer, generator
    
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None, None, None

def format_messages_for_model(messages):
    """
    Format chat messages into a prompt suitable for the model.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        str: Formatted prompt
    """
    formatted_prompt = ""
    
    # Simple formatting - can be customized based on the model's expected format
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        
        if role == "system":
            formatted_prompt += f"System: {content}\n\n"
        elif role == "user":
            formatted_prompt += f"User: {content}\n\n"
        elif role == "assistant":
            formatted_prompt += f"Assistant: {content}\n\n"
    
    # Add the final assistant prompt
    formatted_prompt += "Assistant: "
    
    return formatted_prompt

def extract_personnel_from_summary(summary, canonical_names):
    """
    Extract personnel names from a calendar event summary.
    This is the specific functionality needed for the calendar analysis project.
    
    Args:
        summary: The calendar event summary text
        canonical_names: List of known canonical names to extract
        
    Returns:
        list: List of extracted names
    """
    if not summary or not canonical_names:
        return []
    
    try:
        # Skip model loading if we're doing a simple check
        if summary.lower() in ["test", "ping", "hello"]:
            return []
        
        # Ensure the model is loaded
        global model, tokenizer, generator
        if model is None or tokenizer is None or generator is None:
            model, tokenizer, generator = load_model()
            if model is None:
                return ["Unknown_Error"]
        
        # Construct the prompt for personnel extraction
        prompt = f"""
        Your task is to identify physicist names from the provided calendar event summary.
        You are given a specific list of known canonical physicist names.
        Analyze the summary and identify ONLY the canonical names from the list below that are mentioned or clearly referenced in the summary.
        Consider variations in names (e.g., initials, last names only, common misspellings if obvious) but map them back EXACTLY to a name present in the canonical list.
        Do not guess or include names not on the list. If multiple physicists are mentioned, include all of them.
        If no physicists from the list are clearly identified, return an empty list.

        Known Canonical Physicist Names:
        {json.dumps(canonical_names, indent=2)}

        Event Summary:
        "{summary}"

        IMPORTANT: Respond ONLY with a valid JSON array. Do not use any other JSON format.
        Correct format examples:
        []
        ["D. Solis"]
        ["C. Chu", "D. Solis"]
        """
        
        # Generate the response
        response = generator(
            prompt,
            max_new_tokens=150,
            temperature=0.1,  # Low temperature for more deterministic outputs
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )
        
        # Extract the generated text
        generated_text = response[0]["generated_text"][len(prompt):].strip()
        logger.debug(f"Generated response: {generated_text}")
        
        # Parse the JSON list from the generated text
        try:
            # Try to find array in response using simple regex-like approach
            start_idx = generated_text.find('[')
            end_idx = generated_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = generated_text[start_idx:end_idx+1]
                names = json.loads(json_text)
                # Filter names to only include those in the canonical list
                valid_names = [name for name in names if name in canonical_names]
                return valid_names
            else:
                logger.warning(f"Could not find JSON array in response: {generated_text}")
                return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response: {generated_text}")
            return []
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return ["Unknown_Error"]
            
    except Exception as e:
        logger.error(f"Error extracting personnel from summary: {e}")
        return ["Unknown_Error"]

def start_server(host="0.0.0.0", port=8000):
    """Start the MCP server."""
    logger.info(f"Starting MCP server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # You can override the host and port using environment variables
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", 8000))
    start_server(host, port)
