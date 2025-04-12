# functions/llm_extraction/smart_router.py
"""
Smart router for optimized LLM extraction leveraging dual-GPU setup:
- Ollama server with dual RTX 3090s in NVLink
- Local MCP server with RTX 4090

This router intelligently distributes workloads between GPU systems based on:
1. Task complexity and size
2. Current load on each system
3. Hardware-specific strengths
4. Error rates and observed performance
"""

import logging
import time
import threading
import random
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Import settings for provider configuration
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Constants for router configuration
MAX_MCP_CONCURRENT_REQUESTS = 18  # RTX 4090
MAX_OLLAMA_CONCURRENT_REQUESTS = 24  # Dual RTX 3090s in NVLink
ERROR_THRESHOLD = 0.3  # 30% error rate triggers throttling
ERROR_MEMORY_WINDOW = 50  # Remember last 50 requests for error rate calculation
LATENCY_MEMORY_WINDOW = 30  # Remember last 30 requests for latency calculation
LOAD_CHECK_INTERVAL = 5  # Seconds between load checks
LARGE_MODELS = ['llama3:70b', 'llama3:70b-q4_0', 'mixtral', 'phi3', 'llama3:8b-instruct']

class SmartRouter:
    """
    Hardware-optimized router for distributing LLM tasks between:
    - Local MCP server with RTX 4090 (faster for smaller tasks, lower latency)
    - Ollama server with dual RTX 3090s in NVLink (better for complex tasks with large memory requirements)
    """
    
    def __init__(self):
        # Track current load on each provider
        self._mcp_active_requests = 0
        self._ollama_active_requests = 0
        
        # Track request success/failure
        self._error_history = {
            'mcp': deque(maxlen=ERROR_MEMORY_WINDOW),
            'ollama': deque(maxlen=ERROR_MEMORY_WINDOW)
        }
        
        # Track latency per provider
        self._latency_history = {
            'mcp': deque(maxlen=LATENCY_MEMORY_WINDOW),
            'ollama': deque(maxlen=LATENCY_MEMORY_WINDOW)
        }
        
        # Configure providers
        self._providers = {
            'mcp': {
                'max_concurrent': MAX_MCP_CONCURRENT_REQUESTS,
                'hardware': 'RTX 4090',
                'strengths': 'Lower latency, better for smaller tasks'
            },
            'ollama': {
                'max_concurrent': MAX_OLLAMA_CONCURRENT_REQUESTS,
                'hardware': 'Dual RTX 3090s in NVLink',
                'strengths': 'More VRAM, better for memory-intensive tasks'
            }
        }
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Initialize with default thresholds
        self._dynamic_thresholds = {
            'mcp': MAX_MCP_CONCURRENT_REQUESTS,
            'ollama': MAX_OLLAMA_CONCURRENT_REQUESTS
        }
        
        # Track recent errors to prevent using a failing provider
        self._recent_errors = {
            'mcp': 0,
            'ollama': 0
        }
        
        logger.info(f"Smart router initialized with provider configuration: "
                    f"MCP (RTX 4090): {MAX_MCP_CONCURRENT_REQUESTS} max concurrent, "
                    f"Ollama (Dual RTX 3090s): {MAX_OLLAMA_CONCURRENT_REQUESTS} max concurrent")
    
    def record_request_start(self, provider: str) -> None:
        """Record the start of a request to a provider"""
        with self._lock:
            if provider == 'mcp':
                self._mcp_active_requests += 1
            elif provider == 'ollama':
                self._ollama_active_requests += 1
            
            logger.debug(f"Started request to {provider}. Current load - "
                         f"MCP: {self._mcp_active_requests}/{self._dynamic_thresholds['mcp']}, "
                         f"Ollama: {self._ollama_active_requests}/{self._dynamic_thresholds['ollama']}")
    
    def record_request_end(self, provider: str, success: bool, latency_ms: float) -> None:
        """Record the end of a request to a provider with status and latency"""
        with self._lock:
            if provider == 'mcp':
                self._mcp_active_requests = max(0, self._mcp_active_requests - 1)
            elif provider == 'ollama':
                self._ollama_active_requests = max(0, self._ollama_active_requests - 1)
            
            # Record success/failure
            self._error_history[provider].append(0 if success else 1)
            
            # Record latency for successful requests
            if success:
                self._latency_history[provider].append(latency_ms)
                # Reset recent error count on success
                self._recent_errors[provider] = max(0, self._recent_errors[provider] - 1)
            else:
                # Increment recent error count
                self._recent_errors[provider] += 1
            
            # Update dynamic thresholds based on error rate
            self._update_dynamic_thresholds()
            
            logger.debug(f"Completed request to {provider} (success={success}, latency={latency_ms}ms). "
                         f"Current load - MCP: {self._mcp_active_requests}/{self._dynamic_thresholds['mcp']}, "
                         f"Ollama: {self._ollama_active_requests}/{self._dynamic_thresholds['ollama']}")
    
    def _update_dynamic_thresholds(self) -> None:
        """Update dynamic thresholds based on error rates"""
        for provider in ['mcp', 'ollama']:
            if not self._error_history[provider]:
                continue
            
            error_rate = sum(self._error_history[provider]) / len(self._error_history[provider])
            base_threshold = self._providers[provider]['max_concurrent']
            
            # Reduce threshold if error rate is high
            if error_rate > ERROR_THRESHOLD:
                reduction_factor = 1.0 - (error_rate - ERROR_THRESHOLD)
                new_threshold = max(1, int(base_threshold * reduction_factor))
                self._dynamic_thresholds[provider] = new_threshold
                logger.warning(f"High error rate ({error_rate:.2f}) for {provider}. "
                              f"Reducing threshold to {new_threshold} (was {base_threshold})")
            else:
                # Gradually restore threshold if error rate is acceptable
                current = self._dynamic_thresholds[provider]
                if current < base_threshold:
                    self._dynamic_thresholds[provider] = min(base_threshold, current + 1)
                    logger.info(f"Restoring threshold for {provider} to {self._dynamic_thresholds[provider]}")
    
    def get_average_latency(self, provider: str) -> float:
        """Get average latency for a provider (in ms)"""
        with self._lock:
            history = self._latency_history[provider]
            if not history:
                return 1000.0  # Default assumption if no data
            return sum(history) / len(history)
    
    def get_error_rate(self, provider: str) -> float:
        """Get error rate for a provider (0.0 to 1.0)"""
        with self._lock:
            history = self._error_history[provider]
            if not history:
                return 0.0
            return sum(history) / len(history)
    
    def get_load_percentage(self, provider: str) -> float:
        """Get current load percentage for a provider (0.0 to 1.0)"""
        with self._lock:
            if provider == 'mcp':
                return self._mcp_active_requests / self._dynamic_thresholds['mcp']
            elif provider == 'ollama':
                return self._ollama_active_requests / self._dynamic_thresholds['ollama']
            return 0.0
    
    def is_provider_at_capacity(self, provider: str) -> bool:
        """Check if a provider is at capacity"""
        with self._lock:
            if provider == 'mcp':
                return self._mcp_active_requests >= self._dynamic_thresholds['mcp']
            elif provider == 'ollama':
                return self._ollama_active_requests >= self._dynamic_thresholds['ollama']
            return False
    
    def has_recent_errors(self, provider: str) -> bool:
        """Check if a provider has recent errors that suggest it might be failing"""
        with self._lock:
            return self._recent_errors[provider] >= 3  # Consider failing if 3+ recent errors
    
    def route_request(self, 
                      task_size: int, 
                      task_complexity: int, 
                      model_name: str = None, 
                      prefer_provider: str = None) -> str:
        """
        Route a request to the appropriate provider based on task characteristics and current load
        
        Args:
            task_size: Size of the input task (e.g., token count) (1-10, where 10 is largest)
            task_complexity: Complexity of the task (1-10, where 10 is most complex)
            model_name: Name of the model to use (optional)
            prefer_provider: Provider to prefer if available (optional)
            
        Returns:
            Provider to use ('mcp' or 'ollama')
        """
        with self._lock:
            # Check if either provider is completely down (3+ consecutive errors)
            mcp_down = self.has_recent_errors('mcp')
            ollama_down = self.has_recent_errors('ollama')
            
            # If one is down but not the other, use the one that's up
            if mcp_down and not ollama_down:
                logger.warning("MCP appears to be down, routing to Ollama (dual RTX 3090s)")
                return 'ollama'
            if ollama_down and not mcp_down:
                logger.warning("Ollama appears to be down, routing to MCP (RTX 4090)")
                return 'mcp'
            
            # If both are down or both are up, continue with normal routing logic
            
            # If model is specified and it's large, prefer Ollama's dual RTX 3090s
            if model_name and model_name in LARGE_MODELS:
                if not self.is_provider_at_capacity('ollama'):
                    logger.info(f"Routing large model {model_name} to Ollama (dual RTX 3090s)")
                    return 'ollama'
            
            # If explicit preference and not at capacity, honor it
            if prefer_provider and not self.is_provider_at_capacity(prefer_provider):
                return prefer_provider
            
            # Check if either provider is at capacity
            mcp_at_capacity = self.is_provider_at_capacity('mcp')
            ollama_at_capacity = self.is_provider_at_capacity('ollama')
            
            # If only one is at capacity, use the other
            if mcp_at_capacity and not ollama_at_capacity:
                return 'ollama'
            if ollama_at_capacity and not mcp_at_capacity:
                return 'mcp'
            
            # If both are at capacity, pick the one with the lowest load percentage
            if mcp_at_capacity and ollama_at_capacity:
                mcp_load = self.get_load_percentage('mcp')
                ollama_load = self.get_load_percentage('ollama')
                return 'mcp' if mcp_load < ollama_load else 'ollama'
            
            # For low complexity tasks (1-3), prefer MCP/RTX 4090 for better latency
            if task_complexity <= 3:
                # Small, simple tasks are perfect for the RTX 4090's speed
                logger.debug(f"Routing low complexity task ({task_complexity}) to MCP (RTX 4090)")
                return 'mcp'
            
            # For high complexity tasks (8-10), prefer Ollama/dual RTX 3090s for more VRAM
            if task_complexity >= 8:
                # Complex tasks benefit from the combined VRAM of dual RTX 3090s
                logger.debug(f"Routing high complexity task ({task_complexity}) to Ollama (dual RTX 3090s)")
                return 'ollama'
            
            # For medium complexity tasks (4-7), consider size and load balance
            if task_size >= 7:  # Large tasks may benefit from more VRAM
                logger.debug(f"Routing large task (size={task_size}) to Ollama (dual RTX 3090s)")
                return 'ollama'
            
            # For medium tasks with medium complexity, balance between providers
            # considering current load and task characteristics
            
            # Calculate load-adjusted weights
            mcp_load = self.get_load_percentage('mcp')
            ollama_load = self.get_load_percentage('ollama')
            
            # Adjust weight based on complexity within medium range (4-7)
            # As complexity increases from 4 to 7, favor dual RTX 3090s more
            mcp_weight = 0.7 - ((task_complexity - 4) * 0.1)  # 0.7 to 0.4 as complexity increases
            ollama_weight = 1.0 - mcp_weight  # 0.3 to 0.6 as complexity increases
            
            # Adjust for load
            mcp_weight *= (1.0 - mcp_load)
            ollama_weight *= (1.0 - ollama_load)
            
            # If one weight is significantly higher, choose that provider
            if mcp_weight > (ollama_weight * 1.5):
                logger.debug(f"Routing medium task to MCP (RTX 4090) based on weighted decision")
                return 'mcp'
            elif ollama_weight > (mcp_weight * 1.5):
                logger.debug(f"Routing medium task to Ollama (dual RTX 3090s) based on weighted decision")
                return 'ollama'
            
            # If weights are similar, flip a weighted coin
            threshold = mcp_weight / (mcp_weight + ollama_weight)
            if random.random() < threshold:
                return 'mcp'
            else:
                return 'ollama'

# Singleton router instance
_router_instance = None

def get_router() -> SmartRouter:
    """Get the singleton router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = SmartRouter()
    return _router_instance
