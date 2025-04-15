"""
Enhanced batch processor for LLM extraction that optimizes workload distribution
between Ollama server (dual RTX 3090s in NVLink) and local MCP server (RTX 4090).
"""
import logging
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import threading
import queue

from config import settings
from .smart_router import get_router
from .utils import get_persistent_progress_bar

# Configure logging
logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Advanced batch processor for optimized LLM extraction across multiple hardware setups.
    Distributes workload between Ollama (dual RTX 3090s) and MCP (RTX 4090) servers.
    """
    
    def __init__(self, batch_size=8, max_workers=None, progress_callback=None):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
            max_workers: Maximum number of concurrent workers (default: auto-configured)
            progress_callback: Optional callback function to report progress
        """
        self.batch_size = batch_size
        # Auto-configure max_workers based on hardware
        if max_workers is None:
            # Default to higher concurrency to utilize both systems
            self.max_workers = getattr(settings, "LLM_MAX_WORKERS", 12)
        else:
            self.max_workers = max_workers
            
        self.progress_callback = progress_callback
        self.router = get_router()
        
        # Thread safety
        self._lock = threading.Lock()
        self._results_queue = queue.Queue()
        
        # Performance tracking
        self._batch_times = []
        self._provider_usage = {"mcp": 0, "ollama": 0}
        self._errors = {"mcp": 0, "ollama": 0, "total": 0}
    
    def _process_batch(self, items, batch_idx, extra_args=None):
        """
        Process a batch of items, with intelligent error handling and retry logic.
        
        Args:
            items: List of items to process in this batch
            batch_idx: Index of the current batch
            extra_args: Additional arguments for processing
            
        Returns:
            List of results in the same order as items
        """
        start_time = time.time()
        results = [None] * len(items)
        canonical_names = extra_args.get("canonical_names", [])
        
        logger.debug(f"Starting batch {batch_idx} with {len(items)} items")
        
        # Use ThreadPoolExecutor to process items within the batch in parallel
        with ThreadPoolExecutor(max_workers=min(len(items), self.max_workers)) as executor:
            # Track futures to original indices
            future_to_idx = {}
            
            # Submit all tasks
            for i, item in enumerate(items):
                # Submit with task routing hint to better distribute between GPUs
                # Shorter items (likely simpler) -> MCP/4090
                # Longer/complex items -> Ollama/dual 3090s
                task_complexity = min(10, max(1, len(str(item)) // 100))
                
                # Determine task type based on context
                # For personnel extraction, use extraction type
                task_type = "extraction"
                
                future = executor.submit(
                    self._process_single_item, 
                    item, 
                    task_type=task_type,
                    task_complexity=task_complexity,
                    canonical_names=canonical_names
                )
                future_to_idx[future] = i
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result, provider_name = future.result()
                    results[idx] = result
                    
            # Track provider usage (simplified)
                    with self._lock:
                        if provider_name:
                            if provider_name not in self._provider_usage:
                                self._provider_usage[provider_name] = 0
                            self._provider_usage[provider_name] += 1
                            
                except Exception as e:
                    logger.error(f"Error processing item {idx} in batch {batch_idx}: {e}")
                    results[idx] = ["Unknown_Error"]
                    
                    with self._lock:
                        self._errors["total"] += 1
        
        # Record batch processing time for auto-tuning
        elapsed = time.time() - start_time
        with self._lock:
            self._batch_times.append(elapsed)
            
            # Dynamically adjust batch size based on performance
            if len(self._batch_times) >= 3:
                avg_time = sum(self._batch_times[-3:]) / 3
                  # Scale batch size if processing is fast or slow
                if avg_time < 1.0 and self.batch_size < 16:
                    self.batch_size += 2
                    logger.info(f"Increasing batch size to {self.batch_size} (avg time: {avg_time:.2f}s)")
                elif avg_time > 5.0 and self.batch_size > 4:
                    self.batch_size -= 2
                    logger.info(f"Decreasing batch size to {self.batch_size} (avg time: {avg_time:.2f}s)")
        
        logger.debug(f"Completed batch {batch_idx} in {elapsed:.2f}s")
        return results
        
    def _process_single_item(self, item, task_type="extraction", task_complexity=1, **kwargs):
        """
        Process a single item using the simplified direct provider.
        
        Args:
            item: The item to process
            task_type: Type of task (extraction, chat, etc.)
            task_complexity: Complexity level (1-10)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (result, provider_name)
        """
        canonical_names = kwargs.get("canonical_names", [])        # Get the direct provider
        try:
            provider_name, client = self.router.get_best_provider(task_type=task_type, complexity=task_complexity)
        except Exception as e:
            logger.error(f"Failed to get LLM provider: {e}")
            return ["Unknown_Error"], None
            
        if not client:
            logger.error("No LLM provider available")
            return ["Unknown_Error"], None
        
        # For extraction tasks, use the extract_personnel method
        if task_type == "extraction":
            try:
                start = time.time()
                # Get the result with retry logic built in
                result = self._extract_with_retry(item, client, canonical_names)
                elapsed = time.time() - start
                
                # Log performance stats
                logger.debug(f"Item processed by {provider_name} in {elapsed:.2f}s")
                
                return result, provider_name
            except Exception as e:
                logger.error(f"Error during extraction with {provider_name}: {e}")
                with self._lock:
                    if provider_name not in self._errors:
                        self._errors[provider_name] = 0
                    self._errors[provider_name] += 1
                    self._errors["total"] += 1
                return ["Unknown_Error"], provider_name
        
        # Default fallback
        return ["Unknown_Error"], None
    
    def _extract_with_retry(self, summary, client, canonical_names, max_retries=2):
        """
        Extract personnel with retry logic.
        
        Args:
            summary: Event summary text
            client: LLM client
            canonical_names: List of canonical names
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of extracted personnel names
        """
        from .extractor import _extract_single_physicist_llm
        
        # Try extraction up to max_retries+1 times
        for attempt in range(max_retries + 1):
            try:
                result = _extract_single_physicist_llm(summary, client, canonical_names)
                return result
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt+1} failed: {e}")
                if attempt < max_retries:
                    # Exponential backoff
                    time.sleep(1 * (2 ** attempt))
                else:
                    logger.error(f"All {max_retries+1} extraction attempts failed")
                    raise
    
    def process_dataframe(self, df, summary_col="summary", canonical_names=None):
        """
        Process a dataframe using smart batching and optimal workload distribution.
        
        Args:
            df: Input dataframe
            summary_col: Column containing text to process
            canonical_names: List of canonical names to extract
            
        Returns:
            Dataframe with added 'extracted_personnel' column
        """
        if not canonical_names:
            logger.error("No canonical names provided")
            return df
            
        df_copy = df.copy()
        df_copy[summary_col] = df_copy[summary_col].fillna('')
        
        summaries = df_copy[summary_col].tolist()
        total_items = len(summaries)
        
        if total_items == 0:
            logger.warning("Empty dataframe, nothing to process")
            return df_copy
            
        logger.info(f"Processing {total_items} items with smart batching")
        
        # Initialize results storage
        results = [None] * total_items
        
        # Create batches
        batch_size = min(self.batch_size, max(1, total_items // 10))  # Ensures reasonable batch count
        batches = [summaries[i:i+batch_size] for i in range(0, total_items, batch_size)]
        
        # Process batches with ThreadPoolExecutor for parallel batch processing
        completed_count = 0
        progress_bar = get_persistent_progress_bar(total_items, "SmartBatch Extraction")
        
        try:
            with ThreadPoolExecutor(max_workers=min(len(batches), self.max_workers // 2)) as executor:
                # Track futures to batch indices
                future_to_batch_idx = {}
                
                # Submit all batches
                for batch_idx, batch in enumerate(batches):
                    future = executor.submit(
                        self._process_batch, 
                        batch, 
                        batch_idx,
                        {"canonical_names": canonical_names}
                    )
                    future_to_batch_idx[future] = batch_idx
                
                # Collect results as they complete
                for future in as_completed(future_to_batch_idx):
                    batch_idx = future_to_batch_idx[future]
                    start_idx = batch_idx * batch_size
                    
                    try:
                        batch_results = future.result()
                          # Store results in the correct positions
                        for i, result in enumerate(batch_results):
                            if start_idx + i < total_items:
                                results[start_idx + i] = result
                                completed_count += 1
                                progress_bar.update(1)
                                  # Call progress callback if provided
                                # Make progress updates much more visible and frequent
                                if self.progress_callback:
                                    # Use a longer delay to ensure Streamlit fully refreshes
                                    import time
                                    time.sleep(0.05)  # Increased delay for better Streamlit refresh
                                    
                                    # Log every progress update
                                    logger.info(f"Batch progress: {completed_count}/{total_items} items complete ({completed_count/total_items*100:.1f}%)")
                                    
                                    # Call the progress callback to update UI
                                    self.progress_callback(completed_count, total_items)
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {e}")
                        # Mark all items in this batch as errors
                        for i in range(len(batches[batch_idx])):
                            if start_idx + i < total_items:
                                results[start_idx + i] = ["Unknown_Error"]
                                completed_count += 1
                                progress_bar.update(1)
        finally:
            progress_bar.close()
            
        # Log performance statistics
        provider_usage = dict(self._provider_usage)
        logger.info(f"Provider usage: {provider_usage}")
        
        if self._batch_times:
            avg_time = sum(self._batch_times) / len(self._batch_times)
            logger.info(f"Average batch processing time: {avg_time:.2f}s")
            
        error_rate = (self._errors.get("total", 0) / max(1, total_items)) * 100
        logger.info(f"Error rate: {error_rate:.2f}% ({self._errors.get('total', 0)} errors)")
        
        # Ensure all results are set
        for i in range(total_items):
            if results[i] is None:
                results[i] = ["Unknown_Error"]
                
        # Add results to dataframe
        df_copy['extracted_personnel'] = results
        return df_copy
