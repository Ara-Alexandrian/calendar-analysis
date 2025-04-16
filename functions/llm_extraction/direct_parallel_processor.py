"""
Direct parallel processor optimized for powerful GPU setups.
Designed specifically for dual RTX 3090s in NVLINK configuration to handle LLM extraction
without unnecessary batching.
"""
import logging
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import settings
from .utils import get_persistent_progress_bar
from .extractor import _extract_single_physicist_llm
from .ollama_client import OllamaClient

# Configure logging
logger = logging.getLogger(__name__)

class DirectParallelProcessor:
    """
    Optimized processor for direct LLM extraction taking advantage of dual RTX 3090s in NVLINK.
    Maximizes throughput for powerful GPU setups without unnecessary batching.
    """
    
    def __init__(self, max_workers=None, progress_callback=None):
        """
        Initialize the processor.
        
        Args:
            max_workers: Maximum number of concurrent workers (default: auto-configured)
            progress_callback: Optional callback function to report progress
        """
        # Auto-configure max_workers based on hardware
        if max_workers is None:
            # Take full advantage of dual RTX 3090s in NVLINK
            # 6 is a good starting point for dual RTX 3090s to avoid overwhelming them
            self.max_workers = getattr(settings, "LLM_MAX_WORKERS", 6)
        else:
            self.max_workers = max_workers
            
        self.progress_callback = progress_callback
        self.client = OllamaClient()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self._times = []
        self._errors = 0
        
    def _extract_with_retry(self, summary, canonical_names, max_retries=2):
        """
        Extract personnel with retry logic.
        
        Args:
            summary: Event summary text
            canonical_names: List of canonical names
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of extracted personnel names
        """
        # Try extraction up to max_retries+1 times
        for attempt in range(max_retries + 1):
            try:
                result = _extract_single_physicist_llm(summary, self.client, canonical_names)
                return result
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt+1} failed: {e}")
                if attempt < max_retries:
                    # Exponential backoff
                    time.sleep(1 * (2 ** attempt))
                else:
                    logger.error(f"All {max_retries+1} extraction attempts failed")
                    raise Exception(f"Extraction failed after {max_retries+1} attempts: {str(e)}")
    
    def process_dataframe(self, df, summary_col="summary", canonical_names=None):
        """
        Process a dataframe using direct parallel processing optimized for powerful GPUs.
        Takes full advantage of dual RTX 3090s in NVLINK configuration.
        
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
            
        logger.info(f"Processing {total_items} items with direct parallel processing (dual RTX 3090s)")
        
        # Initialize results storage
        results = [None] * total_items
        
        # Process items directly with high parallelism for powerful GPUs
        completed_count = 0
        progress_bar = get_persistent_progress_bar(total_items, "Direct GPU Extraction")
        
        try:
            # Use a thread pool optimized for direct processing on powerful GPUs
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Track futures to item indices
                future_to_idx = {}
                
                # Submit items individually for maximum GPU utilization
                for idx, summary in enumerate(summaries):
                    # Add a small staggered delay to avoid overwhelming Ollama on startup
                    if idx < self.max_workers:
                        time.sleep(0.5)
                        
                    future = executor.submit(
                        self._extract_with_retry,
                        summary, 
                        canonical_names
                    )
                    future_to_idx[future] = idx
                
                # Collect results as they complete
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    
                    try:
                        # Get individual item result
                        item_result = future.result()
                        results[idx] = item_result
                        completed_count += 1
                        progress_bar.update(1)
                        
                        # Call progress callback if provided
                        if self.progress_callback and completed_count % 5 == 0:  # Update every 5 items
                            # Use a short delay to ensure Streamlit refreshes
                            time.sleep(0.02)
                            
                            # Log progress periodically
                            if completed_count % 20 == 0 or completed_count == total_items:
                                logger.info(f"Progress: {completed_count}/{total_items} items complete ({completed_count/total_items*100:.1f}%)")
                            
                            # Call the progress callback to update UI
                            self.progress_callback(completed_count, total_items)
                    
                    except Exception as e:
                        logger.error(f"Error processing item {idx}: {e}")
                        results[idx] = ["Unknown_Error"]
                        completed_count += 1
                        progress_bar.update(1)
                        
                        # Track errors
                        with self._lock:
                            self._errors += 1
        finally:
            progress_bar.close()
            
        # Log performance statistics
        error_rate = (self._errors / max(1, total_items)) * 100
        logger.info(f"Error rate: {error_rate:.2f}% ({self._errors} errors)")
        
        # Ensure all results are set
        for i in range(total_items):
            if results[i] is None:
                results[i] = ["Unknown_Error"]
                
        # Add results to dataframe
        df_copy['extracted_personnel'] = results
        return df_copy
