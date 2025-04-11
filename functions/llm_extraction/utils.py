# functions/llm_extraction/utils.py
"""
Utility functions and classes for LLM extraction.
"""
import time
import sys
from threading import Thread, Lock

class SimpleTqdm:
    """
    A simple progress bar implementation that doesn't require tqdm.
    Used as a fallback when tqdm is not available.
    """
    def __init__(self, iterable=None, total=None, desc=None):
        self.iterable = iterable
        self.total = total if total is not None else len(iterable) if iterable is not None else 100
        self.desc = desc or "Processing"
        self.current = 0
        self.start_time = time.time()
        
    def __iter__(self):
        if self.iterable:
            for item in self.iterable:
                yield item
                self.current += 1
                self._update_progress()
    
    def update(self, n=1):
        self.current += n
        self._update_progress()
    
    def _update_progress(self):
        percent = min(100, int((self.current / self.total) * 100))
        elapsed = time.time() - self.start_time
        bar_length = 30
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        sys.stdout.write(f'\r{self.desc}: [{bar}] {percent}% {self.current}/{self.total} - {elapsed:.1f}s')
        if self.current >= self.total:
            sys.stdout.write('\n')
        sys.stdout.flush()

class PersistentProgressBar:
    """
    A progress bar that can be updated from different threads and maintains
    its position at the bottom of the terminal.
    """
    def __init__(self, total=100, desc="Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.lock = Lock()
        self.visible = False
        self.active = False
        
    def start(self):
        """Start the progress bar thread."""
        self.active = True
        self.update_thread = Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        return self
        
    def update(self, n=1):
        """Update the progress count."""
        with self.lock:
            self.current += n
            if not self.visible:
                self._display()
                self.visible = True
                
    def set_description(self, desc):
        """Change the description text."""
        with self.lock:
            self.desc = desc
            
    def _update_loop(self):
        """Update the display periodically."""
        while self.active and self.current < self.total:
            self._display()
            time.sleep(0.5)
        self._display()  # Final update
        
    def _display(self):
        """Display the progress bar."""
        with self.lock:
            percent = min(100, int((self.current / self.total) * 100))
            elapsed = time.time() - self.start_time
            bar_length = 30
            filled_length = int(bar_length * self.current // self.total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Calculate rate and ETA
            if elapsed > 0:
                rate = self.current / elapsed
                eta = (self.total - self.current) / rate if rate > 0 else 0
                eta_str = f"ETA: {eta:.1f}s" if eta > 0 else "Complete"
            else:
                eta_str = "Calculating..."
                
            # Save cursor position, move to bottom of terminal, clear line
            sys.stdout.write(f'\r{self.desc}: [{bar}] {percent}% {self.current}/{self.total} - {elapsed:.1f}s - {eta_str}')
            sys.stdout.flush()
            
    def close(self):
        """Clean up the progress bar."""
        self.active = False
        if self.visible:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self.visible = False
            
def get_persistent_progress_bar(total, desc="Processing"):
    """Factory function to create and start a persistent progress bar."""
    return PersistentProgressBar(total, desc).start()