import concurrent.futures
import logging
import os
import time
from typing import List, Dict, Any, Callable, Optional

class ThreadPoolManager:
    """
    Manages a pool of worker threads for parallel document processing.
    """
    def __init__(self, 
                 max_workers: int = None, 
                 logging_level: str = 'INFO'):
        """
        Initialize the thread pool manager.
        
        Args:
            max_workers: Maximum number of worker threads (defaults to min(32, os.cpu_count() + 4))
            logging_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)
        self.logger = self._setup_logger(logging_level)
        self.logger.info(f"Initializing ThreadPoolManager with {self.max_workers} workers")
    
    def process_batch(self, 
                     items: List[Any], 
                     processing_func: Callable,
                     progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Process multiple items in parallel.
        
        Args:
            items: List of items to process
            processing_func: Function to process each item
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of results from processing each item
        """
        start_time = time.time()
        self.logger.info(f"Starting parallel processing of {len(items)} items")
        
        results = []
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(processing_func, item): item 
                for item in items
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(completed, len(items), result)
                        
                except Exception as e:
                    self.logger.error(f"Error processing item: {str(e)}", exc_info=True)
                    # Add failed result
                    results.append({
                        'success': False,
                        'error': str(e),
                        'item': item
                    })
                    completed += 1
                    
                    # Call progress callback for failed item
                    if progress_callback:
                        progress_callback(completed, len(items), {
                            'success': False,
                            'error': str(e),
                            'item': item
                        })
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Completed processing {len(items)} items in {elapsed_time:.2f} seconds")
        
        return results
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.ThreadPoolManager')
        
        if level.upper() == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        elif level.upper() == 'INFO':
            logger.setLevel(logging.INFO)
        elif level.upper() == 'WARNING':
            logger.setLevel(logging.WARNING)
        elif level.upper() == 'ERROR':
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)
        
        # Add console handler if not already added
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger