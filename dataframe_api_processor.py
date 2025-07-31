import pandas as pd
import requests
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional, Callable
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataFrameAPIProcessor:
    """
    A comprehensive class for processing pandas DataFrame rows and making API calls.
    Supports synchronous, asynchronous, and batch processing approaches.
    """
    
    def __init__(self, base_url: str = None, headers: Dict = None, timeout: int = 30):
        self.base_url = base_url or os.getenv('API_BASE_URL', 'https://jsonplaceholder.typicode.com')
        self.headers = headers or {'Content-Type': 'application/json'}
        self.timeout = timeout
        self.results = []
        self.errors = []
    
    def process_row_sync(self, row: pd.Series, endpoint: str = '/posts', 
                        method: str = 'POST') -> Dict[str, Any]:
        """
        Process a single DataFrame row with synchronous API call.
        
        Args:
            row: pandas Series representing a DataFrame row
            endpoint: API endpoint to call
            method: HTTP method (GET, POST, PUT, DELETE)
            
        Returns:
            Dictionary containing the API response and metadata
        """
        try:
            url = f"{self.base_url}{endpoint}"
            row_data = row.to_dict()
            
            # Remove NaN values
            row_data = {k: v for k, v in row_data.items() if pd.notna(v)}
            
            logger.info(f"Making {method} request to {url} with data: {row_data}")
            
            if method.upper() == 'GET':
                response = requests.get(url, params=row_data, 
                                      headers=self.headers, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, json=row_data, 
                                       headers=self.headers, timeout=self.timeout)
            elif method.upper() == 'PUT':
                response = requests.put(url, json=row_data, 
                                      headers=self.headers, timeout=self.timeout)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=self.headers, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            result = {
                'row_index': row.name,
                'status_code': response.status_code,
                'response_data': response.json() if response.content else {},
                'request_data': row_data,
                'success': True,
                'error': None
            }
            
            logger.info(f"Successfully processed row {row.name}")
            return result
            
        except requests.exceptions.RequestException as e:
            error_result = {
                'row_index': row.name,
                'status_code': getattr(e.response, 'status_code', None),
                'response_data': {},
                'request_data': row_data,
                'success': False,
                'error': str(e)
            }
            logger.error(f"Error processing row {row.name}: {str(e)}")
            return error_result
        except Exception as e:
            error_result = {
                'row_index': row.name,
                'status_code': None,
                'response_data': {},
                'request_data': row.to_dict() if 'row' in locals() else {},
                'success': False,
                'error': str(e)
            }
            logger.error(f"Unexpected error processing row {getattr(row, 'name', 'unknown')}: {str(e)}")
            return error_result
    
    async def process_row_async(self, session: aiohttp.ClientSession, 
                               row: pd.Series, endpoint: str = '/posts', 
                               method: str = 'POST') -> Dict[str, Any]:
        """
        Process a single DataFrame row with asynchronous API call.
        
        Args:
            session: aiohttp ClientSession
            row: pandas Series representing a DataFrame row
            endpoint: API endpoint to call
            method: HTTP method
            
        Returns:
            Dictionary containing the API response and metadata
        """
        try:
            url = f"{self.base_url}{endpoint}"
            row_data = row.to_dict()
            
            # Remove NaN values
            row_data = {k: v for k, v in row_data.items() if pd.notna(v)}
            
            logger.info(f"Making async {method} request to {url} with data: {row_data}")
            
            if method.upper() == 'GET':
                async with session.get(url, params=row_data, 
                                     headers=self.headers, timeout=self.timeout) as response:
                    response_data = await response.json() if response.content_length else {}
            elif method.upper() == 'POST':
                async with session.post(url, json=row_data, 
                                      headers=self.headers, timeout=self.timeout) as response:
                    response_data = await response.json() if response.content_length else {}
            elif method.upper() == 'PUT':
                async with session.put(url, json=row_data, 
                                     headers=self.headers, timeout=self.timeout) as response:
                    response_data = await response.json() if response.content_length else {}
            elif method.upper() == 'DELETE':
                async with session.delete(url, headers=self.headers, timeout=self.timeout) as response:
                    response_data = await response.json() if response.content_length else {}
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            result = {
                'row_index': row.name,
                'status_code': response.status,
                'response_data': response_data,
                'request_data': row_data,
                'success': True,
                'error': None
            }
            
            logger.info(f"Successfully processed row {row.name} (async)")
            return result
            
        except asyncio.TimeoutError:
            error_result = {
                'row_index': row.name,
                'status_code': None,
                'response_data': {},
                'request_data': row_data,
                'success': False,
                'error': 'Request timeout'
            }
            logger.error(f"Timeout error processing row {row.name}")
            return error_result
        except Exception as e:
            error_result = {
                'row_index': row.name,
                'status_code': None,
                'response_data': {},
                'request_data': row.to_dict() if 'row' in locals() else {},
                'success': False,
                'error': str(e)
            }
            logger.error(f"Error processing row {row.name} (async): {str(e)}")
            return error_result
    
    def process_dataframe_sync(self, df: pd.DataFrame, endpoint: str = '/posts', 
                              method: str = 'POST', max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Process all DataFrame rows synchronously with optional threading.
        
        Args:
            df: pandas DataFrame to process
            endpoint: API endpoint to call
            method: HTTP method
            max_workers: Number of threads for concurrent processing
            
        Returns:
            List of dictionaries containing API responses and metadata
        """
        results = []
        
        if max_workers == 1:
            # Sequential processing
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
                result = self.process_row_sync(row, endpoint, method)
                results.append(result)
        else:
            # Concurrent processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_row = {
                    executor.submit(self.process_row_sync, row, endpoint, method): (index, row)
                    for index, row in df.iterrows()
                }
                
                # Collect results
                for future in tqdm(as_completed(future_to_row), 
                                 total=len(future_to_row), desc="Processing rows"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        index, row = future_to_row[future]
                        error_result = {
                            'row_index': index,
                            'status_code': None,
                            'response_data': {},
                            'request_data': row.to_dict(),
                            'success': False,
                            'error': str(e)
                        }
                        results.append(error_result)
                        logger.error(f"Error in thread processing row {index}: {str(e)}")
        
        # Separate successful results and errors
        self.results = [r for r in results if r['success']]
        self.errors = [r for r in results if not r['success']]
        
        logger.info(f"Completed processing {len(df)} rows. "
                   f"Successful: {len(self.results)}, Errors: {len(self.errors)}")
        
        return results
    
    async def process_dataframe_async(self, df: pd.DataFrame, endpoint: str = '/posts', 
                                    method: str = 'POST', max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """
        Process all DataFrame rows asynchronously.
        
        Args:
            df: pandas DataFrame to process
            endpoint: API endpoint to call
            method: HTTP method
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of dictionaries containing API responses and metadata
        """
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(row):
                async with semaphore:
                    return await self.process_row_async(session, row, endpoint, method)
            
            # Create tasks for all rows
            tasks = [process_with_semaphore(row) for index, row in df.iterrows()]
            
            # Execute all tasks with progress bar
            results = []
            for coro in tqdm(asyncio.as_completed(tasks), 
                           total=len(tasks), desc="Processing rows (async)"):
                result = await coro
                results.append(result)
        
        # Separate successful results and errors
        self.results = [r for r in results if r['success']]
        self.errors = [r for r in results if not r['success']]
        
        logger.info(f"Completed async processing {len(df)} rows. "
                   f"Successful: {len(self.results)}, Errors: {len(self.errors)}")
        
        return results
    
    def process_dataframe_batch(self, df: pd.DataFrame, endpoint: str = '/posts', 
                               batch_size: int = 100, method: str = 'POST') -> List[Dict[str, Any]]:
        """
        Process DataFrame rows in batches (useful for APIs that support batch operations).
        
        Args:
            df: pandas DataFrame to process
            endpoint: API endpoint to call
            batch_size: Number of rows to process in each batch
            method: HTTP method
            
        Returns:
            List of dictionaries containing API responses and metadata
        """
        results = []
        
        # Split DataFrame into batches
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_df = df.iloc[i:i + batch_size]
            
            try:
                # Convert batch to list of dictionaries
                batch_data = []
                for index, row in batch_df.iterrows():
                    row_data = row.to_dict()
                    # Remove NaN values
                    row_data = {k: v for k, v in row_data.items() if pd.notna(v)}
                    row_data['_row_index'] = index  # Keep track of original row index
                    batch_data.append(row_data)
                
                url = f"{self.base_url}{endpoint}"
                logger.info(f"Making batch {method} request to {url} with {len(batch_data)} items")
                
                if method.upper() == 'POST':
                    response = requests.post(url, json=batch_data, 
                                           headers=self.headers, timeout=self.timeout)
                elif method.upper() == 'PUT':
                    response = requests.put(url, json=batch_data, 
                                          headers=self.headers, timeout=self.timeout)
                else:
                    # For GET/DELETE, process individually within the batch
                    for item in batch_data:
                        row_index = item.pop('_row_index')
                        row = df.loc[row_index]
                        result = self.process_row_sync(row, endpoint, method)
                        results.append(result)
                    continue
                
                response.raise_for_status()
                response_data = response.json() if response.content else []
                
                # Create results for each item in the batch
                for j, item in enumerate(batch_data):
                    row_index = item.pop('_row_index')
                    result = {
                        'row_index': row_index,
                        'status_code': response.status_code,
                        'response_data': response_data[j] if isinstance(response_data, list) and j < len(response_data) else response_data,
                        'request_data': item,
                        'success': True,
                        'error': None
                    }
                    results.append(result)
                
                logger.info(f"Successfully processed batch {i//batch_size + 1}")
                
            except Exception as e:
                # Handle batch errors
                for index, row in batch_df.iterrows():
                    error_result = {
                        'row_index': index,
                        'status_code': None,
                        'response_data': {},
                        'request_data': row.to_dict(),
                        'success': False,
                        'error': f"Batch error: {str(e)}"
                    }
                    results.append(error_result)
                
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
        
        # Separate successful results and errors
        self.results = [r for r in results if r['success']]
        self.errors = [r for r in results if not r['success']]
        
        logger.info(f"Completed batch processing {len(df)} rows. "
                   f"Successful: {len(self.results)}, Errors: {len(self.errors)}")
        
        return results
    
    def save_results(self, filename: str = 'api_results.json'):
        """Save processing results to a JSON file."""
        output_data = {
            'successful_results': self.results,
            'errors': self.errors,
            'summary': {
                'total_processed': len(self.results) + len(self.errors),
                'successful': len(self.results),
                'failed': len(self.errors),
                'success_rate': len(self.results) / (len(self.results) + len(self.errors)) * 100 if (len(self.results) + len(self.errors)) > 0 else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of processing results."""
        total = len(self.results) + len(self.errors)
        return {
            'total_processed': total,
            'successful': len(self.results),
            'failed': len(self.errors),
            'success_rate': len(self.results) / total * 100 if total > 0 else 0,
            'errors': [{'row_index': e['row_index'], 'error': e['error']} for e in self.errors]
        }


# Example usage functions
def example_sync_processing():
    """Example of synchronous DataFrame processing with API calls."""
    # Create sample DataFrame
    df = pd.DataFrame({
        'title': ['Post 1', 'Post 2', 'Post 3'],
        'body': ['This is post 1 content', 'This is post 2 content', 'This is post 3 content'],
        'userId': [1, 2, 3]
    })
    
    # Initialize processor
    processor = DataFrameAPIProcessor()
    
    # Process DataFrame synchronously
    results = processor.process_dataframe_sync(df, endpoint='/posts', method='POST', max_workers=3)
    
    # Print summary
    summary = processor.get_summary()
    print(f"Processing Summary: {summary}")
    
    # Save results
    processor.save_results('sync_results.json')
    
    return results


async def example_async_processing():
    """Example of asynchronous DataFrame processing with API calls."""
    # Create sample DataFrame
    df = pd.DataFrame({
        'title': [f'Async Post {i}' for i in range(1, 6)],
        'body': [f'This is async post {i} content' for i in range(1, 6)],
        'userId': [i for i in range(1, 6)]
    })
    
    # Initialize processor
    processor = DataFrameAPIProcessor()
    
    # Process DataFrame asynchronously
    results = await processor.process_dataframe_async(df, endpoint='/posts', method='POST', max_concurrent=3)
    
    # Print summary
    summary = processor.get_summary()
    print(f"Async Processing Summary: {summary}")
    
    # Save results
    processor.save_results('async_results.json')
    
    return results


def example_batch_processing():
    """Example of batch DataFrame processing with API calls."""
    # Create larger sample DataFrame
    df = pd.DataFrame({
        'title': [f'Batch Post {i}' for i in range(1, 21)],
        'body': [f'This is batch post {i} content' for i in range(1, 21)],
        'userId': [i % 5 + 1 for i in range(1, 21)]
    })
    
    # Initialize processor
    processor = DataFrameAPIProcessor()
    
    # Process DataFrame in batches
    results = processor.process_dataframe_batch(df, endpoint='/posts', batch_size=5, method='POST')
    
    # Print summary
    summary = processor.get_summary()
    print(f"Batch Processing Summary: {summary}")
    
    # Save results
    processor.save_results('batch_results.json')
    
    return results


def example_custom_processing():
    """Example of custom DataFrame processing with custom API endpoint."""
    # Load DataFrame from CSV (you can replace this with your own data)
    try:
        df = pd.read_csv('sample_data.csv')
    except FileNotFoundError:
        # Create sample data if CSV doesn't exist
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
            'age': [30, 25, 35],
            'city': ['New York', 'Los Angeles', 'Chicago']
        })
    
    # Initialize processor with custom settings
    processor = DataFrameAPIProcessor(
        base_url='https://reqres.in/api',  # Different API for demonstration
        headers={'Content-Type': 'application/json', 'User-Agent': 'DataFrameProcessor/1.0'}
    )
    
    # Process with custom endpoint
    results = processor.process_dataframe_sync(df, endpoint='/users', method='POST', max_workers=2)
    
    # Print summary
    summary = processor.get_summary()
    print(f"Custom Processing Summary: {summary}")
    
    # Save results
    processor.save_results('custom_results.json')
    
    return results


if __name__ == "__main__":
    print("DataFrame API Processor - Multiple Processing Examples")
    print("=" * 60)
    
    # Example 1: Synchronous processing
    print("\n1. Synchronous Processing Example:")
    sync_results = example_sync_processing()
    
    # Example 2: Asynchronous processing
    print("\n2. Asynchronous Processing Example:")
    async_results = asyncio.run(example_async_processing())
    
    # Example 3: Batch processing
    print("\n3. Batch Processing Example:")
    batch_results = example_batch_processing()
    
    # Example 4: Custom processing
    print("\n4. Custom Processing Example:")
    custom_results = example_custom_processing()
    
    print("\nAll examples completed! Check the generated JSON files for detailed results.")