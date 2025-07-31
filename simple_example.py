#!/usr/bin/env python3
"""
Simple example of processing pandas DataFrame rows with API calls.
This is a minimal version for quick start and understanding.
"""

import pandas as pd
import requests
import json
from typing import Dict, Any

def process_row_with_api(row: pd.Series, api_url: str = "https://jsonplaceholder.typicode.com/posts") -> Dict[str, Any]:
    """
    Process a single DataFrame row by making an API call.
    
    Args:
        row: A pandas Series representing one row of the DataFrame
        api_url: The API endpoint to call
        
    Returns:
        Dictionary with the API response and metadata
    """
    try:
        # Convert row to dictionary and remove NaN values
        row_data = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
        
        print(f"Processing row {row.name}: {row_data}")
        
        # Make API call
        response = requests.post(
            api_url,
            json=row_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = {
            'row_index': row.name,
            'input_data': row_data,
            'api_response': response.json(),
            'status_code': response.status_code,
            'success': True
        }
        
        print(f"✓ Successfully processed row {row.name}")
        return result
        
    except Exception as e:
        error_result = {
            'row_index': row.name,
            'input_data': row.to_dict(),
            'api_response': None,
            'status_code': None,
            'success': False,
            'error': str(e)
        }
        print(f"✗ Error processing row {row.name}: {str(e)}")
        return error_result

def process_dataframe(df: pd.DataFrame, api_url: str = "https://jsonplaceholder.typicode.com/posts") -> list:
    """
    Process all rows in a DataFrame by making API calls.
    
    Args:
        df: pandas DataFrame to process
        api_url: The API endpoint to call
        
    Returns:
        List of results from API calls
    """
    results = []
    
    print(f"Processing {len(df)} rows...")
    print("-" * 50)
    
    # Iterate through each row
    for index, row in df.iterrows():
        result = process_row_with_api(row, api_url)
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Total rows: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    return results

def save_results(results: list, filename: str = "results.json"):
    """Save results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filename}")

def main():
    """Main function demonstrating the usage."""
    
    # Example 1: Create sample DataFrame
    print("Example 1: Creating sample DataFrame")
    df = pd.DataFrame({
        'title': ['My First Post', 'My Second Post', 'My Third Post'],
        'body': ['This is the content of my first post', 
                'This is the content of my second post',
                'This is the content of my third post'],
        'userId': [1, 2, 1]
    })
    
    print("DataFrame:")
    print(df)
    print()
    
    # Process the DataFrame
    results = process_dataframe(df)
    
    # Save results
    save_results(results, "simple_results.json")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Load from CSV file
    print("Example 2: Loading from CSV file")
    try:
        df_csv = pd.read_csv('sample_data.csv')
        print("Loaded DataFrame from CSV:")
        print(df_csv.head())
        print()
        
        # Process first 3 rows only for demo
        df_sample = df_csv.head(3)
        results_csv = process_dataframe(df_sample)
        
        # Save results
        save_results(results_csv, "csv_results.json")
        
    except FileNotFoundError:
        print("sample_data.csv not found. Skipping CSV example.")
    
    print("\nDone! Check the generated JSON files for detailed results.")

if __name__ == "__main__":
    main()