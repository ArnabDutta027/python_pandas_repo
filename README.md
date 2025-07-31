# DataFrame API Processor

A comprehensive Python solution for parsing pandas DataFrame rows and making API calls. This project provides multiple approaches for processing DataFrames with API integration, including synchronous, asynchronous, and batch processing methods.

## Features

- **Multiple Processing Methods**: Synchronous, asynchronous, and batch processing
- **Error Handling**: Comprehensive error handling with detailed logging
- **Progress Tracking**: Visual progress bars using tqdm
- **Flexible Configuration**: Environment variable support and customizable settings
- **Result Management**: Automatic result saving and summary generation
- **Thread-Safe**: Concurrent processing with configurable thread pools
- **Rate Limiting**: Built-in rate limiting for API calls

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
import pandas as pd
from simple_example import process_dataframe

# Create a DataFrame
df = pd.DataFrame({
    'title': ['Post 1', 'Post 2', 'Post 3'],
    'body': ['Content 1', 'Content 2', 'Content 3'],
    'userId': [1, 2, 3]
})

# Process all rows with API calls
results = process_dataframe(df)
```

### 3. Run Simple Example

```bash
python simple_example.py
```

## Advanced Usage

### Using the DataFrameAPIProcessor Class

```python
from dataframe_api_processor import DataFrameAPIProcessor
import pandas as pd

# Initialize processor
processor = DataFrameAPIProcessor(
    base_url='https://jsonplaceholder.typicode.com',
    headers={'Content-Type': 'application/json'},
    timeout=30
)

# Load your data
df = pd.read_csv('your_data.csv')

# Method 1: Synchronous processing with threading
results = processor.process_dataframe_sync(
    df, 
    endpoint='/posts', 
    method='POST', 
    max_workers=5
)

# Method 2: Asynchronous processing
import asyncio
results = asyncio.run(processor.process_dataframe_async(
    df, 
    endpoint='/posts', 
    method='POST', 
    max_concurrent=10
))

# Method 3: Batch processing
results = processor.process_dataframe_batch(
    df, 
    endpoint='/posts', 
    batch_size=100, 
    method='POST'
)

# Get processing summary
summary = processor.get_summary()
print(f"Success rate: {summary['success_rate']:.1f}%")

# Save results
processor.save_results('my_results.json')
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit the `.env` file with your settings:

```env
API_BASE_URL=https://your-api.com/v1
API_KEY=your_api_key_here
API_TIMEOUT=30
MAX_CONCURRENT_REQUESTS=10
MAX_WORKERS=5
BATCH_SIZE=100
LOG_LEVEL=INFO
```

### Custom Headers and Authentication

```python
# API Key authentication
processor = DataFrameAPIProcessor(
    base_url='https://api.example.com/v1',
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer your_token_here',
        'X-API-Key': 'your_api_key'
    }
)

# Custom timeout and settings
processor = DataFrameAPIProcessor(
    base_url='https://api.example.com',
    timeout=60,  # 60 seconds timeout
    headers={'User-Agent': 'MyApp/1.0'}
)
```

## Processing Methods Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Synchronous** | Small datasets, simple APIs | Easy to debug, reliable | Slower for large datasets |
| **Asynchronous** | Large datasets, I/O bound tasks | Fast, efficient | More complex error handling |
| **Batch** | APIs supporting batch operations | Efficient for bulk operations | Limited API support |
| **Threading** | Medium datasets, CPU-bound tasks | Good balance of speed/simplicity | GIL limitations in Python |

## Error Handling

The processor provides comprehensive error handling:

```python
# Check results
successful_results = processor.results
failed_results = processor.errors

# Print error summary
for error in processor.errors:
    print(f"Row {error['row_index']}: {error['error']}")

# Get success rate
summary = processor.get_summary()
print(f"Success rate: {summary['success_rate']:.1f}%")
```

## Examples

### Example 1: Processing CSV Data

```python
import pandas as pd
from dataframe_api_processor import DataFrameAPIProcessor

# Load CSV data
df = pd.read_csv('employee_data.csv')

# Process with custom API
processor = DataFrameAPIProcessor(
    base_url='https://company-api.com/v1'
)

results = processor.process_dataframe_sync(
    df, 
    endpoint='/employees', 
    method='POST'
)
```

### Example 2: Async Processing with Rate Limiting

```python
import asyncio
from dataframe_api_processor import DataFrameAPIProcessor

async def process_large_dataset():
    df = pd.read_csv('large_dataset.csv')
    
    processor = DataFrameAPIProcessor(
        base_url='https://api.example.com'
    )
    
    # Process with rate limiting
    results = await processor.process_dataframe_async(
        df,
        endpoint='/data',
        method='POST',
        max_concurrent=5  # Limit to 5 concurrent requests
    )
    
    return results

# Run async processing
results = asyncio.run(process_large_dataset())
```

### Example 3: Batch Processing

```python
# For APIs that support batch operations
processor = DataFrameAPIProcessor(
    base_url='https://batch-api.com'
)

results = processor.process_dataframe_batch(
    df,
    endpoint='/batch-upload',
    batch_size=50,  # Process 50 rows at a time
    method='POST'
)
```

## File Structure

```
├── dataframe_api_processor.py    # Main processor class
├── simple_example.py             # Simple usage example
├── requirements.txt              # Python dependencies
├── sample_data.csv              # Sample data for testing
├── .env.example                 # Environment variables template
└── README.md                    # This documentation
```

## Logging

The processor automatically logs all operations:

- **Console output**: Real-time progress and status
- **Log file**: Detailed logs saved to `api_processing.log`
- **JSON results**: Complete results saved to JSON files

## Performance Tips

1. **Choose the right method**:
   - Use async for I/O-bound operations
   - Use threading for mixed workloads
   - Use batch processing when supported by the API

2. **Optimize concurrency**:
   - Start with 5-10 concurrent requests
   - Monitor API rate limits
   - Adjust based on API response times

3. **Handle large datasets**:
   - Process in chunks if memory is limited
   - Use batch processing for bulk operations
   - Implement retry logic for failed requests

## Troubleshooting

### Common Issues

1. **Rate Limiting**: Reduce `max_concurrent` or `max_workers`
2. **Timeout Errors**: Increase the `timeout` parameter
3. **Memory Issues**: Process data in smaller batches
4. **Authentication**: Check API keys and headers

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
