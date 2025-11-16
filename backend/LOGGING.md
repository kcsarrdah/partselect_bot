# Logging & Monitoring

## Overview

The PartSelect RAG backend now includes comprehensive structured logging throughout the pipeline to track request flow, debug issues, and monitor performance.

## Features

### 1. Colored Console Output
- ✅ **INFO** (Green): Normal operations
- ⚠️  **WARNING** (Yellow): Potential issues
- ✗ **ERROR** (Red): Failures
- 🔍 **DEBUG** (Cyan): Detailed debugging

### 2. Pipeline Step Tracking

Each RAG query logs 4 distinct steps:

```
=================================================================
STEP 1: Retrieving context
=================================================================
🔍 Searching for: 'dishwasher not working' (k=5)
✓ Retrieved 5 documents

=================================================================
STEP 2: Building prompt
=================================================================
📊 Prompt size: 2150 chars

=================================================================
STEP 3: Generating response
=================================================================
✓ Response generated

=================================================================
STEP 4: Extracting sources
=================================================================
✓ Extracted 5 sources

=================================================================
✓ Query Complete (4.28s, 747 tokens)
=================================================================
```

### 3. What's Logged

#### API Level (`main.py`)
- Service initialization
- Request receipt
- Response times
- Error handling

#### RAG Service (`rag_service.py`)
- Query initiation
- Context retrieval (with k value)
- Prompt building (size metrics)
- LLM generation
- Source extraction
- Total completion time & tokens used

#### Pipeline Tracking
- Step-by-step validation
- Success/failure indicators
- Performance metrics

## Usage

### Default Logging
All services automatically log to console with INFO level:

```python
from utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Normal operation")
```

### Log Levels

```python
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical failure")
```

### Helper Functions

```python
from utils.logger import (
    log_success,      # ✓ Success message
    log_error,        # ✗ Error message
    log_warning,      # ⚠️  Warning message
    log_metric,       # 📊 Metric/stat
    log_pipeline_step # Step header with borders
)

log_success(logger, "Document loaded")
log_metric(logger, "Documents", 224)
log_pipeline_step(logger, 1, "Loading data")
```

## Configuration

Logging is configured in `utils/logger.py`:
- **Format**: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`
- **Date Format**: `%Y-%m-%d %H:%M:%S`
- **Output**: Console (stdout)
- **Colors**: Terminal ANSI codes

## Benefits

1. **Debugging**: Quickly identify where in the pipeline issues occur
2. **Performance**: Track response times and token usage
3. **Monitoring**: Validate each step of the RAG workflow
4. **Production Ready**: Structured logs can be parsed by log aggregators

## Future Enhancements

- File logging (rotating logs)
- Log aggregation (e.g., ELK stack, Datadog)
- Request ID tracking across services
- Metrics dashboard integration
- Alert thresholds for errors/latency

