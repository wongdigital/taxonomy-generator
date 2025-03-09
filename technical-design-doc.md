# Technical Design Document: Markdown Tag Taxonomy Generator

## 1. Overview

This document outlines the design for a Python application that analyzes a collection of Markdown files to generate tags based on semantic content analysis. The application will process content to identify and count tag occurrences, allowing users to specify their desired number of tags. The output will be a simple JSON file containing tags and their frequencies.

## 2. System Architecture

The system is structured as a Python package with the following core components:

```
.
├── taxonomy_generator.py   # Core implementation for tag generation
├── generate_tags.py       # CLI for tag generation
└── requirements.txt     # Project dependencies
```

## 3. Detailed Component Design

### 3.1. Core Components (`taxonomy_generator.py`)

**Key Classes:**
- `MarkdownProcessor`: Handles loading and processing Markdown documents
- `TagExtractor`: Extracts tags from documents using LLMs
- `TagFrequencyCounter`: Tracks tag occurrences across documents
- Data Models:
  - `Tag`: Represents a single tag with frequency
  - `TagOutput`: Collection of tags with frequencies

### 3.2. Tag Generation (`generate_tags.py`)

**Responsibilities:**
- Provide command-line interface for tag generation
- Handle configuration loading and validation
- Set up LLM clients (OpenAI, Anthropic)
- Prompt user for desired number of tags
- Orchestrate the tag generation process

**Key Functions:**
- `setup_argparser`: Parse command-line arguments
- `get_user_tag_count`: Prompt for desired number of tags
- `get_api_key`: Handle API key management
- `get_llm_client`: Initialize appropriate LLM client

## 4. Data Flow

1. User provides input directory and desired number of tags
2. `MarkdownProcessor` loads documents
3. `TagExtractor` processes each document:
   - Uses LLM to extract relevant tags
   - Counts tag frequencies across all content
4. Results are exported as JSON file:
   - Simple list of tags with frequencies in `tags.json`
   - Format matches product requirements:
     ```json
     {
         "tagName": "example",
         "frequency": "12"
     }
     ```

## 5. Key Algorithms and Methods

### 5.1. Document Processing
- Markdown content extraction
- Full document reading as per requirements
- Simple text processing

### 5.2. Tag Extraction
- LLM-based semantic analysis
- Frequency counting across documents
- Tag normalization and deduplication
- User-specified tag count handling

## 6. Dependencies

### Core Dependencies:
- `langchain`: LLM integration and document processing
- `openai` or `anthropic`: LLM API client
- `pydantic`: Data validation and serialization
- `pyyaml`: Configuration management
- `python-dotenv`: Environment variable management

## 7. Performance Considerations

- Efficient document reading
- Memory management for large documents
- Progress tracking for long-running operations

## 8. Future Extensions

### 8.1. Enhanced Tag Generation
- Improved semantic analysis
- Better frequency counting algorithms
- Support for different content types

### 8.2. User Experience
- Interactive tag count adjustment
- Progress reporting
- Tag quality metrics

## 9. Implementation Plan

### Phase 1: Core Functionality
- Basic document processing
- LLM integration for tag extraction
- Frequency counting
- JSON output generation
- User input handling

### Phase 2: Enhancements
- Improved tag quality
- Better progress reporting
- Error handling
- Documentation

## 10. Testing Strategy

### 10.1. Unit Tests
- Test document processing
- Validate tag extraction
- Test frequency counting
- Verify JSON output format

### 10.2. Integration Tests
- Test end-to-end workflow
- Verify LLM integration
- Validate user input handling

### 10.3. Performance Tests
- Measure processing time
- Verify memory usage
- Test with various document sizes

## 11. Conclusion

This design outlines our implementation for generating tags from Markdown files, focusing on the core requirements of semantic tag extraction with frequency counting. The system is designed to be simple and effective, with room for future enhancements while maintaining focus on the primary use case.