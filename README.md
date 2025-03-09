# Markdown Tag Taxonomy Generator

A Python tool that uses LangChain and Large Language Models (LLMs) to analyze Markdown files, extract relevant tags, and build semantic relationships between them.

## Overview

This tool helps organize collections of Markdown files (like blog posts, notes, documentation) by:

1. Extracting meaningful tags from document content
2. Building relationships between tags to create a taxonomy
3. Saving the taxonomy and tag information in structured JSON format

It uses LLMs (like GPT-3.5/4 or Claude) to understand the semantic meaning of your content and create meaningful tag relationships.

## Features

- **Smart Tag Extraction**: Uses LLMs to identify relevant topics, concepts, and themes
- **Relationship Building**: Creates hierarchical and associative relationships between tags
- **Parallel Processing**: Utilizes multiple CPU cores for faster document analysis
- **Multi-Provider Support**: Works with OpenAI GPT models and Anthropic Claude
- **Adaptive Rate Limiting**: Smart handling of API rate limits
- **Configurable Output**: Control the number of tags and relationship confidence
- **JSON Output**: Clean, structured output in JSON format
- **Interactive Mode**: Prompts for tag counts if not specified
- **Flexible Configuration**: Support for config files and command-line options
- **Type-Safe Implementation**: Uses Pydantic models for robust data handling

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key or Anthropic API key

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/wongdigital/taxonomy-generator.git
   cd taxonomy-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key (in order of precedence):
   - Use the `--api-key` option when running the tool
   - Create a `config.yaml` file (recommended):
     ```yaml
     llm:
       provider: "openai"  # or "anthropic"
       api_key: "your-api-key-here"
       model: "gpt-4"  # or "claude-3-opus-20240229"
       temperature: 0.2
     ```
   - Set the environment variable: `export OPENAI_API_KEY="your-api-key"` or `export ANTHROPIC_API_KEY="your-api-key"`

## Usage

### Basic Usage

```bash
python generate_tags.py input_dir output_dir
# You will be prompted for the number of tags to generate
```

### Command Line Options

```
usage: generate_tags.py [-h] [--model MODEL] [--api-key API_KEY]
              [--config CONFIG] [--num-tags NUM_TAGS] [--verbose] 
              input_dir output_dir

Generate a tag taxonomy from a collection of Markdown files.

positional arguments:
  input_dir             Directory containing Markdown files
  output_dir            Directory to save the generated taxonomy

optional arguments:
  -h, --help           show this help message and exit
  --model, -m          LLM model to use (default: from config)
  --api-key, -k        API key for the LLM provider
  --config, -c         Path to configuration file (default: config.yaml)
  --num-tags, -n       Number of tags to generate
  --verbose, -v        Enable verbose output
```

### Examples

**Generate tags with default settings:**
```bash
python generate_tags.py my_markdown_files/ output/
```

**Generate specific number of tags:**
```bash
python generate_tags.py my_markdown_files/ output/ --num-tags 50
```

**Using a custom model:**
```bash
python generate_tags.py my_markdown_files/ output/ --model gpt-4
```

**Using a custom config file:**
```bash
python generate_tags.py my_markdown_files/ output/ --config custom_config.yaml
```

## Configuration

The tool can be configured through a YAML configuration file. By default, it looks for `config.yaml` in the current directory.

Example configuration file:
```yaml
# LLM Configuration
llm:
  provider: openai  # or anthropic
  model: gpt-4  # or claude-3-opus-20240229
  temperature: 0.2
  api_key: ${OPENAI_API_KEY}  # Use environment variable

# Processing Configuration
processing:
  batch_size: 5
  max_retries: 5
  retry_delay: 2
  max_concurrency: 3
  adaptive_rate_limiting: true

# Taxonomy Configuration
taxonomy:
  min_confidence: 0.6
  max_tags: 100
  chunk_size: 1000
  chunk_overlap: 200
```

Configuration values are loaded in the following order of precedence (highest to lowest):
1. Command-line arguments
2. Config file settings
3. Environment variables
4. Built-in defaults

## Output Format

The tool generates a `tags.json` file containing extracted tags and their frequencies:

```json
{
    "tags": [
        {
            "tagName": "example-tag",
            "frequency": 5
        }
    ]
}
```

Additionally, you can use the `apply_tags.py` script to automatically apply the generated tags to your markdown files:

```bash
python apply_tags.py input_dir output_dir [--max-tags MAX_TAGS]
```

## Project Structure

```
taxonomy-generator/
├── taxonomy_generator.py   # Core implementation
├── generate_tags.py       # Tag generation CLI
├── apply_tags.py         # Tag application CLI
├── parallel_processor.py # Parallel processing utilities
├── prompts/             # LLM prompt templates
│   ├── tag_extraction.yaml
│   └── relationship_building.yaml
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure your API key is correct and has sufficient credits
   - Check the API key is being loaded correctly
   - Verify the correct provider is specified in config

2. **Rate Limiting**:
   - The tool includes adaptive rate limiting, but you may need to adjust:
     - `max_concurrency` in config
     - `retry_delay` for failed requests
     - `batch_size` for document processing

3. **Memory Issues**:
   - Adjust `chunk_size` and `chunk_overlap` in config
   - Reduce `batch_size` for parallel processing
   - Process large collections in smaller batches

4. **Poor Quality Results**:
   - Try a more capable model (e.g., GPT-4 or Claude)
   - Adjust `temperature` in config
   - Ensure your content is well-structured

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [OpenAI](https://openai.com/) and [Anthropic](https://www.anthropic.com/) for LLM APIs
