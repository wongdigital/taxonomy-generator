# Markdown Tag Taxonomy Generator

A Python tool that uses LangChain and Large Language Models (LLMs) to analyze Markdown files and generate a semantic taxonomy with categories and tags.

## Overview

This tool helps organize collections of Markdown files (like blog posts, notes, documentation) by:

1. Generating semantic categories based on content analysis
2. Creating relevant tags for each category
3. Saving the taxonomy in structured JSON format

It uses LLMs (like GPT-3.5/4) to understand the semantic meaning of your content and create meaningful categories and tags.

## Features

- **Two-Phase Taxonomy Generation**:
  - Category Generation: Creates general categories based on content analysis
  - Tag Generation: Generates specific tags for each category
- **Smart Content Analysis**: Uses LLMs to identify relevant topics and concepts
- **Parallel Processing**: Utilizes multiple CPU cores for faster document analysis
- **Configurable Output**: Control the number of categories and tags
- **JSON Output**: Clean, structured output in JSON format
- **Interactive Mode**: Prompts for category and tag counts if not specified
- **Flexible Configuration**: Support for config files and command-line options

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/taxonomy-generator.git
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
       api_key: "your-api-key-here"
       model: "gpt-3.5-turbo"  # optional
     ```
   - Set the environment variable: `export OPENAI_API_KEY="your-api-key"`

## Usage

### Basic Usage

The tool operates in two modes:
```bash
python generate_tags.py path/to/markdown/files
# You will be prompted for the number of categories to generate
```

2. **Tag Generation Mode** (when `categories.json` exists):
```bash
python generate_tags.py path/to/markdown/files
# You will be prompted for the number of tags to generate per category
```

### Command Line Options

```
usage: generate_tags.py [-h] [--output-dir OUTPUT_DIR] [--model MODEL] [--api-key API_KEY]
              [--config CONFIG] [--num-categories NUM_CATEGORIES] 
              [--num-tags NUM_TAGS] [--verbose] input_dir

Generate a tag taxonomy from a collection of Markdown files.

positional arguments:
  input_dir             Directory containing Markdown files

optional arguments:
  -h, --help           show this help message and exit
  --output-dir, -o     Directory to save the generated taxonomy (default: ./output)
  --model, -m          LLM model to use (default: gpt-3.5-turbo)
  --api-key, -k        OpenAI API key
  --config, -c         Path to configuration file (default: config.yaml)
  --num-categories     Number of categories to generate
  --num-tags           Number of tags to generate per category
  --verbose, -v        Enable verbose output
```

### Examples

**Generate 5 categories:**
```bash
python generate_tags.py my_markdown_files/ --num-categories 5
```

**Generate 10 tags per category:**
```bash
python generate_tags.py my_markdown_files/ --num-tags 10
```

**Using a custom model:**
```bash
python generate_tags.py my_markdown_files/ --model gpt-4
```

**Using a custom config file:**
```bash
python generate_tags.py my_markdown_files/ --config custom_config.yaml
```

## Configuration

The tool can be configured through a YAML configuration file. By default, it looks for `config.yaml` in the current directory.

Example configuration file:
```yaml
# LLM settings
llm:
  api_key: "your-api-key-here"  # OpenAI API key
  model: "gpt-3.5-turbo"        # Model to use
  temperature: 0.2              # Controls randomness in responses
```

Configuration values are loaded in the following order of precedence (highest to lowest):
1. Command-line arguments
2. Config file settings
3. Environment variables
4. Built-in defaults

## Output Format

The tool generates two types of files:

1. `categories.json` (Category Generation Mode):
```json
{
    "categories": [
        {
            "categoryName": "example",
            "frequency": 5
        }
    ]
}
```

2. `taxonomy.json` (Tag Generation Mode):
```json
{
    "categories": [
        {
            "categoryName": "example",
            "tags": ["tag1", "tag2", "tag3"]
        }
    ]
}
```

## Project Structure

```
taxonomy-generator/
├── taxonomy_generator.py  # Core implementation
├── generate_tags.py                # Command-line interface
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure your API key is correct and has sufficient credits
   - Check the API key is being loaded correctly

2. **Memory Issues**:
   - For large collections, try processing files in smaller batches
   - Reduce the number of categories or tags per category

3. **Poor Quality Results**:
   - Try a different LLM model (e.g., GPT-4)
   - Ensure your content is well-structured and relevant

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [OpenAI](https://openai.com/) for GPT models

## Performance Considerations

### Parallel Processing
The tool automatically utilizes all available CPU cores to process documents in parallel, which significantly improves performance for large document collections. The parallel processing:
- Distributes document analysis across multiple CPU cores
- Processes documents in optimized batch sizes
- Maintains progress tracking while processing
- Handles errors gracefully without stopping the entire process

### Memory Usage
For large collections of documents:
- Documents are processed in batches to manage memory efficiently
- Each CPU core processes its own batch independently
- Results are aggregated incrementally to prevent memory spikes

### Optimization Tips
- Adjust the batch size using the configuration file if needed:
  ```yaml
  processing:
    batch_size: 10  # Adjust based on your system's capabilities
  ```
- For very large collections, consider processing in multiple runs
- Monitor system resources during processing to optimize settings