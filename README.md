# Markdown Tag Taxonomy Generator

A Python tool that uses LangChain and Large Language Models (LLMs) to analyze Markdown files, extract meaningful tags, and generate a hierarchical taxonomy.

## Overview

This tool helps organize collections of Markdown files (like blog posts, notes, documentation) by:

1. Extracting relevant tags and topics from the content
2. Building relationships between tags (hierarchical and associative)
3. Creating a visual and interactive taxonomy
4. Exporting the taxonomy in various formats

It uses LLMs (like GPT-3.5/4) to understand the semantic meaning of your content and create a rich, meaningful taxonomy.

## Features

- **Smart Tag Extraction**: Uses LLMs to identify relevant topics and concepts
- **Hierarchical Relationships**: Builds parent-child relationships between tags
- **Frontmatter Support**: Extracts existing tags from YAML frontmatter
- **Visualization**: Creates graph visualizations of the taxonomy
- **Multiple Export Formats**: JSON, YAML, GraphML
- **Configurable**: Extensive configuration options

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (or other supported LLM provider)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/markdown-taxonomy.git
   cd markdown-taxonomy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key (in order of precedence):
   - Use the `--api-key` option when running the tool
   - Create a `config.yaml` file in the project directory (recommended)
   - Set the environment variable: `export OPENAI_API_KEY="your-api-key"`

## Usage

### Basic Usage

The tool will automatically use `config.yaml` from the current directory if it exists. Run the tool on a directory of Markdown files:

```bash
python cli.py path/to/markdown/files --output-dir taxonomy_output
```

This will:
1. Load configuration from `config.yaml` if present
2. Analyze all `.md` files in the directory (including subdirectories)
3. Extract tags using LLM
4. Build relationships between tags
5. Generate a taxonomy graph visualization
6. Export the taxonomy data as JSON

### Command Line Options

All command-line options override their corresponding values in the config file.

```
usage: cli.py [-h] [--output-dir OUTPUT_DIR] [--model MODEL] [--api-key API_KEY] [--verbose] [--config CONFIG] [--format {json,yaml,graphml}] input_dir

Generate a tag taxonomy from a collection of Markdown files.

positional arguments:
  input_dir             Directory containing Markdown files

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to save the generated taxonomy and visualizations (default: ./taxonomy_output)
  --model MODEL, -m MODEL
                        LLM model to use (e.g., gpt-3.5-turbo, gpt-4) (default: gpt-3.5-turbo)
  --api-key API_KEY, -k API_KEY
                        OpenAI API key (defaults to config file, then OPENAI_API_KEY environment variable)
  --verbose, -v         Enable verbose output (default: False)
  --config CONFIG, -c CONFIG
                        Path to custom configuration file (default: ./config.yaml)
  --format {json,yaml,graphml}, -f {json,yaml,graphml}
                        Format for taxonomy export (default: json)
```

### Examples

**Using a custom model:**
```bash
python cli.py my_markdown_files/ --model gpt-4
```

**Using a custom config file:**
```bash
python cli.py my_markdown_files/ --config custom_config.yaml
```

**Generating multiple export formats:**
```bash
python cli.py my_markdown_files/ --format yaml
```

## Configuration

The tool is configured through a YAML configuration file. By default, it looks for `config.yaml` in the current directory. You can also specify a different config file using the `--config` option.

Configuration values are loaded in the following order of precedence (highest to lowest):
1. Command-line arguments
2. Specified config file (via --config)
3. Default config.yaml in current directory
4. Built-in defaults

Example configuration file:

```yaml
# LLM settings
llm:
  provider: openai  # openai, anthropic, etc.
  api_key: "your-api-key-here"  # API key for the LLM provider
  model: gpt-3.5-turbo  # model name
  temperature: 0.2  # randomness (0-1)
  max_tokens: 2000  # max tokens per response

# Document processing settings
document:
  chunk_size: 1000  # size of document chunks
  chunk_overlap: 200  # overlap between chunks
  exclude_patterns:  # glob patterns to exclude
    - "**/node_modules/**"
    - "**/.git/**"
  frontmatter: true  # extract frontmatter metadata
  parse_links: true  # extract and analyze links

# Tag extraction settings
tag_extraction:
  min_importance: 3  # minimum importance score (1-10)
  max_tags_per_document: 15  # max tags per document
  use_existing_tags: true  # use tags from frontmatter
  consolidate_similar: true  # merge similar tags

# Taxonomy settings
taxonomy:
  min_confidence: 0.6  # minimum relationship confidence (0-1)
  max_depth: 5  # maximum hierarchy depth
  visualization:
    format: png  # png, svg, html
    node_size: 700
    font_size: 10
    layout: hierarchical  # hierarchical, circular, force
  export_formats:
    - json
    - yaml
    - graphml

# Output settings
output:
  detailed_report: true  # generate detailed report
  tag_descriptions: true  # include tag descriptions
  include_sources: true  # link tags to source documents
```

## Output

The tool generates several files in the output directory:

- `taxonomy_graph.png`: Visual representation of the taxonomy
- `taxonomy.json` (or .yaml/.graphml): Structured data of the taxonomy
- Detailed reports and other visualizations (if configured)

## Project Structure

```
markdown-taxonomy/
├── taxonomy_generator.py  # Core implementation
├── cli.py                 # Command-line interface
├── prompts.py             # LLM prompt templates
├── requirements.txt       # Dependencies
├── config.yaml            # Default configuration
└── README.md              # This file
```

## Extending the Tool

### Using Different LLM Providers

The tool is designed to work with multiple LLM providers. To use a different provider:

1. Install the required package (e.g., `pip install anthropic`)
2. Update the config file with the provider and API key
3. Update the prompt templates if necessary

### Adding Custom Analysis

You can extend the `TagExtractor` and `TaxonomyBuilder` classes to implement custom analysis logic:

```python
class CustomTagExtractor(TagExtractor):
    def __init__(self, llm):
        super().__init__(llm)
        # Add your custom initialization
        
    def extract_tags_from_document(self, document):
        # Implement your custom extraction logic
        # ...
```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure your API key is correct and has sufficient credits
   - Check the API key is being loaded correctly

2. **Memory Issues**:
   - For large collections, adjust the chunk size in config
   - Process files in smaller batches

3. **Poor Quality Tags**:
   - Try a different LLM model
   - Adjust the prompts in `prompts.py`
   - Increase the temperature for more diverse tags

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [NetworkX](https://networkx.org/) for graph visualization
- [OpenAI](https://openai.com/) for GPT models