#!/usr/bin/env python
"""
Markdown Tag Taxonomy Generator - Command Line Interface
"""

import os
import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

from taxonomy_generator import MarkdownTaxonomyGenerator


def setup_argparser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate a tag taxonomy from a collection of Markdown files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing Markdown files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="./taxonomy_output",
        help="Directory to save the generated taxonomy and visualizations"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-3.5-turbo",
        help="LLM model to use (e.g., gpt-3.5-turbo, gpt-4)"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "yaml", "graphml"],
        default="json",
        help="Format for taxonomy export"
    )
    
    return parser


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file."""
    if not config_path:
        return {}
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                return json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                import yaml
                return yaml.safe_load(f)
            else:
                print(f"Warning: Unsupported config format: {config_path.suffix}. Using defaults.")
                return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def main():
    """Main entry point for the CLI."""
    # Parse arguments
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Try to load config in this order:
    # 1. Command line specified config
    # 2. Default config.yaml in current directory
    # 3. Empty config
    config_path = args.config
    if not config_path and os.path.exists('config.yaml'):
        config_path = 'config.yaml'
        print("Using default config.yaml file")
    
    config = load_config(config_path)
    
    # Command line args take precedence over config file
    model = args.model or config.get("llm", {}).get("model")
    api_key = args.api_key or config.get("llm", {}).get("api_key") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OpenAI API key not provided. Please set it in the config file, use the --api-key option, or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize taxonomy generator with model from config or default
    try:
        generator = MarkdownTaxonomyGenerator(
            llm_model_name=model,
            config=config
        )
    except Exception as e:
        print(f"Error initializing taxonomy generator: {e}")
        sys.exit(1)
    
    # Generate taxonomy
    print(f"Analyzing Markdown files in '{args.input_dir}'...")
    
    try:
        result = generator.generate_taxonomy(
            directory_path=args.input_dir,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\nTaxonomy Generation Summary:")
        print(f"- Input directory: {args.input_dir}")
        print(f"- Output directory: {args.output_dir}")
        print(f"- Unique tags identified: {len(result['tags'])}")
        print(f"- Relationships established: {len(result['relationships'].relationships)}")
        print(f"- Graph visualization saved to: {result['visualization_path']}")
        print(f"- Taxonomy data saved to: {result['json_path']}")
        
    except Exception as e:
        print(f"Error generating taxonomy: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("\nTaxonomy generation completed successfully!")


if __name__ == "__main__":
    main()