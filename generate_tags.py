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
import yaml

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
        default="./output",
        help="Directory to save the generated taxonomy and visualizations"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-3.5-turbo",
        help="LLM model to use (e.g., gpt-3.5-turbo, gpt-4)"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        help="OpenAI API key (defaults to config.yaml, then OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--num-categories",
        type=int,
        help="Number of categories to generate (required in category generation mode)"
    )
    
    parser.add_argument(
        "--num-tags", "-n",
        type=int,
        help="Number of tags to generate"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def load_config(config_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    config: Dict[str, Any] = {}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                if verbose:
                    print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
    elif verbose:
        print(f"Config file {config_path} not found, using defaults")
    
    return config


def get_api_key(args: argparse.Namespace, config: Dict[str, Any]) -> Optional[str]:
    """Get API key from command line args, config file, or environment variable."""
    # Command line argument takes precedence
    if args.api_key:
        return args.api_key
    
    # Then check config file
    if config.get('llm', {}).get('api_key'):
        return config['llm']['api_key']
    
    # Finally, check environment variable
    return os.getenv("OPENAI_API_KEY")


def main():
    """Main entry point for the CLI."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.verbose)
    
    # Get API key
    api_key = get_api_key(args, config)
    if not api_key:
        print("Error: OpenAI API key not found. Please provide it via --api-key, config.yaml, or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Set up OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # If number of tags not provided, ask user
    num_tags = args.num_tags
    if num_tags is None:
        while True:
            try:
                num_tags = int(input("How many tags would you like to generate? "))
                if num_tags > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    
    try:
        # Initialize and run the taxonomy generator
        generator = MarkdownTaxonomyGenerator(
            llm_model_name=args.model,
            config=config
        )
        
        generator.generate_taxonomy(
            directory_path=args.input_dir,
            output_dir=args.output_dir,
            num_tags=num_tags
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()