#!/usr/bin/env python
"""
Markdown Tag Taxonomy Generator - Command Line Interface
"""

import os
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from openai import OpenAI
from anthropic import Anthropic

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
        "output_dir",
        help="Directory to save the generated taxonomy and visualizations"
    )
    
    parser.add_argument(
        "--model", "-m",
        help="Override the model specified in config.yaml"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        help="Override the API key from config.yaml"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file"
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
    
    # Finally, check environment variable based on provider
    provider = config.get('llm', {}).get('provider', 'openai').lower()
    if provider == 'openai':
        return os.getenv("OPENAI_API_KEY")
    elif provider == 'anthropic':
        return os.getenv("ANTHROPIC_API_KEY")
    
    return None


def get_llm_client(config: Dict):
    """Get the appropriate LLM client based on configuration.
    
    Args:
        config: Configuration dictionary containing LLM settings
        
    Returns:
        LLM client instance
    """
    provider = config['llm'].get('provider', 'openai').lower()
    api_key = config['llm'].get('api_key')
    
    if not api_key:
        raise ValueError(f"API key not found in config for provider {provider}")
    
    if provider == 'openai':
        return OpenAI(api_key=api_key)
    elif provider == 'anthropic':
        return Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def main():
    """Main entry point for the CLI."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.verbose)
    
    # Override config with command line arguments if provided
    if args.model:
        if not config.get('llm'):
            config['llm'] = {}
        config['llm']['model'] = args.model
    
    # Get API key
    api_key = get_api_key(args, config)
    if not api_key:
        print("Error: API key not found. Please provide it via --api-key, config.yaml, or environment variable.")
        sys.exit(1)
    
    # Update config with API key
    if not config.get('llm'):
        config['llm'] = {}
    config['llm']['api_key'] = api_key
    
    # Validate config
    if not config.get('llm', {}).get('provider'):
        print("Error: LLM provider not specified in config.yaml")
        sys.exit(1)
    if not config.get('llm', {}).get('model'):
        print("Error: Model not specified in config.yaml or via --model")
        sys.exit(1)
    
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
    
    # Update config with number of tags
    if not config.get('taxonomy'):
        config['taxonomy'] = {}
    config['taxonomy']['max_tags'] = num_tags
    
    try:
        # Initialize and run the taxonomy generator
        generator = MarkdownTaxonomyGenerator(
            config=config,
            client=get_llm_client(config)
        )
        
        print("\nGenerating tags from markdown files...")
        tags_path = generator.generate_taxonomy(
            directory_path=args.input_dir,
            output_dir=args.output_dir
        )
        print(f"\nTags successfully generated and saved to: {tags_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()