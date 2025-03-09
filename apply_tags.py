#!/usr/bin/env python3
"""
Markdown Tag Applier - Command Line Interface
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Any
import re
from openai import OpenAI
from anthropic import Anthropic
from tqdm import tqdm
import yaml
import argparse
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_argparser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Apply tags to markdown files using LLM analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing markdown files and tags.json"
    )
    
    parser.add_argument(
        "output_dir",
        help="Directory to save processed markdown files"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file"
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
                
                # Handle environment variable substitution
                def replace_env_vars(obj: Any) -> Any:
                    if isinstance(obj, dict):
                        return {k: replace_env_vars(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [replace_env_vars(v) for v in obj]
                    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                        env_var = obj[2:-1]
                        env_value = os.getenv(env_var)
                        if env_value is None and verbose:
                            print(f"Warning: Environment variable {env_var} not found")
                        return env_value
                    return obj
                
                config = replace_env_vars(config)
                
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

def load_available_tags(tags_file: Path) -> Set[str]:
    """Load available tags from a JSON file.
    
    Args:
        tags_file: Path to the JSON file containing available tags
        
    Returns:
        Set of available tags
    """
    with open(tags_file, 'r', encoding='utf-8') as f:
        tags_data = json.load(f)
        # Extract just the tagName from each tag object
        return {tag['tagName'] for tag in tags_data['tags']}

def extract_frontmatter(content: str) -> tuple[Optional[str], str]:
    """Extract YAML frontmatter from markdown content.
    
    Args:
        content: The markdown content
        
    Returns:
        Tuple of (frontmatter, remaining_content)
    """
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1), match.group(2)
    return None, content

def update_frontmatter_tags(frontmatter: Optional[str], tags: List[str]) -> str:
    """Update or add tags in frontmatter.
    
    Args:
        frontmatter: Existing frontmatter content or None
        tags: List of tags to apply
        
    Returns:
        Updated frontmatter content
    """
    if frontmatter is None:
        return f"tags:\n  - " + "\n  - ".join(tags)
    
    # Remove existing tags section if present
    lines = frontmatter.split('\n')
    new_lines = []
    skip_mode = False
    
    for line in lines:
        if line.strip().startswith('tags:'):
            skip_mode = True
            continue
        if skip_mode and line.strip().startswith('-'):
            continue
        if skip_mode and not line.strip().startswith('-'):
            skip_mode = False
        if not skip_mode:
            new_lines.append(line)
    
    # Add new tags section
    new_frontmatter = '\n'.join(new_lines).strip()
    if new_frontmatter:
        new_frontmatter += '\n'
    new_frontmatter += f"tags:\n  - " + "\n  - ".join(tags)
    
    return new_frontmatter

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

def get_content_tags(content: str, available_tags: Set[str], client: Union[OpenAI, Anthropic], config: Dict) -> List[str]:
    """Use LLM to analyze content and select appropriate tags.
    
    Args:
        content: The markdown content to analyze
        available_tags: Set of available tags to choose from
        client: LLM client instance
        config: Configuration dictionary containing LLM settings
        
    Returns:
        List of selected tags
    """
    # Create a mapping of lowercase tags to their correct case
    tag_mapping = {tag.lower(): tag for tag in available_tags}

    prompt = f"""Analyze the following markdown content and select ALL appropriate tags from the available set that apply to the content.

Guidelines for tag selection:
1. Include both specific and broader category tags that are relevant (e.g., both 'web design' and 'design' if both apply)
2. Consider the main themes, topics, technologies, and concepts discussed
3. Include tags for both primary and significant secondary topics
4. Don't omit relevant broader categories just because more specific tags exist
5. Select tags based on substantial discussion, not just brief mentions
6. IMPORTANT: Only use EXACT tags from the list below - do not modify or create variations

Available tags (use these EXACT tags only):
{', '.join(sorted(available_tags))}

Content:
{content}

Return a JSON object with a 'tags' array containing ONLY tags from the provided list, like this:
{{"tags": ["tag1", "tag2", "tag3"]}}

Remember:
- Use only the exact tags provided above
- Include both specific and broader categories when relevant
- Do not modify or create variations of the tags"""

    system_prompt = """You are a precise document tagging assistant. You identify relevant tags, but ONLY use exact tags from the provided list.
                    
Your task is to be thorough in selecting tags, but you must NEVER modify the tags or create variations.
Only use the exact tags provided in the list."""

    try:
        provider = config['llm'].get('provider', 'openai').lower()
        
        if provider == 'openai':
            response = client.chat.completions.create(
                model=config['llm'].get('model', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=config['llm'].get('temperature', 0.0),
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        elif provider == 'anthropic':
            response = client.messages.create(
                model=config['llm'].get('model', 'claude-3-opus-20240229'),
                max_tokens=config['llm'].get('max_tokens', 1024),
                temperature=config['llm'].get('temperature', 0.0),
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            result = json.loads(response.content[0].text)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        selected_tags = result.get('tags', [])
        
        # Normalize and validate tags
        valid_tags = []
        for tag in selected_tags:
            # Try to match the tag exactly first
            if tag in available_tags:
                valid_tags.append(tag)
            # If no exact match, try case-insensitive matching
            elif tag.lower() in tag_mapping:
                valid_tags.append(tag_mapping[tag.lower()])
        
        if not valid_tags:
            print(f"Warning: No valid tags were selected from the available set.")
        elif len(valid_tags) != len(selected_tags):
            print(f"Warning: Some selected tags were invalid. Original: {selected_tags}, Valid: {valid_tags}")
        
        return sorted(set(valid_tags))  # Remove duplicates and sort
    except Exception as e:
        print(f"Warning: Error processing content for tags. Error: {e}")
        return []

def process_markdown_file(
    input_file: Path,
    output_file: Path,
    available_tags: Set[str],
    client: Union[OpenAI, Anthropic],
    config: Dict
) -> None:
    """Process a single markdown file, updating its tags.
    
    Args:
        input_file: Path to input markdown file
        output_file: Path to output markdown file
        available_tags: Set of available tags to choose from
        client: LLM client instance
        config: Configuration dictionary
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    frontmatter, remaining_content = extract_frontmatter(content)
    
    # Get tags based on the entire content
    selected_tags = get_content_tags(content, available_tags, client, config)
    updated_frontmatter = update_frontmatter_tags(frontmatter, selected_tags)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('---\n')
        f.write(updated_frontmatter)
        f.write('\n---\n')
        f.write(remaining_content)

def main() -> None:
    """Main function to process all markdown files."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_file = Path(args.config)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    tags_file = input_dir / 'tags.json'
    if not tags_file.exists():
        print(f"Error: Tags file not found: {tags_file}")
        sys.exit(1)
    
    try:
        # Load available tags
        available_tags = load_available_tags(tags_file)
        if not available_tags:
            print("Error: No tags found in tags.json")
            sys.exit(1)
        if args.verbose:
            print(f"Available tags: {sorted(available_tags)}")
        
        # Initialize LLM client based on config
        client = get_llm_client(config)
        
        # Process all markdown files
        markdown_files = list(input_dir.glob('*.md'))
        if not markdown_files:
            print(f"Warning: No markdown files found in {input_dir}")
            sys.exit(0)
            
        print(f"Found {len(markdown_files)} markdown files to process")
        for md_file in tqdm(markdown_files, desc="Processing files"):
            output_file = output_dir / md_file.name
            process_markdown_file(md_file, output_file, available_tags, client, config)
            if args.verbose:
                print(f"Processed {md_file.name}")
        
        print(f"\nProcessing complete. Tagged files saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 