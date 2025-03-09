#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
import re
from openai import OpenAI
from tqdm import tqdm
import yaml

def load_config(config_file: Path) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_file: Path to the config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

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

def get_content_tags(content: str, available_tags: Set[str], client: OpenAI, config: Dict) -> List[str]:
    """Use OpenAI to analyze content and select appropriate tags.
    
    Args:
        content: The markdown content to analyze
        available_tags: Set of available tags to choose from
        client: OpenAI client instance
        config: Configuration dictionary containing LLM settings
        
    Returns:
        List of selected tags
    """
    # Use configured model or fall back to gpt-3.5-turbo
    model = config['llm'].get('model', 'gpt-3.5-turbo')

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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a precise document tagging assistant. You identify relevant tags, but ONLY use exact tags from the provided list.
                    
Your task is to be thorough in selecting tags, but you must NEVER modify the tags or create variations.
Only use the exact tags provided in the list."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=config['llm'].get('temperature', 0.0),  # Use config temperature or default to 0
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
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
    client: OpenAI,
    config: Dict
) -> None:
    """Process a single markdown file, updating its tags.
    
    Args:
        input_file: Path to input markdown file
        output_file: Path to output markdown file
        available_tags: Set of available tags to choose from
        client: OpenAI client instance
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
    config_file = Path('config.yaml')
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Load configuration
    config = load_config(config_file)
    if not config.get('llm', {}).get('api_key'):
        raise ValueError("OpenAI API key not found in config.yaml")
    if not config.get('llm', {}).get('model'):
        raise ValueError("Model not specified in config.yaml")
    
    input_dir = Path('input')
    output_dir = Path('output')
    tags_file = input_dir / 'tags.json'
    
    if not tags_file.exists():
        raise FileNotFoundError(f"Tags file not found: {tags_file}")
    
    # Load available tags
    available_tags = load_available_tags(tags_file)
    if not available_tags:
        raise ValueError("No tags found in tags.json")
    print(f"Available tags: {sorted(available_tags)}")  # Debug output
    
    # Initialize OpenAI client with API key from config
    client = OpenAI(api_key=config['llm']['api_key'])
    
    # Process all markdown files
    markdown_files = list(input_dir.glob('*.md'))
    for md_file in tqdm(markdown_files, desc="Processing files"):
        output_file = output_dir / md_file.name
        process_markdown_file(md_file, output_file, available_tags, client, config)
        print(f"Processed {md_file.name}")

if __name__ == '__main__':
    main() 