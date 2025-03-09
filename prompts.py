"""
Prompt templates for LLM interactions in the Markdown Tag Taxonomy Generator.
"""

# Tag extraction prompt
TAG_EXTRACTION_PROMPT = """
You are an expert at analyzing content and extracting meaningful tags.

Extract relevant tags from the following document content. Consider topics, concepts, technologies, 
methodologies, and other significant themes present in the text.

Focus on extracting tags that would be useful in a knowledge management system.

Document content:
```
{document_content}
```

For each tag, provide:
1. A normalized name (lowercase, consistent formatting)
2. A brief description of what this tag represents in this context
3. An importance score from 1-10

Format your response as a JSON object with a 'tags' array of objects containing 'name', 'description', and 'importance' fields.
"""

# Relationship building prompt
RELATIONSHIP_PROMPT = """
You are an expert at organizing knowledge and building taxonomies.

I have a collection of tags extracted from a set of documents. I need you to analyze these tags 
and identify hierarchical and associative relationships between them to build a taxonomy.

Here are the tags:
{tag_list}

For each pair of related tags, specify:
1. The parent tag (broader concept)
2. The child tag (narrower concept)
3. The type of relationship ('broader', 'narrower', or 'related')
4. A confidence score between 0 and 1

Only include relationships that are meaningful and have a confidence score of at least 0.6.

Format your response as a JSON object with a 'relationships' array of objects containing 'parent_tag', 'child_tag', 'relationship_type', and 'confidence' fields.
"""

# Tag normalization prompt
TAG_NORMALIZATION_PROMPT = """
You are an expert at normalizing and consolidating taxonomies and controlled vocabularies.

I have a collection of tags that may have duplicates, synonyms, or variations in formatting.
I need you to normalize these tags into a consistent, deduplicated set.

Here are the tags to normalize:
{raw_tags}

For each set of equivalent tags, provide:
1. A preferred/canonical tag name (lowercase, consistent formatting)
2. A list of variant forms that should be mapped to this canonical form
3. A confidence score for each mapping (0-1)

Format your response as a JSON object with a 'normalized_tags' array of objects containing 'canonical', 'variants', and 'confidence_scores' fields.
"""

# Tag description enrichment prompt
TAG_DESCRIPTION_PROMPT = """
You are an expert at creating rich, informative descriptions for taxonomy terms.

I have a set of tags from a knowledge management system, and I need you to provide 
detailed descriptions for each tag to help users understand their meaning and relevance.

Here are the tags to describe:
{tags}

For each tag, provide:
1. A concise but informative description (2-3 sentences)
2. Key characteristics or attributes associated with this concept
3. How this concept relates to other areas of knowledge (if applicable)
4. Example contexts where this tag would be relevant

Format your response as a JSON object with a 'tag_descriptions' array of objects containing 'tag_name' and 'description' fields.
"""

# Taxonomy validation prompt
TAXONOMY_VALIDATION_PROMPT = """
You are an expert at evaluating and validating knowledge taxonomies.

I have generated a taxonomy of tags and relationships. I need you to analyze this taxonomy 
for logical consistency, completeness, and overall quality.

Here is the taxonomy to validate:
{taxonomy_data}

Please evaluate the taxonomy on the following criteria:
1. Logical consistency (no contradictory relationships)
2. Completeness (no obvious missing relationships)
3. Balance (appropriate breadth and depth)
4. Clarity of relationships
5. Usefulness for navigation and discovery

Identify any issues, inconsistencies, or opportunities for improvement.
Provide specific recommendations for addressing these issues.

Format your response as a JSON object with an 'evaluation' object containing 'score' (0-10), 
'issues', 'strengths', and 'recommendations' fields.
"""