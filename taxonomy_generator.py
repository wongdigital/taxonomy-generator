"""
Markdown Tag Taxonomy Generator
A tool that uses LangChain and LLMs to extract tags from Markdown files and build a taxonomy.
"""

import os
import glob
import yaml
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# LangChain imports
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from parallel_processor import parallel_process_documents


# Define Pydantic models for structured LLM outputs
class Category(BaseModel):
    """Information about a content category."""
    categoryName: str = Field(description="The name of the category")
    frequency: int = Field(description="How frequently this category appears in the content")

class Categories(BaseModel):
    """Collection of content categories."""
    categories: List[Category] = Field(description="List of content categories")

class Tag(BaseModel):
    """Information about a single tag."""
    tagName: str = Field(description="The name of the tag")
    frequency: int = Field(description="How frequently this tag appears semantically in the content")

class CategoryTags(BaseModel):
    """Tags associated with a category."""
    categoryName: str = Field(description="The name of the category")
    tags: List[str] = Field(description="List of tags for this category")

class TaxonomyOutput(BaseModel):
    """Final taxonomy output with categories and their tags."""
    categories: List[CategoryTags] = Field(description="List of categories and their associated tags")

class TagInfo(BaseModel):
    """Information about a single tag."""
    tagName: str = Field(description="The name of the tag")
    description: Optional[str] = Field(None, description="A brief description of what this tag represents")
    importance: int = Field(description="Importance score from 1-10, with 10 being most important")
    
class DocumentTags(BaseModel):
    """Tags extracted from a document."""
    tags: List[TagInfo] = Field(description="List of tags extracted from the document")
    
class TagRelationship(BaseModel):
    """Represents a relationship between two tags."""
    parent_tag: str = Field(description="The parent/broader tag")
    child_tag: str = Field(description="The child/narrower tag")
    relationship_type: str = Field(description="Type of relationship (e.g., 'broader', 'narrower', 'related')")
    confidence: float = Field(description="Confidence score for this relationship (0-1)")

class TaxonomyRelationships(BaseModel):
    """Collection of tag relationships forming a taxonomy."""
    relationships: List[TagRelationship] = Field(description="List of tag relationships")

class TagOutput(BaseModel):
    """Collection of tags with their frequencies."""
    tags: List[Tag] = Field(description="List of tags and their frequencies")


class MarkdownProcessor:
    """Process Markdown files to extract content and metadata."""
    
    def __init__(self):
        self.text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    def load_markdown_files(self, directory_path: str) -> List[Document]:
        """Load all markdown files from a directory."""
        markdown_files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
        documents = []
        
        print("Loading markdown files...")
        for file_path in tqdm(markdown_files, desc="Processing files"):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Extract front matter if present
            front_matter = {}
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    try:
                        front_matter = yaml.safe_load(parts[1])
                        content = parts[2]
                    except yaml.YAMLError:
                        pass
            
            # Create Document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    **front_matter
                }
            )
            documents.append(doc)
            
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for processing."""
        all_splits = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                all_splits.append(Document(
                    page_content=split,
                    metadata={
                        **doc.metadata,
                        "chunk": i
                    }
                ))
        return all_splits


class TagExtractor:
    """Extract tags from documents using LLMs."""
    
    def __init__(self, llm):
        self.llm = llm
        
        # Define prompt for tag extraction
        self.tag_extraction_prompt = PromptTemplate(
            input_variables=["document_content"],
            template="""
            You are an expert at analyzing content and extracting meaningful tags.
            
            Extract relevant tags from the following document content. Consider topics, concepts, technologies, 
            methodologies, and other significant themes present in the text.
            
            Focus on extracting tags that would be useful in a knowledge management system.
            
            Document content:
            ```
            {document_content}
            ```
            
            For each tag, provide:
            1. A normalized name (lowercase, consistent formatting) as 'tagName'
            2. A brief description of what this tag represents in this context
            3. An importance score from 1-10
            
            Format your response as a JSON object with a 'tags' array of objects containing 'tagName', 'description', and 'importance' fields.
            """
        )
        
        # Setup the chain using the new RunnableSequence
        self.tag_extraction_chain = (
            self.tag_extraction_prompt 
            | self.llm 
            | PydanticOutputParser(pydantic_object=DocumentTags)
        )
    
    def extract_tags_from_document(self, document: Document) -> DocumentTags:
        """Extract tags from a single document using LLM."""
        try:
            result = self.tag_extraction_chain.invoke({"document_content": document.page_content})
            return result
        except Exception as e:
            print(f"Error extracting tags: {e}")
            return DocumentTags(tags=[])
    
    def extract_tags_from_collection(self, documents: List[Document]) -> Dict[str, DocumentTags]:
        """Extract tags from a collection of documents."""
        results = {}
        for doc in documents:
            doc_id = doc.metadata.get("source", f"doc_{id(doc)}")
            results[doc_id] = self.extract_tags_from_document(doc)
        return results


class TaxonomyBuilder:
    """Build taxonomy relationships between tags."""
    
    def __init__(self, llm):
        self.llm = llm
        
        # Define prompt for relationship building
        self.relationship_prompt = PromptTemplate(
            input_variables=["tag_list"],
            template="""
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
        )
        
        # Setup the chain using the new RunnableSequence
        self.relationship_chain = (
            self.relationship_prompt 
            | self.llm 
            | PydanticOutputParser(pydantic_object=TaxonomyRelationships)
        )
    
    def build_tag_relationships(self, all_tags: List[TagInfo]) -> TaxonomyRelationships:
        """Build relationships between tags."""
        tag_list = "\n".join([f"- {tag.tagName}: {tag.description}" for tag in all_tags])
        try:
            result = self.relationship_chain.invoke({"tag_list": tag_list})
            return result
        except Exception as e:
            print(f"Error building relationships: {e}")
            return TaxonomyRelationships(relationships=[])
    
    def create_taxonomy_graph(self, relationships: TaxonomyRelationships) -> nx.DiGraph:
        """Create a directed graph representing the taxonomy."""
        G = nx.DiGraph()
        
        # Add edges for each relationship
        for rel in relationships.relationships:
            G.add_edge(
                rel.parent_tag, 
                rel.child_tag, 
                relationship_type=rel.relationship_type,
                confidence=rel.confidence
            )
        
        return G


class TaxonomyVisualizer:
    """Visualize the taxonomy graph."""
    
    @staticmethod
    def visualize_graph(graph: nx.DiGraph, output_path: str = "taxonomy_graph.png"):
        """Create a visual representation of the taxonomy graph."""
        plt.figure(figsize=(12, 8))
        
        # Use hierarchical layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="lightblue")
        nx.draw_networkx_edges(graph, pos, arrowsize=20, width=2, edge_color="gray")
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        # Add edge labels for relationship types
        edge_labels = {(u, v): data["relationship_type"] 
                      for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    @staticmethod
    def export_taxonomy(graph: nx.DiGraph, output_path: str = "taxonomy.json"):
        """Export the taxonomy in JSON format."""
        data = {
            "nodes": [{"id": node, "label": node} for node in graph.nodes()],
            "edges": [{"source": u, "target": v, "type": data["relationship_type"]} 
                     for u, v, data in graph.edges(data=True)]
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path


class MarkdownTaxonomyGenerator:
    """Main class for generating taxonomy from markdown files."""
    
    def __init__(self, llm_model_name: str = "gpt-3.5-turbo", config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm = ChatOpenAI(
            model_name=llm_model_name,
            temperature=0.2
        )
        self.markdown_processor = MarkdownProcessor()
        self.embeddings = OpenAIEmbeddings()
        
    def check_mode(self, input_dir: str) -> Tuple[str, Optional[List[Category]]]:
        """Determine if we should run in category or tag generation mode."""
        categories_path = os.path.join(input_dir, "categories.json")
        if os.path.exists(categories_path):
            with open(categories_path, 'r') as f:
                categories_data = json.load(f)
                categories = [Category(**cat) for cat in categories_data["categories"]]
            return "tag", categories
        return "category", None

    def generate_categories(self, documents: List[Document], num_categories: int) -> Categories:
        """Generate categories based on document content and count their actual frequencies."""
        # First, get initial categories from multiple content samples
        prompt = PromptTemplate(
            template="""Based on the following content samples, identify {num_categories} main categories that best represent the overall themes.
            Consider the entire scope of topics present across all documents.
            
            Important requirements:
            1. Categories should be specific enough to be meaningful but broad enough to group related content
            2. Include categories for design-related topics (e.g., Typography, UX, Product Design) if present
            3. Include categories for specific technologies or platforms (e.g., Apple, Web Development) if present
            4. Consider both technical and non-technical themes
            
            Content Samples: {content}
            
            Return the categories in JSON format matching this schema:
            {format_instructions}
            """,
            input_variables=["content", "num_categories"],
            partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Categories).get_format_instructions()}
        )
        
        chain = (
            prompt 
            | self.llm 
            | PydanticOutputParser(pydantic_object=Categories)
        )
        
        print("\nAnalyzing content and generating categories...")
        
        # Take more samples from throughout the document collection
        sample_size = 15000  # Increased sample size
        num_samples = min(10, len(documents))  # Increased number of samples
        step = max(1, len(documents) // num_samples)
        samples = []
        
        # Ensure we get a good distribution of samples
        for i in range(0, len(documents), step):
            if len(samples) < num_samples:
                doc = documents[i]
                # Skip very short documents
                if len(doc.page_content) < 100:
                    continue
                samples.append(doc.page_content[:sample_size // num_samples])
        
        combined_samples = "\n\n---\n\n".join(samples)
        
        # Get initial categories
        with tqdm(total=1, desc="Generating initial categories") as pbar:
            initial_categories = chain.invoke({
                "content": combined_samples,
                "num_categories": num_categories
            })
            pbar.update(1)
            
        # Now count actual frequencies for each category
        print("\nCounting category frequencies across all documents...")
        category_counts = {cat.categoryName: 0 for cat in initial_categories.categories}
        
        # Create better category embeddings with more context
        category_texts = [
            f"{cat.categoryName}: A category about {cat.categoryName}. This category includes content related to {cat.categoryName} "
            f"and associated topics, techniques, tools, and concepts within the domain of {cat.categoryName}."
            for cat in initial_categories.categories
        ]
        category_embeddings = self.embeddings.embed_documents(category_texts)
        
        # Process documents in batches to count frequencies
        batch_size = 10
        with tqdm(total=len(documents), desc="Analyzing documents") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_texts = [doc.page_content for doc in batch]
                
                # Get embeddings for the batch of documents
                doc_embeddings = self.embeddings.embed_documents(batch_texts)
                
                # Compare each document against each category
                for doc_idx, doc_embedding in enumerate(doc_embeddings):
                    # Track all categories that match well enough
                    matching_categories = []
                    
                    for cat_idx, cat_embedding in enumerate(category_embeddings):
                        # Compute cosine similarity
                        similarity = sum(a * b for a, b in zip(doc_embedding, cat_embedding)) / (
                            (sum(a * a for a in doc_embedding) ** 0.5) * 
                            (sum(b * b for b in cat_embedding) ** 0.5)
                        )
                        
                        # If similarity is good enough, count this category
                        if similarity > 0.2:  # Lowered threshold to catch more matches
                            category_name = initial_categories.categories[cat_idx].categoryName
                            matching_categories.append((category_name, similarity))
                    
                    # Sort by similarity and take top 2 categories if they exist
                    matching_categories.sort(key=lambda x: x[1], reverse=True)
                    for category_name, _ in matching_categories[:2]:
                        category_counts[category_name] += 1
                
                pbar.update(len(batch))
        
        # Create final categories with actual frequencies
        final_categories = [
            Category(
                categoryName=cat.categoryName,
                frequency=category_counts[cat.categoryName]
            )
            for cat in initial_categories.categories
        ]
        
        # Sort categories by frequency and filter out any with zero frequency
        final_categories = [cat for cat in final_categories if cat.frequency > 0]
        final_categories.sort(key=lambda x: x.frequency, reverse=True)
        
        return Categories(categories=final_categories)

    def generate_tags(self, documents: List[Document], categories: List[Category], num_tags_per_category: int) -> TaxonomyOutput:
        """Generate tags for each category."""
        prompt = PromptTemplate(
            template="""For the category "{category}", generate {num_tags} relevant tags based on the following content.
            The tags should be specific terms or concepts that fall under this category.
            
            Content: {content}
            
            Return the tags in a comma-separated list.
            """,
            input_variables=["category", "content", "num_tags"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        taxonomy = []
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        print("\nGenerating tags for each category...")
        for category in tqdm(categories, desc="Processing categories"):
            result = chain.invoke({
                "category": category.categoryName,
                "content": combined_content[:10000],
                "num_tags": num_tags_per_category
            })
            tags = [tag.strip() for tag in result.split(",")]
            taxonomy.append(CategoryTags(categoryName=category.categoryName, tags=tags))
        
        return TaxonomyOutput(categories=taxonomy)

    def generate_taxonomy(self, directory_path: str, output_dir: str = "./output", num_tags: Optional[int] = None):
        """Generate a taxonomy from a directory of markdown files."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process markdown files
        processor = MarkdownProcessor()
        documents = processor.load_markdown_files(directory_path)
        documents = processor.split_documents(documents)
        
        if not documents:
            raise ValueError("No markdown files found in the specified directory")

        # Initialize embeddings for semantic search
        embeddings = OpenAIEmbeddings()
        
        # Create vector store for semantic search
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        print("Analyzing content for semantic tag frequencies...")
        # Use parallel processing for document analysis with model settings
        all_tags = parallel_process_documents(
            documents=documents,
            model_name=self.llm.model_name,
            temperature=self.llm.temperature,
            batch_size=10
        )
        
        # Sort tags by frequency and take top N
        sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
        if num_tags:
            sorted_tags = sorted_tags[:num_tags]
        
        # Create output in required format
        tag_output = TagOutput(
            tags=[
                Tag(tagName=tag_name, frequency=freq)
                for tag_name, freq in sorted_tags
            ]
        )
        
        # Save tags to output/tags.json
        output_path = os.path.join(output_dir, "tags.json")
        with open(output_path, "w") as f:
            json.dump(tag_output.model_dump(), f, indent=4)
        
        print(f"Tags have been saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize the taxonomy generator
    generator = MarkdownTaxonomyGenerator()
    
    # Generate taxonomy from markdown files
    result = generator.generate_taxonomy("./markdown_files")