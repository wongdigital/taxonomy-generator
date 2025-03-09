"""
Markdown Tag Taxonomy Generator
A tool that uses LangChain and LLMs to extract tags from Markdown files and build a taxonomy.
"""

import os
import glob
import yaml
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# LangChain imports
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.retrievers import BaseRetriever


# Define Pydantic models for structured LLM outputs
class TagInfo(BaseModel):
    """Information about a single tag."""
    name: str = Field(description="The normalized name of the tag")
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


class MarkdownProcessor:
    """Process Markdown files to extract content and metadata."""
    
    def __init__(self):
        self.text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    def load_markdown_files(self, directory_path: str) -> List[Document]:
        """Load all markdown files from a directory."""
        markdown_files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
        documents = []
        
        for file_path in markdown_files:
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
            1. A normalized name (lowercase, consistent formatting)
            2. A brief description of what this tag represents in this context
            3. An importance score from 1-10
            
            Format your response as a JSON object with a 'tags' array of objects containing 'name', 'description', and 'importance' fields.
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
        tag_list = "\n".join([f"- {tag.name}: {tag.description}" for tag in all_tags])
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
    """Main class to orchestrate the taxonomy generation process."""
    
    def __init__(self, llm_model_name: str = "gpt-3.5-turbo", config: Optional[Dict[str, Any]] = None):
        # Load config or use defaults
        self.config = config or {}
        
        # Get LLM settings from config
        llm_config = self.config.get("llm", {})
        temperature = llm_config.get("temperature", 0.2)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=llm_model_name,
            temperature=temperature
        )
        
        # Initialize components with config
        doc_config = self.config.get("document", {})
        chunk_size = doc_config.get("chunk_size", 1000)
        chunk_overlap = doc_config.get("chunk_overlap", 200)
        
        self.markdown_processor = MarkdownProcessor()
        self.markdown_processor.text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.tag_extractor = TagExtractor(self.llm)
        self.taxonomy_builder = TaxonomyBuilder(self.llm)
        self.visualizer = TaxonomyVisualizer()
    
    def generate_taxonomy(self, directory_path: str, output_dir: str = "./output"):
        """Generate a taxonomy from Markdown files in a directory."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load markdown files
        print("Loading markdown files...")
        documents = self.markdown_processor.load_markdown_files(directory_path)
        print(f"Loaded {len(documents)} markdown files.")
        
        # Step 2: Extract tags from documents
        print("Extracting tags from documents...")
        tag_results = self.tag_extractor.extract_tags_from_collection(documents)
        
        # Step 3: Consolidate all unique tags
        all_tags = []
        tag_names = set()
        for doc_id, doc_tags in tag_results.items():
            for tag in doc_tags.tags:
                if tag.name not in tag_names:
                    all_tags.append(tag)
                    tag_names.add(tag.name)
        
        print(f"Extracted {len(all_tags)} unique tags.")
        
        # Step 4: Build relationships between tags
        print("Building tag relationships...")
        relationships = self.taxonomy_builder.build_tag_relationships(all_tags)
        
        # Step 5: Create taxonomy graph
        print("Creating taxonomy graph...")
        taxonomy_graph = self.taxonomy_builder.create_taxonomy_graph(relationships)
        
        # Step 6: Visualize and export taxonomy
        print("Visualizing taxonomy...")
        graph_path = self.visualizer.visualize_graph(
            taxonomy_graph, 
            output_path=os.path.join(output_dir, "taxonomy_graph.png")
        )
        
        json_path = self.visualizer.export_taxonomy(
            taxonomy_graph,
            output_path=os.path.join(output_dir, "taxonomy.json")
        )
        
        print(f"Taxonomy generation complete!")
        print(f"- Graph visualization: {graph_path}")
        print(f"- JSON export: {json_path}")
        
        return {
            "graph": taxonomy_graph,
            "visualization_path": graph_path,
            "json_path": json_path,
            "tags": all_tags,
            "relationships": relationships
        }


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize the taxonomy generator
    generator = MarkdownTaxonomyGenerator()
    
    # Generate taxonomy from markdown files
    result = generator.generate_taxonomy("./markdown_files")