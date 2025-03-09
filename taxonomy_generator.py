"""
Markdown Tag Taxonomy Generator
A tool that uses LangChain and LLMs to extract tags from Markdown files and build a taxonomy.
"""

import os
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

import glob
import yaml
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Type
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import asyncio
import time
import logging

# Pydantic imports
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.runnables import RunnableSequence, RunnableConfig
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.callbacks import StreamingStdOutCallbackHandler  # Updated tracer import

from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.globals import set_debug

from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Update Cohere imports
from langchain_openai import OpenAIEmbeddings

# Configure logging - only show warnings and errors by default
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize caching
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

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
    
    def __init__(self, llm: BaseLanguageModel, config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.config = config or {}
        
        # Setup adaptive rate limiting
        self.min_delay = 0.1  # Minimum delay between requests
        self.max_delay = 5.0  # Maximum delay
        self.current_delay = self.min_delay
        self.success_threshold = 5  # Number of successful requests before reducing delay
        self.success_count = 0
        
        # Define the prompt template directly
        self.tag_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing content and extracting meaningful tags.
            Extract relevant tags from the following document content. Consider topics, concepts, technologies, 
            methodologies, and other significant themes present in the text.
            
            Focus on extracting tags that would be useful in a knowledge management system.
            
            For each tag, provide:
            1. A normalized name (lowercase, consistent formatting)
            2. A brief description of what this tag represents in this context
            3. An importance score from 1-10
            
            Format your response as a JSON object with a 'tags' array of objects containing 'tagName', 'description', and 'importance' fields."""),
            ("user", "{document_content}")
        ])
        
        # Setup the extraction chain using LCEL with error handling
        self.tag_extraction_chain = (
            self.tag_extraction_prompt 
            | self.llm.with_config({"timeout": 30})  # Add timeout
            | StrOutputParser()  # Use string output parser
        ).with_config({
            "run_name": "tag_extraction",
            "callbacks": self.config.get("callbacks", []),
            "max_concurrency": self.config.get("max_concurrency", 3),  # Reduced concurrency
            "metadata": {"task": "tag_extraction"}
        })
        
        # Setup retry configuration
        self.max_retries = self.config.get("max_retries", 5)  # Increased retries
        self.retry_delay = self.config.get("retry_delay", 2)  # Increased initial retry delay
        self.retry_count = 0  # Initialize retry counter
    
    def _reset_retry_count(self):
        """Reset the retry counter."""
        self.retry_count = 0
    
    def get_current_delay(self) -> float:
        """Get the current delay between requests."""
        return self.current_delay
    
    def _adjust_rate_limit(self, success: bool):
        """Adjust the rate limit based on success/failure."""
        if success:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.current_delay = max(self.min_delay, self.current_delay * 0.8)
                self.success_count = 0
        else:
            self.current_delay = min(self.max_delay, self.current_delay * 1.5)
            self.success_count = 0
    
    async def aextract_tags_from_document(self, document: Document) -> DocumentTags:
        """Asynchronously extract tags from a single document using LLM."""
        start_time = time.time()
        try:
            result_str = await self.tag_extraction_chain.ainvoke({
                "document_content": document.page_content,
                "metadata": document.metadata  # Pass document metadata
            })
            
            # Add error handling for JSON parsing
            try:
                # Try to parse the JSON string
                if not result_str or not result_str.strip():
                    logger.warning("Received empty response from LLM")
                    self._adjust_rate_limit(False)
                    return DocumentTags(tags=[])
                
                # Clean the response string - sometimes LLMs add markdown code block markers
                result_str = result_str.strip()
                if result_str.startswith("```json"):
                    result_str = result_str[7:]
                if result_str.startswith("```"):
                    result_str = result_str[3:]
                if result_str.endswith("```"):
                    result_str = result_str[:-3]
                result_str = result_str.strip()
                
                result_dict = json.loads(result_str)
                self._reset_retry_count()  # Reset on success
                self._adjust_rate_limit(True)  # Adjust rate limit on success
                return DocumentTags(**result_dict)
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse JSON response: {je}")
                logger.debug(f"Raw response: {result_str}")
                self._adjust_rate_limit(False)  # Adjust rate limit on failure
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    await asyncio.sleep(self.retry_delay * self.retry_count)  # Exponential backoff
                    return await self.aextract_tags_from_document(document)
                return DocumentTags(tags=[])
                
        except Exception as e:
            self._adjust_rate_limit(False)  # Adjust rate limit on failure
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                await asyncio.sleep(self.retry_delay * self.retry_count)  # Exponential backoff
                return await self.aextract_tags_from_document(document)
            logger.error(f"Error extracting tags: {e}")
            return DocumentTags(tags=[])
        finally:
            # Ensure minimum delay between requests
            elapsed = time.time() - start_time
            if elapsed < self.current_delay:
                await asyncio.sleep(self.current_delay - elapsed)
    
    def extract_tags_from_document(self, document: Document) -> DocumentTags:
        """Extract tags from a single document using LLM."""
        try:
            result = self.tag_extraction_chain.invoke(
                {"document_content": document.page_content}
            )
            return result
        except Exception as e:
            if hasattr(self, 'retry_count') and self.retry_count < self.max_retries:
                self.retry_count += 1
                time.sleep(self.retry_delay)
                return self.extract_tags_from_document(document)
            print(f"Error extracting tags: {e}")
            return DocumentTags(tags=[])
    
    async def aextract_tags_from_collection(
        self, 
        documents: List[Document]
    ) -> Dict[str, DocumentTags]:
        """Asynchronously extract tags from a collection of documents."""
        results = {}
        tasks = []
        
        # Create tasks
        for doc in documents:
            doc_id = doc.metadata.get("source", f"doc_{id(doc)}")
            task = self.aextract_tags_from_document(doc)
            tasks.append((doc_id, task))
        
        # Process tasks with progress bar
        with tqdm(total=len(tasks), desc="Extracting tags", leave=False, position=1) as pbar:
            for i, (doc_id, task) in enumerate(tasks):
                try:
                    result = await task
                    results[doc_id] = result
                    # Update progress bar with success/failure info
                    if result.tags:
                        pbar.set_postfix_str(f"Found {len(result.tags)} tags", refresh=True)
                    else:
                        pbar.set_postfix_str("No tags found", refresh=True)
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {e}")
                    results[doc_id] = DocumentTags(tags=[])
                pbar.update(1)
        
        return results
    
    def extract_tags_from_collection(
        self, 
        documents: List[Document]
    ) -> Dict[str, DocumentTags]:
        """Extract tags from a collection of documents."""
        results = {}
        for doc in tqdm(documents, desc="Extracting tags"):
            doc_id = doc.metadata.get("source", f"doc_{id(doc)}")
            results[doc_id] = self.extract_tags_from_document(doc)
        return results


class TaxonomyBuilder:
    """Build taxonomy relationships between tags."""
    
    def __init__(self, llm: BaseLanguageModel, config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.config = config or {}
        
        # Define the prompt template directly
        self.relationship_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at organizing and structuring taxonomies.
            Analyze the following list of tags and establish relationships between them to create a hierarchical taxonomy.
            
            For each relationship, determine:
            1. Which tag is broader/parent
            2. Which tag is narrower/child
            3. The type of relationship (e.g., 'broader', 'narrower', 'related')
            4. A confidence score (0-1) for this relationship
            
            Format your response as a JSON object with a 'relationships' array containing objects with 'parent_tag', 'child_tag', 'relationship_type', and 'confidence' fields."""),
            ("user", "Here are the tags to analyze:\n{tag_list}")
        ])
        
        # Setup the relationship chain using LCEL with error handling
        self.relationship_chain = (
            self.relationship_prompt 
            | self.llm.with_config({"timeout": 30})  # Add timeout
            | StrOutputParser()  # Use string output parser
        ).with_config({
            "run_name": "relationship_building",
            "callbacks": self.config.get("callbacks", []),
            "max_concurrency": self.config.get("max_concurrency", 5),
            "metadata": {"task": "relationship_building"}
        })
        
        # Setup retry configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1)
        self.retry_count = 0  # Initialize retry counter
    
    def _reset_retry_count(self):
        """Reset the retry counter."""
        self.retry_count = 0
    
    def _format_tag_list(self, all_tags: List[TagInfo]) -> str:
        """Format tags into a string for the prompt."""
        return "\n".join([
            f"- {tag.tagName}: {tag.description or 'No description available'}"
            for tag in all_tags
        ])
    
    async def abuild_tag_relationships(self, all_tags: List[TagInfo]) -> TaxonomyRelationships:
        """Asynchronously build relationships between tags."""
        try:
            result_str = await self.relationship_chain.ainvoke({
                "tag_list": self._format_tag_list(all_tags),
                "metadata": {"num_tags": len(all_tags)}
            })
            # Parse JSON string into TaxonomyRelationships
            result_dict = json.loads(result_str)
            self._reset_retry_count()  # Reset on success
            return TaxonomyRelationships(**result_dict)
        except Exception as e:
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                await asyncio.sleep(self.retry_delay * self.retry_count)  # Exponential backoff
                return await self.abuild_tag_relationships(all_tags)
            print(f"Error building relationships: {e}")
            return TaxonomyRelationships(relationships=[])
    
    def build_tag_relationships(self, all_tags: List[TagInfo]) -> TaxonomyRelationships:
        """Build relationships between tags."""
        try:
            result = self.relationship_chain.invoke({
                "tag_list": self._format_tag_list(all_tags),
                "metadata": {"num_tags": len(all_tags)}
            })
            self._reset_retry_count()  # Reset on success
            return result
        except Exception as e:
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                time.sleep(self.retry_delay * self.retry_count)  # Exponential backoff
                return self.build_tag_relationships(all_tags)
            print(f"Error building relationships: {e}")
            return TaxonomyRelationships(relationships=[])


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
    def export_taxonomy(graph: nx.DiGraph, output_path: str = "taxonomy.json", tag_frequencies: Optional[Dict[str, int]] = None):
        """Export the taxonomy in JSON format."""
        # Create nodes list with frequencies if available
        nodes = []
        for node in graph.nodes():
            node_data = {
                "id": node,
                "label": node
            }
            if tag_frequencies and node in tag_frequencies:
                node_data["frequency"] = tag_frequencies[node]
            nodes.append(node_data)
        
        # Create edges list with relationship types
        edges = [
            {
                "source": u,
                "target": v,
                "type": data["relationship_type"],
                "confidence": data.get("confidence", 1.0)
            } 
            for u, v, data in graph.edges(data=True)
        ]
        
        data = {
            "nodes": nodes,
            "edges": edges
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    @staticmethod
    def export_tags(tag_frequencies: Dict[str, int], output_path: str = "tags.json"):
        """Export the raw tags with their frequencies to a JSON file."""
        tags = [
            {
                "tagName": tag,
                "frequency": freq
            }
            for tag, freq in sorted(tag_frequencies.items(), key=lambda x: x[1], reverse=True)
        ]
        
        data = {
            "tags": tags
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path


class MarkdownTaxonomyGenerator:
    """Generate a taxonomy from a collection of Markdown files."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        client: Optional[Any] = None
    ):
        """Initialize the taxonomy generator."""
        self.config = config or {}
        
        # Set up logging
        self.verbose = self.config.get('verbose', False)
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Set up callbacks
        callbacks = []
        if self.verbose:
            callbacks.append(StreamingStdOutCallbackHandler())
        callback_manager = CallbackManager(handlers=callbacks)
        
        # Set up caching
        cache_path = self.config.get('caching', {}).get('llm_cache_path', '.langchain.db')
        set_llm_cache(SQLiteCache(database_path=cache_path))
        
        try:
            # Initialize components
            self.markdown_processor = MarkdownProcessor()
            
            # Set up LLM based on provider
            llm_config = self.config.get('llm', {})
            provider = llm_config.get('provider', 'openai').lower()
            model = llm_config.get('model', 'gpt-3.5-turbo')
            temperature = llm_config.get('temperature', 0.0)
            api_key = llm_config.get('api_key')
            
            # If the API key is an environment variable reference, resolve it
            if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                api_key = os.getenv(env_var)
            
            if not api_key:
                # Try to get from environment based on provider
                if provider == 'openai':
                    api_key = os.getenv("OPENAI_API_KEY")
                elif provider == 'anthropic':
                    api_key = os.getenv("ANTHROPIC_API_KEY")
            
            if not api_key:
                raise ValueError(f"API key not found for provider {provider}")
            
            # Set up embeddings with caching
            embeddings_cache_dir = self.config.get('caching', {}).get('embeddings_cache_dir', './cache/embeddings')
            fs = LocalFileStore(embeddings_cache_dir)
            
            # Get OpenAI API key for embeddings
            embeddings_config = self.config.get('embeddings', {})
            embeddings_api_key = embeddings_config.get('api_key')
            
            # If the API key is an environment variable reference, resolve it
            if isinstance(embeddings_api_key, str) and embeddings_api_key.startswith("${") and embeddings_api_key.endswith("}"):
                env_var = embeddings_api_key[2:-1]
                embeddings_api_key = os.getenv(env_var)
            
            if not embeddings_api_key:
                embeddings_api_key = os.getenv("OPENAI_API_KEY")
            
            if not embeddings_api_key:
                logger.error("OpenAI API key not found in config or environment variables")
                logger.error("Please either:")
                logger.error("1. Add OPENAI_API_KEY to your .env file")
                logger.error("2. Add embeddings.api_key to your config.yaml")
                raise ValueError("OpenAI API key required for embeddings")
            
            # Use OpenAI embeddings
            try:
                underlying_embeddings = OpenAIEmbeddings(
                    api_key=embeddings_api_key,
                    model="text-embedding-3-small"
                )
            except Exception as e:
                logger.error(f"Error initializing OpenAI embeddings: {e}")
                raise
            
            # Set up LLM based on provider
            if provider == 'openai':
                self.llm = ChatOpenAI(
                    api_key=api_key,
                    model_name=model,
                    temperature=temperature,
                    callbacks=callbacks
                )
            elif provider == 'anthropic':
                from langchain_anthropic import ChatAnthropic
                # Remove 'y' prefix from Anthropic API key if present
                if api_key and api_key.startswith('ysk-'):
                    api_key = 'sk-' + api_key[4:]
                self.llm = ChatAnthropic(
                    anthropic_api_key=api_key,
                    model=model,
                    temperature=temperature,
                    callbacks=callbacks,
                    max_retries=5,  # Increase max retries
                    request_timeout=60.0,  # Increase timeout
                    anthropic_api_url="https://api.anthropic.com/v1",  # Explicitly set API URL
                    max_concurrent_requests=2,  # Limit concurrent requests
                    retry_delay=10,  # Start with 10 second delay
                    exponential_backoff=True  # Use exponential backoff
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
            # Set up embeddings caching
            self.embeddings = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings,
                fs,
                namespace=underlying_embeddings.model
            )
            
            # Initialize other components with the configured LLM
            self.tag_extractor = TagExtractor(self.llm, self.config)
            self.taxonomy_builder = TaxonomyBuilder(self.llm, self.config)
            self.visualizer = TaxonomyVisualizer()
            
            # Configure processing parameters
            self.chunk_size = self.config.get('taxonomy', {}).get('chunk_size', 1000)
            self.chunk_overlap = self.config.get('taxonomy', {}).get('chunk_overlap', 200)
            self.max_tags = self.config.get('taxonomy', {}).get('max_tags', 100)
            self.min_confidence = self.config.get('taxonomy', {}).get('min_confidence', 0.6)
            
        except Exception as e:
            logger.error(f"Error initializing MarkdownTaxonomyGenerator: {e}")
            raise

    def get_model_name(self) -> str:
        """Get the model name regardless of provider."""
        if isinstance(self.llm, ChatOpenAI):
            return self.llm.model_name
        # For Anthropic
        return self.llm.model

    async def agenerate_taxonomy(
        self,
        directory_path: str,
        output_dir: str = "./output"
    ) -> str:
        """Asynchronously generate a taxonomy from a directory of markdown files."""
        try:
            # Create output directory and temp directory for batch results
            os.makedirs(output_dir, exist_ok=True)
            temp_dir = os.path.join(output_dir, "temp_batches")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Process markdown files
            documents = self.markdown_processor.load_markdown_files(directory_path)
            documents = self.markdown_processor.split_documents(documents)
            
            if not documents:
                raise ValueError("No markdown files found in the specified directory")
            
            # Extract tags from documents in batches
            batch_size = self.config.get('processing', {}).get('batch_size', 5)  # Smaller batch size
            tag_frequencies: Dict[str, int] = {}
            
            # Process documents in batches
            total_batches = (len(documents) + batch_size - 1) // batch_size
            with tqdm(total=total_batches, desc="Processing documents", position=0) as pbar:
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    
                    # Process batch
                    batch_results = await self.tag_extractor.aextract_tags_from_collection(batch)
                    
                    # Convert Pydantic models to dict before serializing
                    serializable_results = {
                        doc_id: doc_tags.model_dump()
                        for doc_id, doc_tags in batch_results.items()
                    }
                    
                    # Write batch results to temp file
                    batch_file = os.path.join(temp_dir, f"batch_{i}.json")
                    with open(batch_file, 'w') as f:
                        json.dump(serializable_results, f)
                    
                    # Update tag frequencies incrementally
                    for doc_result in batch_results.values():
                        for tag in doc_result.tags:
                            if tag.tagName in tag_frequencies:
                                tag_frequencies[tag.tagName] += 1
                            else:
                                tag_frequencies[tag.tagName] = 1
                    
                    # Count successful extractions
                    successful = sum(1 for tags in batch_results.values() if tags.tags)
                    pbar.set_postfix_str(f"Found {successful} tags", refresh=True)
                    pbar.update(1)
                    
                    # Clear batch results from memory
                    del batch_results
                    del serializable_results
                    
                    # Add adaptive delay between batches
                    if i + batch_size < len(documents):
                        delay = self.tag_extractor.get_current_delay()
                        await asyncio.sleep(delay)
            
            # Sort tags by frequency and take top N
            sorted_tags = sorted(
                tag_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.max_tags]
            
            # Export tags to JSON
            tags_path = os.path.join(output_dir, "tags.json")
            tags_data = {
                "tags": [
                    {
                        "tagName": tag_name,
                        "frequency": freq
                    }
                    for tag_name, freq in sorted_tags
                ]
            }
            
            with open(tags_path, 'w') as f:
                json.dump(tags_data, f, indent=2)
            
            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir)
            
            return tags_path
            
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            raise

    def generate_taxonomy(
        self,
        directory_path: str,
        output_dir: str = "./output"
    ) -> str:
        """Generate a taxonomy from a directory of markdown files."""
        import asyncio
        return asyncio.run(self.agenerate_taxonomy(directory_path, output_dir))

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


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Debug logging for environment variables
    logger.info("Checking environment variables...")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
    if not anthropic_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables")
    
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration."""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, using default config-template.yaml")
            config_path = "config-template.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables in config
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(v) for v in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    logger.error(f"Environment variable {env_var} not found")
                return env_value
            return obj
        
        return replace_env_vars(config)
    
    parser = argparse.ArgumentParser(description="Generate taxonomy from markdown files")
    parser.add_argument("input_dir", help="Directory containing markdown files")
    parser.add_argument("output_dir", help="Directory to save output files")
    parser.add_argument("--config", help="Path to config file", default="config.yaml")
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    
    # Initialize the taxonomy generator with config
    generator = MarkdownTaxonomyGenerator(config=config)
    
    # Generate taxonomy from markdown files
    tags_path = generator.generate_taxonomy(args.input_dir, args.output_dir)
    print(f"Generated tags at: {tags_path}")