"""
Parallel processing utilities for the Markdown Tag Taxonomy Generator.
"""

from typing import List, Dict, Callable, Any
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

def create_tag_extractor(model_name: str = "gpt-3.5-turbo", temperature: float = 0.2):
    """
    Create a new tag extractor instance for each process.
    This ensures each process has its own LLM instance.
    """
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature
    )
    
    # Define prompt for tag extraction
    tag_extraction_prompt = PromptTemplate(
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
    
    # Setup the chain
    from taxonomy_generator import DocumentTags  # Import here to avoid circular import
    tag_extraction_chain = (
        tag_extraction_prompt 
        | llm 
        | PydanticOutputParser(pydantic_object=DocumentTags)
    )
    
    return tag_extraction_chain

def process_document_batch(args: tuple) -> Dict[str, int]:
    """
    Process a batch of documents in parallel.
    Creates a new tag extractor for each process.
    
    Args:
        args: Tuple containing (documents, model_name, temperature)
        
    Returns:
        Dict mapping tag names to their frequencies
    """
    documents, model_name, temperature = args
    tag_frequencies = {}
    
    # Create a new tag extractor for this process
    tag_extraction_chain = create_tag_extractor(model_name, temperature)
    
    for doc in documents:
        try:
            result = tag_extraction_chain.invoke({"document_content": doc.page_content})
            for tag in result.tags:
                if tag.tagName in tag_frequencies:
                    tag_frequencies[tag.tagName] += 1
                else:
                    tag_frequencies[tag.tagName] = 1
        except Exception as e:
            print(f"Error processing document: {e}")
            continue
            
    return tag_frequencies

def parallel_process_documents(
    documents: List[Document],
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    batch_size: int = 10,
    num_processes: int = None
) -> Dict[str, int]:
    """
    Process documents in parallel using multiple CPU cores.
    
    Args:
        documents: List of documents to process
        model_name: Name of the LLM model to use
        temperature: Temperature setting for the LLM
        batch_size: Number of documents to process in each batch
        num_processes: Number of parallel processes to use (defaults to CPU count)
        
    Returns:
        Dict mapping tag names to their frequencies
    """
    if num_processes is None:
        num_processes = cpu_count()
        
    # Split documents into batches
    doc_batches = [
        documents[i:i + batch_size] 
        for i in range(0, len(documents), batch_size)
    ]
    
    # Prepare arguments for parallel processing
    process_args = [(batch, model_name, temperature) for batch in doc_batches]
    
    # Process batches in parallel
    tag_frequencies = {}
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_document_batch, process_args),
            total=len(doc_batches),
            desc="Processing documents in parallel"
        ))
        
        # Combine results from all batches
        for batch_result in results:
            for tag_name, freq in batch_result.items():
                if tag_name in tag_frequencies:
                    tag_frequencies[tag_name] += freq
                else:
                    tag_frequencies[tag_name] = freq
                    
    return tag_frequencies 