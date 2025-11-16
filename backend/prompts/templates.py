"""
Prompt Templates for RAG Context Formatting
Handles formatting of retrieved documents into LLM prompts.
"""

from typing import List, Dict, Any
from .system_prompts import PARTSELECT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


def format_part_info(part_doc: Dict[str, Any]) -> str:
    """
    Format part document for context.
    
    Args:
        part_doc: Part document with metadata
    
    Returns:
        Formatted part information
    """
    return f"""Part Number: {part_doc.get('part_id', 'N/A')}
Name: {part_doc.get('part_name', 'N/A')}
Brand: {part_doc.get('brand', 'N/A')}
Price: {part_doc.get('price', 'N/A')}
Difficulty: {part_doc.get('difficulty', 'N/A')}
Description: {part_doc.get('page_content', '')}"""


def format_repair_info(repair_doc: Dict[str, Any]) -> str:
    """
    Format repair guide document for context.
    
    Args:
        repair_doc: Repair guide document with metadata
    
    Returns:
        Formatted repair information
    """
    return f"""Problem: {repair_doc.get('symptom', 'N/A')}
Appliance: {repair_doc.get('appliance', 'N/A')}
Difficulty: {repair_doc.get('difficulty', 'N/A')}
Parts Needed: {repair_doc.get('parts_needed', 'N/A')}
Solution: {repair_doc.get('page_content', '')}"""


def format_blog_info(blog_doc: Dict[str, Any]) -> str:
    title = blog_doc.get('title', 'N/A')
    url = blog_doc.get('url', 'N/A')
    content = blog_doc.get('page_content', '')
    
    # Format as markdown link so LLM can copy it directly
    return f"""Article: [{title}]({url})
Content: {content}"""


def format_context(documents: List[Dict[str, Any]]) -> str:
    """
    Format all retrieved documents into context string.
    
    Args:
        documents: List of retrieved documents with metadata
    
    Returns:
        Formatted context string for LLM
    """
    if not documents:
        return "No relevant context found."
    
    context_parts = ["CONTEXT INFORMATION:\n"]
    
    for i, doc in enumerate(documents, 1):
        doc_type = doc.get('metadata', {}).get('type', 'unknown')
        
        context_parts.append(f"---\n[Source {i}: {doc_type.title()}]")
        
        # Format based on document type
        if doc_type == 'part':
            formatted = format_part_info(doc.get('metadata', {}))
        elif doc_type == 'repair':
            formatted = format_repair_info(doc.get('metadata', {}))
        elif doc_type == 'blog':
            formatted = format_blog_info(doc.get('metadata', {}))
        else:
            formatted = doc.get('content', doc.get('page_content', ''))
        
        context_parts.append(formatted)
        context_parts.append("---\n")
    
    return "\n".join(context_parts)


def build_rag_prompt(
    user_query: str,
    context_docs: List[Dict[str, Any]],
    system_prompt: str = PARTSELECT_SYSTEM_PROMPT,
    include_examples: bool = False,
    include_safety: bool = True
) -> str:
    """
    Build complete prompt for RAG query.
    
    Args:
        user_query: User's question
        context_docs: Retrieved documents from vector store
        system_prompt: System prompt defining agent behavior
        include_examples: Whether to include few-shot examples
        include_safety: Whether to include safety rules
    
    Returns:
        Complete formatted prompt ready for LLM
    """
    prompt_parts = []
    
    # System prompt
    prompt_parts.append(system_prompt.strip())
    prompt_parts.append("\n")
    
    # Few-shot examples (optional)
    if include_examples:
        prompt_parts.append(FEW_SHOT_EXAMPLES.strip())
        prompt_parts.append("\n")
    
    # Context from retrieved documents
    context = format_context(context_docs)
    prompt_parts.append(context)
    prompt_parts.append("\n")
    
    # User query
    prompt_parts.append(f"USER QUESTION: {user_query}")
    prompt_parts.append("\n")
    
    # Instructions
    # Instructions
    prompt_parts.append("Please answer the user's question based on the context provided above.")
    prompt_parts.append("When citing blog articles, copy the markdown link format EXACTLY as shown in the context.")
    prompt_parts.append("For example, if context shows [Article Title](URL), include that exact same markdown link in your response.")
    prompt_parts.append("If the context doesn't contain enough information, say so.")
    
    return "\n".join(prompt_parts)


def build_simple_prompt(user_query: str, system_prompt: str = PARTSELECT_SYSTEM_PROMPT) -> str:
    """
    Build simple prompt without RAG context (fallback).
    
    Args:
        user_query: User's question
        system_prompt: System prompt
    
    Returns:
        Simple prompt without context
    """
    return f"""{system_prompt.strip()}

USER QUESTION: {user_query}

Please answer to the best of your ability. If you need specific part information, let the user know to check PartSelect.com."""


def get_prompt_stats(prompt: str) -> Dict[str, Any]:
    """
    Get statistics about a prompt.
    
    Args:
        prompt: The prompt string
    
    Returns:
        Dictionary with prompt statistics
    """
    words = prompt.split()
    lines = prompt.split('\n')
    
    return {
        "total_chars": len(prompt),
        "total_words": len(words),
        "total_lines": len(lines),
        "estimated_tokens": len(prompt) / 4  # Rough estimate: 1 token ≈ 4 chars
    }
