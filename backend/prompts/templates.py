"""
Prompt Templates for RAG Context Formatting
Handles formatting of retrieved documents into LLM prompts.
Simplified based on research: few-shot examples > explicit rules (Wei et al. 2022)
"""

from typing import List, Dict, Any, Optional
from .system_prompts import PARTSELECT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


def format_part_info(part_doc: Dict[str, Any]) -> str:
    """Format part document with availability and price prominently."""
    part_id = part_doc.get('part_id', 'N/A')
    part_name = part_doc.get('part_name', 'N/A')
    price = part_doc.get('price', 'N/A')
    availability = part_doc.get('availability', 'N/A')
    brand = part_doc.get('brand', 'N/A')
    difficulty = part_doc.get('difficulty', 'N/A')
    install_time = part_doc.get('install_time', 'N/A')
    product_url = part_doc.get('product_url', 'N/A')
    install_video_url = part_doc.get('install_video_url', 'N/A')
    
    # Build formatted string with availability/price first
    info = f"""Part Number: {part_id}
Name: {part_name}
Price: ${price} | Availability: {availability}
Brand: {brand}
Installation Difficulty: {difficulty}
Installation Time: {install_time}
Product URL: {product_url}
Installation Video URL: {install_video_url}
Description: {part_doc.get('page_content', '')}"""
    
    return info


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
    """Format blog article with markdown link."""
    title = blog_doc.get('title', 'N/A')
    url = blog_doc.get('url', 'N/A')
    content = blog_doc.get('page_content', '')
    
    # Format as markdown link so LLM can copy it directly
    return f"""Article: [{title}]({url})
Content: {content}"""


def detect_conflicts(documents: List[Dict[str, Any]]) -> Optional[str]:
    """
    Detect conflicting information in retrieved docs.
    
    Args:
        documents: List of retrieved documents
    
    Returns:
        Warning string if conflicts found, None otherwise
    """
    # Check for different prices for same part
    part_prices = {}
    for doc in documents:
        if doc.get('metadata', {}).get('type') == 'part':
            part_id = doc['metadata'].get('part_id')
            price = doc['metadata'].get('price')
            if part_id and price:
                if part_id in part_prices and part_prices[part_id] != price:
                    return f"WARNING: Conflicting prices found for {part_id}"
                part_prices[part_id] = price
    
    return None


def format_context(documents: List[Dict[str, Any]], rerank: bool = True) -> str:
    """
    Format all retrieved documents into context string with relevance ordering.
    
    Args:
        documents: List of retrieved documents with metadata
        rerank: Whether to rerank by relevance score
    
    Returns:
        Formatted context string for LLM
    """
    if not documents:
        return "No relevant context found."
    
    # Rerank by relevance score if available
    if rerank and len(documents) > 0 and documents[0].get('score') is not None:
        documents = sorted(documents, 
                          key=lambda x: x.get('score', 0), 
                          reverse=True)
    
    context_parts = ["CONTEXT INFORMATION (ordered by relevance):\n"]
    
    for i, doc in enumerate(documents, 1):
        doc_type = doc.get('metadata', {}).get('type', 'unknown')
        
        # Add relevance indicator
        score = doc.get('score', 0.0)
        if score > 0:
            relevance = "HIGH" if score > 0.8 else "MEDIUM" if score > 0.5 else "LOW"
            context_parts.append(f"---\n[Source {i}: {doc_type.title()} | Relevance: {relevance}]")
        else:
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


def _get_query_guidance(analysis: Dict[str, Any]) -> str:
    """
    Get minimal query-specific guidance.
    Research shows: simple hints > lengthy instructions
    
    Args:
        analysis: Query analysis results
    
    Returns:
        Brief guidance string (or empty if not needed)
    """
    guidance_parts = []
    
    # Installation queries - add Chain-of-Thought trigger (Kojima et al. 2022)
    if analysis.get('is_installation_query'):
        guidance_parts.append("TASK: Provide installation guidance.")
        guidance_parts.append("Let's think step by step:")
        guidance_parts.append("1. What are the installation steps?")
        guidance_parts.append("2. What difficulty/time estimates are shown?")
        guidance_parts.append("3. Are there video guides available?")
        if analysis.get('part_numbers'):
            part_nums = ', '.join(analysis['part_numbers'])
            guidance_parts.append(f"Focus on: {part_nums}")
    
    # Repair queries - add Chain-of-Thought trigger
    elif analysis.get('is_repair_query'):
        # Check for complexity indicators
        query = analysis.get('query', '')
        complexity_indicators = ['and', 'also', 'multiple', 'both']
        is_complex = any(ind in query.lower() for ind in complexity_indicators)
        
        if is_complex:
            # Complex multi-step diagnostic
            guidance_parts.append("TASK: Multi-step diagnostic")
            guidance_parts.append("First, let's break this down:")
            guidance_parts.append("Subquestion 1: What are the possible causes?")
            guidance_parts.append("Subquestion 2: Which cause is most likely given the symptoms?")
            guidance_parts.append("Subquestion 3: Which parts address the most likely cause?")
            guidance_parts.append("Then provide the recommendation.")
        else:
            # Simple repair
            guidance_parts.append("TASK: Diagnose and recommend repair.")
            guidance_parts.append("Let's think step by step:")
            guidance_parts.append("1. What's the likely cause?")
            guidance_parts.append("2. Which parts fix this specific problem?")
            guidance_parts.append("3. What are the installation steps?")
    
    # Part number lookup - simple reminder
    elif analysis.get('part_numbers'):
        part_nums = ', '.join(analysis['part_numbers'])
        guidance_parts.append(f"TASK: Provide information about {part_nums}")
    
    # Brand-specific query
    elif analysis.get('brands'):
        brands = ', '.join(analysis['brands'])
        guidance_parts.append(f"TASK: Show {brands} parts only")
    
    # Stock query
    elif analysis.get('is_stock_query'):
        guidance_parts.append("TASK: Provide price and availability information")
    
    return "\n".join(guidance_parts) if guidance_parts else ""


def build_rag_prompt(
    user_query: str,
    context_docs: List[Dict[str, Any]],
    system_prompt: str = PARTSELECT_SYSTEM_PROMPT,
    include_examples: bool = True,
    query_analysis: Optional[Dict[str, Any]] = None,
    max_context_tokens: int = 4000
) -> str:
    """
    Build complete prompt for RAG query following research best practices.
    """
    prompt_parts = []
    
    # 1. System prompt (contains personality + core rules)
    prompt_parts.append(system_prompt.strip())
    prompt_parts.append("\n")
    
    # 2. Few-shot examples (CRITICAL per Wei et al. 2022)
    if include_examples:
        prompt_parts.append(FEW_SHOT_EXAMPLES.strip())
        prompt_parts.append("\n")
    
    # ✅ NEW: Check if query is about out-of-scope appliances
    out_of_scope_appliances = [
        'washing machine', 'washer', 'dryer', 'oven', 'stove', 'range', 
        'cooktop', 'microwave', 'air conditioner', 'ac unit', 'heater',
        'furnace', 'water heater', 'garbage disposal', 'trash compactor'
    ]
    
    query_lower = user_query.lower()
    detected_out_of_scope = [app for app in out_of_scope_appliances if app in query_lower]
    
    if detected_out_of_scope:
        appliance_list = ', '.join(detected_out_of_scope)
        prompt_parts.append(f"🚨 CRITICAL SCOPE VIOLATION: The user is asking about {appliance_list}.")
        prompt_parts.append("This is OUTSIDE your scope. You MUST:")
        prompt_parts.append("1. Politely decline immediately")
        prompt_parts.append("2. DO NOT use ANY information from the context below")
        prompt_parts.append("3. DO NOT mention any parts, prices, or installation videos")
        prompt_parts.append("4. ONLY say: 'I specialize in refrigerator and dishwasher parts only. For [appliance] parts, please visit PartSelect.com or contact their support team.'")
        prompt_parts.append("\n⚠️ IGNORE ALL CONTEXT BELOW - IT IS IRRELEVANT FOR OUT-OF-SCOPE QUERIES.\n")
    
    # 3. Check for conflicts in retrieved documents
    conflict_warning = detect_conflicts(context_docs)
    if conflict_warning:
        prompt_parts.append(f"⚠️ {conflict_warning}")
        prompt_parts.append("Use the most recent/reliable source.\n")
    
    # 4. Context from retrieved documents with relevance ordering
    context = format_context(context_docs, rerank=True)
    prompt_parts.append(context)
    prompt_parts.append("\n")
    
    # 5. Query-specific guidance (MINIMAL - just a hint)
    if query_analysis:
        if 'query' not in query_analysis:
            query_analysis['query'] = user_query
        
        guidance = _get_query_guidance(query_analysis)
        if guidance:
            prompt_parts.append(guidance)
            prompt_parts.append("\n")
    
    # 6. User query
    prompt_parts.append(f"USER QUESTION: {user_query}")
    prompt_parts.append("\n")
    
    # 7. Final instruction with source attribution requirement
    if detected_out_of_scope:
        # ✅ CRITICAL: When out-of-scope, completely ignore context
        prompt_parts.append(
            "🚨 CRITICAL INSTRUCTION: The user is asking about an appliance OUTSIDE your scope."
        )
        prompt_parts.append(
            "DO NOT use any information from the context above - it is irrelevant."
        )
        prompt_parts.append(
            "You MUST ONLY respond with: 'I specialize in refrigerator and dishwasher parts only. "
            "For [appliance] parts, please visit PartSelect.com or contact their support team for assistance.'"
        )
        prompt_parts.append(
            "DO NOT mention any parts, prices, or information from the context."
        )
    else:
        prompt_parts.append(
            "Answer based on the context above. "
            "Follow the examples shown. "
            "Cite sources using [Source N] when making factual claims."
        )
    
    # ✅ REMOVED: The duplicate reminder at the end - it's already in the instruction above
    
    # Check token budget
    full_prompt = "\n".join(prompt_parts)
    estimated_tokens = len(full_prompt) / 4
    
    if estimated_tokens > max_context_tokens:
        import logging
        logging.warning(f"Prompt exceeds token budget ({estimated_tokens:.0f} > {max_context_tokens}). Consider reducing k.")
    
    return full_prompt


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