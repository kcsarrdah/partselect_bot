"""
Prompts Package
System prompts and templates for PartSelect RAG assistant.
"""

from .system_prompts import (
    PARTSELECT_SYSTEM_PROMPT,
    CONCISE_PROMPT,
    TECHNICAL_PROMPT,
    SAFETY_RULES,
    FEW_SHOT_EXAMPLES,
    PROMPT_CONFIGS
)

from .templates import (
    build_rag_prompt,
    build_simple_prompt,
    format_context,
    format_part_info,
    format_repair_info,
    format_blog_info,
    get_prompt_stats
)

__all__ = [
    # System prompts
    'PARTSELECT_SYSTEM_PROMPT',
    'CONCISE_PROMPT',
    'TECHNICAL_PROMPT',
    'SAFETY_RULES',
    'FEW_SHOT_EXAMPLES',
    'PROMPT_CONFIGS',
    # Template functions
    'build_rag_prompt',
    'build_simple_prompt',
    'format_context',
    'format_part_info',
    'format_repair_info',
    'format_blog_info',
    'get_prompt_stats'
]

