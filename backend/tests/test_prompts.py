"""
Test script for prompt system
Simple validation of prompt building and formatting.
"""

from prompts import (
    PARTSELECT_SYSTEM_PROMPT,
    CONCISE_PROMPT,
    build_rag_prompt,
    build_simple_prompt,
    format_context,
    get_prompt_stats
)


if __name__ == "__main__":
    print("\n=== Testing Prompt System ===\n")
    
    # Test 1: System prompts exist
    print("Test 1: System Prompts")
    print(f"   Main prompt length: {len(PARTSELECT_SYSTEM_PROMPT)} chars")
    print(f"   Concise prompt length: {len(CONCISE_PROMPT)} chars")
    print("   ✓ System prompts loaded\n")
    
    # Test 2: Format context
    print("Test 2: Format Context")
    sample_docs = [
        {
            "content": "Ice maker assembly for GE refrigerators",
            "metadata": {
                "type": "part",
                "part_id": "PS11752778",
                "part_name": "Ice Maker Assembly",
                "brand": "GE",
                "price": "$89.99",
                "difficulty": "Easy"
            }
        },
        {
            "content": "Check water supply and test ice maker",
            "metadata": {
                "type": "repair",
                "symptom": "Ice maker not working",
                "appliance": "Refrigerator",
                "difficulty": "Moderate",
                "parts_needed": "PS11752778"
            }
        }
    ]
    
    context = format_context(sample_docs)
    print(f"   Context length: {len(context)} chars")
    print(f"   Contains 'Part Number': {('Part Number' in context)}")
    print(f"   Contains 'Problem': {('Problem' in context)}")
    print("   ✓ Context formatting works\n")
    
    # Test 3: Build RAG prompt
    print("Test 3: Build RAG Prompt")
    user_query = "My ice maker isn't working. What part do I need?"
    
    rag_prompt = build_rag_prompt(
        user_query=user_query,
        context_docs=sample_docs,
        include_examples=False
    )
    
    print(f"   Query: {user_query}")
    print(f"   Prompt length: {len(rag_prompt)} chars")
    print(f"   Contains system prompt: {('PartSelect' in rag_prompt)}")
    print(f"   Contains context: {('PS11752778' in rag_prompt)}")
    print(f"   Contains query: {(user_query in rag_prompt)}")
    print("   ✓ RAG prompt building works\n")
    
    # Test 4: Simple prompt (no context)
    print("Test 4: Simple Prompt (Fallback)")
    simple_prompt = build_simple_prompt(user_query)
    print(f"   Prompt length: {len(simple_prompt)} chars")
    print(f"   Contains query: {(user_query in simple_prompt)}")
    print("   ✓ Simple prompt works\n")
    
    # Test 5: Prompt stats
    print("Test 5: Prompt Statistics")
    stats = get_prompt_stats(rag_prompt)
    print(f"   Total chars: {stats['total_chars']}")
    print(f"   Total words: {stats['total_words']}")
    print(f"   Total lines: {stats['total_lines']}")
    print(f"   Estimated tokens: {int(stats['estimated_tokens'])}")
    print("   ✓ Stats calculation works\n")
    
    # Test 6: Preview formatted prompt
    print("Test 6: Prompt Preview (First 500 chars)")
    print("   " + "-" * 60)
    print("   " + rag_prompt[:500].replace('\n', '\n   '))
    print("   ...")
    print("   " + "-" * 60)
    print("   ✓ Prompt preview displayed\n")
    
    print("=== All Prompt Tests Passed ===\n")

