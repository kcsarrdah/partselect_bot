"""
Unit tests for Prompt System
Tests prompt building, formatting, and statistics
"""
import pytest
from prompts import (
    PARTSELECT_SYSTEM_PROMPT,
    CONCISE_PROMPT,
    build_rag_prompt,
    build_simple_prompt,
    format_context,
    get_prompt_stats
)


class TestSystemPrompts:
    """Test system prompt definitions"""
    
    def test_main_system_prompt_exists(self):
        """TEST: Main system prompt is defined and non-empty"""
        assert PARTSELECT_SYSTEM_PROMPT is not None
        assert len(PARTSELECT_SYSTEM_PROMPT) > 0
        assert 'PartSelect' in PARTSELECT_SYSTEM_PROMPT
    
    def test_concise_prompt_exists(self):
        """TEST: Concise prompt variant exists"""
        assert CONCISE_PROMPT is not None
        assert len(CONCISE_PROMPT) > 0
    
    def test_prompts_are_strings(self):
        """TEST: All prompts are string types"""
        assert isinstance(PARTSELECT_SYSTEM_PROMPT, str)
        assert isinstance(CONCISE_PROMPT, str)


class TestContextFormatting:
    """Test context formatting functionality"""
    
    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing"""
        return [
            {
                "page_content": "Ice maker assembly for GE refrigerators",
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
                "page_content": "Check water supply and test ice maker",
                "metadata": {
                    "type": "repair",
                    "symptom": "Ice maker not working",
                    "appliance": "Refrigerator",
                    "difficulty": "Moderate",
                    "parts_needed": "PS11752778"
                }
            },
            {
                "page_content": "How to install a new refrigerator water filter",
                "metadata": {
                    "type": "blog",
                    "title": "Water Filter Installation Guide",
                    "url": "https://example.com/filter-guide"
                }
            }
        ]
    
    def test_format_context_returns_string(self, sample_docs):
        """TEST: Context formatting returns a string"""
        context = format_context(sample_docs)
        
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_format_context_includes_parts(self, sample_docs):
        """TEST: Context includes part information"""
        context = format_context(sample_docs)
        
        assert 'PS11752778' in context
        assert 'Ice Maker Assembly' in context or 'GE' in context
    
    def test_format_context_includes_repairs(self, sample_docs):
        """TEST: Context includes repair information"""
        context = format_context(sample_docs)
        
        assert 'Ice maker not working' in context or 'Refrigerator' in context
    
    def test_format_context_includes_blogs(self, sample_docs):
        """TEST: Context includes blog information"""
        context = format_context(sample_docs)
        
        assert 'Water Filter Installation Guide' in context or 'filter-guide' in context
    
    def test_format_empty_context(self):
        """TEST: Empty document list returns empty or minimal context"""
        context = format_context([])
        
        assert isinstance(context, str)


class TestRAGPromptBuilding:
    """Test RAG prompt construction"""
    
    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing"""
        return [
            {
                "page_content": "Ice maker assembly",
                "metadata": {
                    "type": "part",
                    "part_id": "PS11752778",
                    "part_name": "Ice Maker",
                    "price": "$89.99"
                }
            }
        ]
    
    def test_build_rag_prompt_returns_string(self, sample_docs):
        """TEST: RAG prompt builder returns string"""
        prompt = build_rag_prompt(
            user_query="My ice maker isn't working",
            context_docs=sample_docs
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_rag_prompt_includes_system_prompt(self, sample_docs):
        """TEST: RAG prompt includes system instructions"""
        prompt = build_rag_prompt(
            user_query="test query",
            context_docs=sample_docs
        )
        
        assert 'PartSelect' in prompt or len(prompt) > 200
    
    def test_rag_prompt_includes_context(self, sample_docs):
        """TEST: RAG prompt includes retrieved context"""
        prompt = build_rag_prompt(
            user_query="test query",
            context_docs=sample_docs
        )
        
        assert 'PS11752778' in prompt
        assert 'Ice Maker' in prompt or '$89.99' in prompt
    
    def test_rag_prompt_includes_user_query(self, sample_docs):
        """TEST: RAG prompt includes the user's question"""
        user_query = "My ice maker isn't working. What part do I need?"
        prompt = build_rag_prompt(
            user_query=user_query,
            context_docs=sample_docs
        )
        
        assert user_query in prompt
    
    def test_rag_prompt_with_custom_system_prompt(self, sample_docs):
        """TEST: Can use custom system prompt"""
        custom_system = "You are a helpful assistant."
        prompt = build_rag_prompt(
            user_query="test",
            context_docs=sample_docs,
            system_prompt=custom_system
        )
        
        assert custom_system in prompt
    
    def test_rag_prompt_with_examples(self, sample_docs):
        """TEST: Prompt can include few-shot examples"""
        prompt_with = build_rag_prompt(
            user_query="test",
            context_docs=sample_docs,
            include_examples=True
        )
        
        prompt_without = build_rag_prompt(
            user_query="test",
            context_docs=sample_docs,
            include_examples=False
        )
        
        # With examples should be longer
        assert len(prompt_with) >= len(prompt_without)


class TestSimplePromptBuilding:
    """Test simple prompt (no context) construction"""
    
    def test_build_simple_prompt_returns_string(self):
        """TEST: Simple prompt builder returns string"""
        prompt = build_simple_prompt("test query")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_simple_prompt_includes_query(self):
        """TEST: Simple prompt includes user query"""
        user_query = "How do I fix my ice maker?"
        prompt = build_simple_prompt(user_query)
        
        assert user_query in prompt
    
    def test_simple_prompt_shorter_than_rag(self):
        """TEST: Simple prompt is shorter than RAG prompt"""
        query = "test"
        simple = build_simple_prompt(query)
        rag = build_rag_prompt(
            user_query=query,
            context_docs=[{"page_content": "test", "metadata": {"type": "part"}}]
        )
        
        assert len(simple) < len(rag)


class TestPromptStatistics:
    """Test prompt statistics calculation"""
    
    def test_get_prompt_stats_returns_dict(self):
        """TEST: Stats function returns dictionary"""
        stats = get_prompt_stats("This is a test prompt.")
        
        assert isinstance(stats, dict)
    
    def test_stats_include_required_fields(self):
        """TEST: Stats include all required metrics"""
        stats = get_prompt_stats("This is a test prompt with multiple words.")
        
        assert 'total_chars' in stats
        assert 'total_words' in stats
        assert 'total_lines' in stats
        assert 'estimated_tokens' in stats
    
    def test_stats_char_count_correct(self):
        """TEST: Character count is accurate"""
        text = "Hello world"
        stats = get_prompt_stats(text)
        
        assert stats['total_chars'] == len(text)
    
    def test_stats_word_count_reasonable(self):
        """TEST: Word count is reasonable"""
        text = "one two three four five"
        stats = get_prompt_stats(text)
        
        assert stats['total_words'] == 5
    
    def test_stats_token_estimation(self):
        """TEST: Token estimation is reasonable (chars/4)"""
        text = "a" * 100  # 100 characters
        stats = get_prompt_stats(text)
        
        # Tokens should be roughly chars/4 (25 for 100 chars)
        assert 20 <= stats['estimated_tokens'] <= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
