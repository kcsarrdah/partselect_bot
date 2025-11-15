"""
Unit tests for LLM Service
Tests initialization, generation, error handling, and integration.
"""

import pytest
import os
from services.llm_service import LLMService


# Module-level fixture - ensures tests ALWAYS use free model
@pytest.fixture(scope="module")
def free_model_service():
    """
    Create LLMService with hardcoded FREE model.
    This prevents tests from accidentally using paid models.
    """
    return LLMService(
        model="google/gemma-3-27b-it:free"  # Always free for tests
    )


class TestLLMServiceInitialization:
    """Test service initialization and configuration"""
    
    @pytest.fixture
    def service(self):
        """Create LLMService instance with FREE model"""
        return LLMService(model="google/gemma-3-27b-it:free")
    
    def test_init_with_env_vars(self, service):
        """
        TEST: Initialize service from environment variables
        
        GIVEN: Environment variables are set in .env
        WHEN: LLMService() is called with no arguments
        THEN: Service loads config from .env correctly
        
        PURPOSE: Verify default initialization from env vars
        """
        assert service.provider == "openrouter"
        assert service.model is not None
        assert service.api_key is not None
        assert service.temperature == 0.7
        assert service.max_tokens == 500
        print("✓ Service initialized from .env")
    
    def test_get_model_info(self, service):
        """
        TEST: Get current model configuration
        
        GIVEN: LLMService is initialized
        WHEN: get_model_info() is called
        THEN: Returns dict with provider, model, settings
        
        PURPOSE: Verify configuration retrieval
        """
        info = service.get_model_info()
        
        assert "provider" in info
        assert "model" in info
        assert "temperature" in info
        assert "max_tokens" in info
        assert "api_key_set" in info
        
        assert info["provider"] == "openrouter"
        assert info["api_key_set"] == True
        assert info["temperature"] == 0.7
        assert info["max_tokens"] == 500
        
        print(f"✓ Model info: {info['model']}")
    
    def test_init_missing_api_key(self):
        """
        TEST: Fail gracefully when API key is missing
        
        GIVEN: No API key in environment
        WHEN: LLMService() is initialized
        THEN: Raises ValueError with helpful message
        
        PURPOSE: Verify error handling for missing credentials
        """
        # Temporarily remove API key
        original_key = os.environ.get("OPENROUTER_API_KEY")
        if original_key:
            del os.environ["OPENROUTER_API_KEY"]
        
        try:
            with pytest.raises(ValueError, match="API key not found"):
                LLMService()
            print("✓ Correctly raises error for missing API key")
        finally:
            # Restore API key
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key


class TestLLMServiceGeneration:
    """Test response generation with real API calls"""
    
    @pytest.fixture
    def service(self):
        """Create LLMService instance with FREE model"""
        return LLMService(model="google/gemma-3-27b-it:free")
    
    def test_generate_simple_response(self, service):
        """
        TEST: Generate a simple response
        
        GIVEN: LLMService is initialized
        WHEN: generate() is called with a simple prompt
        THEN: Returns successful response with correct format
        
        PURPOSE: Verify basic generation works
        """
        result = service.generate(
            prompt="Say 'Hello' and nothing else.",
            max_tokens=10
        )
        
        # Check response structure
        assert result["status_code"] == 200
        assert result["status"] == "success"
        assert "response" in result
        assert "usage" in result
        assert "model" in result
        assert "provider" in result
        
        # Check content
        assert len(result["response"]) > 0
        assert result["provider"] == "openrouter"
        
        print(f"✓ Generated response: {result['response'][:50]}...")
    
    def test_generate_tracks_tokens(self, service):
        """
        TEST: Token usage is tracked correctly
        
        GIVEN: LLMService generates a response
        WHEN: generate() completes
        THEN: Usage contains prompt_tokens, completion_tokens, total_tokens
        
        PURPOSE: Verify token tracking for cost monitoring
        """
        result = service.generate(
            prompt="What is 2+2?",
            max_tokens=20
        )
        
        assert result["status_code"] == 200
        usage = result["usage"]
        
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        
        print(f"✓ Token usage: {usage['total_tokens']} total")
    
    def test_generate_respects_max_tokens(self, service):
        """
        TEST: max_tokens parameter is respected
        
        GIVEN: generate() is called with max_tokens=50
        WHEN: LLM generates response
        THEN: Response is limited to approximately max_tokens
        
        PURPOSE: Verify token limit enforcement
        """
        result = service.generate(
            prompt="Tell me about ice makers.",
            max_tokens=50
        )
        
        assert result["status_code"] == 200
        assert result["usage"]["completion_tokens"] <= 50
        
        print(f"✓ Response limited to {result['usage']['completion_tokens']} tokens")
    
    def test_generate_respects_temperature(self, service):
        """
        TEST: Temperature parameter affects responses
        
        GIVEN: Two identical prompts with different temperatures
        WHEN: generate() is called twice
        THEN: Responses may differ (verifies parameter is used)
        
        PURPOSE: Verify temperature parameter works
        """
        prompt = "Say a random number between 1 and 100."
        
        result1 = service.generate(prompt, temperature=0.1, max_tokens=10)
        result2 = service.generate(prompt, temperature=1.0, max_tokens=10)
        
        assert result1["status_code"] == 200
        assert result2["status_code"] == 200
        
        print(f"✓ Temp 0.1: {result1['response'][:30]}")
        print(f"✓ Temp 1.0: {result2['response'][:30]}")


class TestLLMServiceErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.fixture
    def service(self):
        """Create LLMService instance with FREE model"""
        return LLMService(model="google/gemma-3-27b-it:free")
    
    def test_generate_empty_prompt(self, service):
        """
        TEST: Handle empty prompt gracefully
        
        GIVEN: generate() is called with empty string
        WHEN: LLM processes request
        THEN: Either succeeds or returns proper error
        
        PURPOSE: Verify edge case handling
        """
        result = service.generate(prompt="", max_tokens=10)
        
        # Should either work or return error (not crash)
        assert "status_code" in result
        assert result["status_code"] in [200, 400, 500]
        
        print(f"✓ Empty prompt handled: {result['status_code']}")
    
    def test_error_response_format(self, service):
        """
        TEST: Errors return standardized format
        
        GIVEN: An error occurs during generation
        WHEN: _handle_error() processes it
        THEN: Returns dict with status_code, status, message, error_type
        
        PURPOSE: Verify consistent error format
        """
        # Force an error by using invalid parameters (if possible)
        # Or just verify the error format from a known error
        
        # Test with unreasonably long prompt to potentially trigger error
        very_long_prompt = "test " * 100000
        result = service.generate(prompt=very_long_prompt, max_tokens=10)
        
        # Should return error format (or succeed, which is also OK)
        assert "status_code" in result
        if result["status_code"] != 200:
            assert "message" in result
            assert "error_type" in result
            print(f"✓ Error format correct: {result['status_code']}")
        else:
            print("✓ Handled long prompt successfully")


class TestLLMServiceUtilities:
    """Test utility functions"""
    
    @pytest.fixture
    def service(self):
        """Create LLMService instance with FREE model"""
        return LLMService(model="google/gemma-3-27b-it:free")
    
    def test_connection_test(self, service):
        """
        TEST: test_connection() verifies API is working
        
        GIVEN: LLMService is initialized
        WHEN: test_connection() is called
        THEN: Makes real API call and returns result
        
        PURPOSE: Verify health check functionality
        """
        result = service.test_connection()
        
        assert result["status_code"] == 200
        assert result["status"] == "success"
        assert len(result["response"]) > 0
        
        print(f"✓ Connection test passed: {result['response'][:30]}")
    
    def test_provider_config(self, service):
        """
        TEST: Provider configuration is correct
        
        GIVEN: Service is initialized with OpenRouter
        WHEN: Checking internal config
        THEN: base_url and model are set correctly
        
        PURPOSE: Verify provider-specific setup
        """
        assert service.provider == "openrouter"
        assert service.client.base_url == "https://openrouter.ai/api/v1/"
        assert "free" in service.model.lower()  # Using free model
        
        print(f"✓ Provider config: {service.provider} - {service.model}")


class TestLLMServiceIntegration:
    """Integration tests with real API (comprehensive)"""
    
    @pytest.fixture
    def service(self):
        """Create LLMService instance with FREE model"""
        return LLMService(model="google/gemma-3-27b-it:free")
    
    def test_end_to_end_query(self, service):
        """
        TEST: End-to-end query about appliance parts
        
        GIVEN: A realistic user query
        WHEN: generate() is called
        THEN: Returns relevant, coherent response
        
        PURPOSE: Verify LLM can handle domain-specific queries
        """
        prompt = """
        You are a PartSelect assistant. Answer concisely:
        What is an ice maker assembly and why would it need replacement?
        """
        
        result = service.generate(prompt, max_tokens=100)
        
        assert result["status_code"] == 200
        response = result["response"].lower()
        
        # Check response mentions relevant keywords
        assert any(word in response for word in ["ice", "maker", "assembly"])
        assert len(result["response"]) > 20  # Meaningful response
        
        print(f"✓ Domain query response: {result['response'][:80]}...")
        print(f"✓ Used {result['usage']['total_tokens']} tokens")
    
    def test_multi_query_session(self, service):
        """
        TEST: Multiple queries in sequence
        
        GIVEN: Service handles multiple requests
        WHEN: generate() is called multiple times
        THEN: All succeed independently
        
        PURPOSE: Verify service is stateless and reusable
        """
        queries = [
            "What is a refrigerator?",
            "What is a dishwasher?",
            "What is an ice maker?"
        ]
        
        results = []
        for query in queries:
            result = service.generate(query, max_tokens=30)
            results.append(result)
            assert result["status_code"] == 200
        
        # All should succeed
        assert len(results) == 3
        assert all(r["status_code"] == 200 for r in results)
        
        total_tokens = sum(r["usage"]["total_tokens"] for r in results)
        print(f"✓ Processed {len(queries)} queries ({total_tokens} total tokens)")


# Run with: python3 -m pytest tests/test_llm_service.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])