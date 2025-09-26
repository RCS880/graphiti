"""
Tests for OllamaClient JSON sanitization and functionality.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from graphiti_core.llm_client.ollama_client import OllamaClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


class TestResponseModel(BaseModel):
    """Test model for response validation."""
    entity: str
    relationship: str
    confidence: float


class TestSanitization:
    """Test suite for JSON sanitization functionality."""

    def setup_method(self):
        """Set up test client."""
        self.client = OllamaClient(config=LLMConfig())

    def test_sanitize_clean_json(self):
        """Test that clean JSON passes through unchanged."""
        clean_json = '{"entity": "test", "relationship": "knows", "confidence": 0.9}'
        result = self.client.sanitize_response(clean_json, TestResponseModel)
        assert result == {"entity": "test", "relationship": "knows", "confidence": 0.9}

    def test_sanitize_markdown_wrapped_json(self):
        """Test extraction from markdown code blocks."""
        markdown_json = '''
        Here's the response:
        ```json
        {"entity": "Alice", "relationship": "works_with", "confidence": 0.85}
        ```
        '''
        result = self.client.sanitize_response(markdown_json, TestResponseModel)
        assert result == {"entity": "Alice", "relationship": "works_with", "confidence": 0.85}

    def test_sanitize_think_blocks(self):
        """Test removal of thinking blocks."""
        thinking_json = '''
        <think>
        I need to extract entities from this text.
        Let me analyze the relationships...
        </think>
        {"entity": "Bob", "relationship": "manages", "confidence": 0.7}
        '''
        result = self.client.sanitize_response(thinking_json, TestResponseModel)
        assert result == {"entity": "Bob", "relationship": "manages", "confidence": 0.7}

    def test_sanitize_schema_echo(self):
        """Test handling of schema echo in response."""
        schema_echo = '''
        The schema is:
        {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "relationship": {"type": "string"},
                "confidence": {"type": "number"}
            }
        }

        Here's my response:
        {"entity": "Charlie", "relationship": "reports_to", "confidence": 0.95}
        '''
        result = self.client.sanitize_response(schema_echo, TestResponseModel)
        assert result == {"entity": "Charlie", "relationship": "reports_to", "confidence": 0.95}

    def test_sanitize_multiple_json_objects(self):
        """Test extraction of first valid JSON object from multiple."""
        multiple_json = '''
        Invalid: {"incomplete":
        Valid: {"entity": "David", "relationship": "collaborates", "confidence": 0.6}
        Another: {"entity": "Eve", "relationship": "knows", "confidence": 0.8}
        '''
        result = self.client.sanitize_response(multiple_json, TestResponseModel)
        # Should extract the first valid one
        assert result["entity"] == "David"

    def test_sanitize_nested_json(self):
        """Test handling of nested JSON structures."""
        nested_json = '''
        {
            "entity": "Complex",
            "relationship": "nested_data",
            "confidence": 0.75
        }
        '''
        result = self.client.sanitize_response(nested_json)
        assert result == {"entity": "Complex", "relationship": "nested_data", "confidence": 0.75}

    def test_sanitize_with_extraneous_text(self):
        """Test extraction with lots of surrounding text."""
        messy_json = '''
        Let me think about this problem step by step.
        First, I need to identify the entities.
        Then I'll determine their relationships.

        <think>
        This is complex reasoning...
        </think>

        Based on my analysis:

        ```json
        {"entity": "Frank", "relationship": "supervises", "confidence": 0.88}
        ```

        I hope this helps with your query!
        '''
        result = self.client.sanitize_response(messy_json, TestResponseModel)
        assert result == {"entity": "Frank", "relationship": "supervises", "confidence": 0.88}

    def test_sanitize_invalid_json_raises(self):
        """Test that completely invalid JSON raises an error."""
        invalid_json = "This is not JSON at all, just plain text."
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            self.client.sanitize_response(invalid_json, TestResponseModel)

    def test_sanitize_tracks_statistics(self):
        """Test that sanitization statistics are tracked correctly."""
        client = OllamaClient(config=LLMConfig())

        # Successful sanitization
        client.sanitize_response('{"entity": "test", "relationship": "test", "confidence": 0.5}', TestResponseModel)
        assert client.sanitization_stats['recoverable'] == 1
        assert client.sanitization_stats['unrecoverable'] == 0

        # Failed sanitization
        try:
            client.sanitize_response('not json', TestResponseModel)
        except ValueError:
            pass
        assert client.sanitization_stats['recoverable'] == 1
        assert client.sanitization_stats['unrecoverable'] == 1
        assert client.sanitization_stats['total_attempts'] == 2


class TestOllamaClientIntegration:
    """Test suite for OllamaClient integration."""

    @pytest.mark.asyncio
    async def test_generate_response_with_mock(self):
        """Test generate_response with mocked HTTP client."""
        config = LLMConfig(
            api_key="test_key",
            model="test_model",
            base_url="http://localhost:11434/v1"
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"entity": "TestEntity", "relationship": "test_rel", "confidence": 0.99}'
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        client = OllamaClient(config=config, client=mock_client)

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Extract entities from this text.")
        ]

        result = await client._generate_response(messages, TestResponseModel)

        assert result == {"entity": "TestEntity", "relationship": "test_rel", "confidence": 0.99}
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_with_messy_output(self):
        """Test that messy Ollama output is properly sanitized."""
        config = LLMConfig()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        # Simulate messy Ollama output
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '''
                    <think>
                    I need to analyze this request carefully.
                    The user wants entity extraction.
                    </think>

                    Here's the extracted information:

                    ```json
                    {"entity": "MessyOutput", "relationship": "cleaned", "confidence": 0.77}
                    ```

                    I've completed the extraction successfully.
                    '''
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        client = OllamaClient(config=config, client=mock_client)

        messages = [Message(role="user", content="Test")]
        result = await client._generate_response(messages, TestResponseModel)

        assert result == {"entity": "MessyOutput", "relationship": "cleaned", "confidence": 0.77}

    @pytest.mark.asyncio
    async def test_create_embedding(self):
        """Test embedding creation."""
        config = LLMConfig()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        client = OllamaClient(config=config, client=mock_client)

        embedding = await client.create_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_embedding_retry_on_empty(self):
        """Test that empty embeddings trigger a retry."""
        config = LLMConfig()

        mock_client = AsyncMock()

        # First call returns empty embedding
        mock_response1 = MagicMock()
        mock_response1.json.return_value = {
            "data": [{
                "embedding": [0, 0, 0]
            }]
        }
        mock_response1.raise_for_status = MagicMock()

        # Second call returns valid embedding
        mock_response2 = MagicMock()
        mock_response2.json.return_value = {
            "data": [{
                "embedding": [0.1, 0.2, 0.3]
            }]
        }
        mock_response2.raise_for_status = MagicMock()

        mock_client.post.side_effect = [mock_response1, mock_response2]

        client = OllamaClient(config=config, client=mock_client)

        embedding = await client.create_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        assert mock_client.post.call_count == 2

    def test_create_example_from_model(self):
        """Test example JSON creation from Pydantic model."""
        client = OllamaClient(config=LLMConfig())

        example = client._create_example_from_model(TestResponseModel)
        example_dict = json.loads(example)

        assert "entity" in example_dict
        assert "relationship" in example_dict
        assert "confidence" in example_dict
        assert example_dict["entity"] == "example_entity"
        assert example_dict["confidence"] == 0.0

    def test_get_sanitization_metrics(self):
        """Test sanitization metrics reporting."""
        client = OllamaClient(config=LLMConfig())

        # Initial state
        metrics = client.get_sanitization_metrics()
        assert metrics['total_attempts'] == 0
        assert metrics['success_rate'] == 0

        # After successful sanitization
        client.sanitize_response('{"entity": "test", "relationship": "test", "confidence": 0.5}', TestResponseModel)
        metrics = client.get_sanitization_metrics()
        assert metrics['total_attempts'] == 1
        assert metrics['success_rate'] == 1.0
        assert metrics['recoverable'] == 1
        assert metrics['unrecoverable'] == 0


class TestEnvironmentConfiguration:
    """Test environment-based configuration."""

    @patch.dict('os.environ', {
        'OLLAMA_COMPLETIONS_BASE_URL': 'http://custom:11434/v1',
        'OLLAMA_MODEL': 'custom-model',
        'OLLAMA_EMBED_MODEL': 'custom-embed',
        'OLLAMA_API_KEY': 'custom-key'
    })
    def test_environment_configuration(self):
        """Test that environment variables are properly loaded."""
        client = OllamaClient()

        assert client.base_url == 'http://custom:11434/v1'
        assert client.model == 'custom-model'
        assert client.embed_model == 'custom-embed'
        assert client.api_key == 'custom-key'

    @patch.dict('os.environ', {
        'OLLAMA_BASE_URL': 'http://fallback:11434/v1',
        'OLLAMA_EMBED_BASE_URL': 'http://embed:11434/v1'
    })
    def test_fallback_environment_configuration(self):
        """Test fallback environment variable handling."""
        client = OllamaClient()

        assert client.base_url == 'http://fallback:11434/v1'
        assert client.embed_base_url == 'http://embed:11434/v1'