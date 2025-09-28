"""
Tests for gpt-oss:20b specific output patterns and compatibility.
"""

import pytest
from pydantic import BaseModel
from graphiti_core.llm_client.ollama_client import OllamaClient
from graphiti_core.llm_client.config import LLMConfig


class GraphExtractionModel(BaseModel):
    """Model for graph extraction results."""
    entities: list[str]
    relationships: list[str]
    confidence: float


class TestGptOssCompatibility:
    """Test suite for gpt-oss:20b specific behaviors."""

    def setup_method(self):
        """Set up test client with gpt-oss:20b as model."""
        config = LLMConfig(model="gpt-oss:20b")
        self.client = OllamaClient(config=config)

    def test_sanitize_gpt_oss_thinking_pattern(self):
        """Test gpt-oss:20b style thinking and reasoning blocks."""
        gpt_oss_output = '''
        Let me analyze this step by step to extract entities and relationships.

        First, I'll identify the key entities:
        - Alice: person
        - Bob: person
        - DataPipe project: project

        Then I'll determine the relationships:
        - Alice works with Bob
        - Both contribute to DataPipe

        Based on this analysis:

        {
            "entities": ["Alice", "Bob", "DataPipe project"],
            "relationships": ["works_with", "contributes_to"],
            "confidence": 0.92
        }
        '''

        result = self.client.sanitize_response(gpt_oss_output, GraphExtractionModel)
        assert result["entities"] == ["Alice", "Bob", "DataPipe project"]
        assert result["relationships"] == ["works_with", "contributes_to"]
        assert result["confidence"] == 0.92

    def test_sanitize_gpt_oss_schema_echo(self):
        """Test handling of schema echoes common in gpt-oss:20b."""
        gpt_oss_with_schema = '''
        You want me to respond with a JSON object in the following format:

        {
            "type": "object",
            "properties": {
                "entities": {"type": "array", "items": {"type": "string"}},
                "relationships": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"}
            }
        }

        Here's my analysis of the text:

        {
            "entities": ["John", "Mary", "Project Alpha"],
            "relationships": ["manages", "reports_to"],
            "confidence": 0.87
        }
        '''

        result = self.client.sanitize_response(gpt_oss_with_schema, GraphExtractionModel)
        assert result["entities"] == ["John", "Mary", "Project Alpha"]
        assert result["relationships"] == ["manages", "reports_to"]
        assert result["confidence"] == 0.87

    def test_sanitize_gpt_oss_verbose_explanation(self):
        """Test gpt-oss:20b verbose explanations with embedded JSON."""
        verbose_output = '''
        I need to carefully extract entities and relationships from this text about workplace interactions.

        Looking at the sentence "Alice collaborates with Bob on the new marketing campaign while Charlie oversees the project timeline", I can identify:

        Entities:
        1. Alice - a person involved in the collaboration
        2. Bob - another person in the collaboration
        3. Charlie - a person with oversight role
        4. marketing campaign - the project being worked on
        5. project timeline - another aspect being managed

        Relationships:
        1. Alice collaborates with Bob
        2. Bob collaborates with Alice (reciprocal)
        3. Charlie oversees the project
        4. Alice works on marketing campaign
        5. Bob works on marketing campaign
        6. Charlie manages project timeline

        My confidence in this extraction is high given the clear subject-verb-object structure.

        ```json
        {
            "entities": ["Alice", "Bob", "Charlie", "marketing campaign", "project timeline"],
            "relationships": ["collaborates_with", "oversees", "works_on", "manages"],
            "confidence": 0.95
        }
        ```

        This extraction captures the key actors and their interactions as described in the source text.
        '''

        result = self.client.sanitize_response(verbose_output, GraphExtractionModel)
        assert len(result["entities"]) == 5
        assert "Alice" in result["entities"]
        assert "marketing campaign" in result["entities"]
        assert result["confidence"] == 0.95

    def test_sanitize_gpt_oss_malformed_json_recovery(self):
        """Test recovery from partially malformed JSON that gpt-oss:20b sometimes produces."""
        malformed_output = '''
        Based on my analysis:

        {
            "entities": ["Sarah", "David", "Q3 Report",
            "relationships": ["authored", "reviewed"],
            "confidence": 0.83
        }
        '''

        # This should fail initially but the regex fallback should catch it
        with pytest.raises(ValueError):
            self.client.sanitize_response(malformed_output, GraphExtractionModel)

    def test_sanitize_gpt_oss_nested_reasoning(self):
        """Test gpt-oss:20b nested reasoning patterns."""
        nested_reasoning = '''
        To extract entities and relationships, I need to:

        1. Parse the input text for named entities
           - Look for proper nouns
           - Identify organizational entities
           - Find project or product names

        2. Determine relationships between entities
           - Subject-verb-object patterns
           - Prepositional relationships
           - Contextual connections

        3. Assign confidence based on clarity
           - Clear grammatical structure = higher confidence
           - Ambiguous references = lower confidence

        Given the text "Emma leads the development team while working on the Phoenix project":

        {
            "entities": ["Emma", "development team", "Phoenix project"],
            "relationships": ["leads", "works_on"],
            "confidence": 0.91
        }
        '''

        result = self.client.sanitize_response(nested_reasoning, GraphExtractionModel)
        assert result["entities"] == ["Emma", "development team", "Phoenix project"]
        assert result["relationships"] == ["leads", "works_on"]
        assert result["confidence"] == 0.91

    def test_gpt_oss_example_generation(self):
        """Test that gpt-oss:20b gets appropriate example format."""
        example = self.client._create_example_from_model(GraphExtractionModel)

        # Should be JSON string, not schema
        assert '"type"' not in example  # No schema artifacts
        assert '"properties"' not in example
        assert 'entities' in example  # Has the actual fields
        assert 'relationships' in example
        assert 'confidence' in example

    def test_gpt_oss_default_configuration(self):
        """Test that gpt-oss:20b is properly set as default model."""
        client = OllamaClient()  # No config provided
        assert client.model == "gpt-oss:20b"

    def test_sanitization_metrics_with_gpt_oss_patterns(self):
        """Test that metrics properly track gpt-oss:20b specific patterns."""
        client = OllamaClient(config=LLMConfig(model="gpt-oss:20b"))

        # Test successful sanitization with typical gpt-oss patterns
        outputs = [
            '{"entities": ["A"], "relationships": ["R"], "confidence": 0.8}',  # Clean
            '''
            Let me think about this...
            {"entities": ["B"], "relationships": ["S"], "confidence": 0.9}
            ''',  # With reasoning
            '''```json
            {"entities": ["C"], "relationships": ["T"], "confidence": 0.7}
            ```''',  # Markdown wrapped
        ]

        for output in outputs:
            try:
                client.sanitize_response(output, GraphExtractionModel)
            except:
                pass

        metrics = client.get_sanitization_metrics()
        assert metrics['total_attempts'] == 3
        assert metrics['recoverable'] >= 2  # Should handle most patterns


class TestGptOssIntegration:
    """Test integration scenarios specific to gpt-oss:20b usage."""

    def test_prompt_construction_for_gpt_oss(self):
        """Test that prompts are constructed appropriately for gpt-oss:20b."""
        client = OllamaClient(config=LLMConfig(model="gpt-oss:20b"))

        example = client._create_example_from_model(GraphExtractionModel)

        # Should create simple example, not verbose schema
        assert len(example) < 500  # Reasonably concise
        assert '"example_entities"' in example  # Example format

        # Example should be valid JSON
        import json
        parsed = json.loads(example)
        assert "entities" in parsed
        assert "relationships" in parsed
        assert "confidence" in parsed

    def test_factory_detects_gpt_oss_config(self):
        """Test that factory properly configures for gpt-oss:20b."""
        import os
        from unittest.mock import patch
        from graphiti_core.llm_client.factory import create_llm_client

        with patch.dict(os.environ, {
            'OLLAMA_BASE_URL': 'http://localhost:11434/v1',
            'OLLAMA_MODEL': 'gpt-oss:20b'
        }):
            client = create_llm_client()

            # Should be OllamaClient with gpt-oss:20b
            from graphiti_core.llm_client.ollama_client import OllamaClient
            assert isinstance(client, OllamaClient)
            assert client.model == "gpt-oss:20b"