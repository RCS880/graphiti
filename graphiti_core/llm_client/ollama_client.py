"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import os
import re
import typing
from typing import Any, ClassVar, Dict, Optional

import httpx
from pydantic import BaseModel, ValidationError

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """
    OllamaClient is a client class for interacting with Ollama's local language models.

    This client handles the specific quirks of Ollama models that output reasoning text,
    echo JSON schemas, and wrap responses in markdown code blocks. It includes robust
    sanitization to extract valid JSON from messy outputs.

    Attributes:
        base_url: The base URL for the Ollama API
        embed_base_url: Optional separate URL for embeddings
        model: The model name to use for completions
        embed_model: The model name to use for embeddings
        sanitization_stats: Counters for sanitization success/failure
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 3
    SANITIZATION_PATTERNS: ClassVar[Dict[str, re.Pattern]] = {
        'think_blocks': re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
        'markdown_json': re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL),
        'json_object': re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'),
        'schema_echo': re.compile(r'"type"\s*:\s*"object"[^}]*"properties"[^}]*\}'),
    }

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cache: bool = False,
        client: Optional[Any] = None,
    ):
        """
        Initialize the OllamaClient with configuration for local LLM interaction.

        Args:
            config: LLM configuration including model, API key, and base URL
            cache: Whether to cache responses (not implemented for Ollama)
            client: Optional HTTP client to use
        """
        if cache:
            raise NotImplementedError('Caching is not implemented for Ollama')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Set Ollama-specific configuration from environment
        self.base_url = os.getenv('OLLAMA_COMPLETIONS_BASE_URL',
                                  os.getenv('OLLAMA_BASE_URL', config.base_url or 'http://localhost:11434/v1'))
        self.embed_base_url = os.getenv('OLLAMA_EMBED_BASE_URL', self.base_url)
        self.model = os.getenv('OLLAMA_MODEL', config.model or 'deepseek-r1:7b')
        self.embed_model = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
        self.api_key = os.getenv('OLLAMA_API_KEY', config.api_key or 'ollama')

        # Initialize sanitization statistics
        self.sanitization_stats = {
            'recoverable': 0,
            'unrecoverable': 0,
            'total_attempts': 0,
        }

        self.client = client or httpx.AsyncClient(timeout=120.0)

    def _create_example_from_model(self, response_model: type[BaseModel]) -> str:
        """
        Create a simplified example JSON from a Pydantic model instead of the full schema.

        Args:
            response_model: The Pydantic model to create an example from

        Returns:
            A JSON string showing an example structure
        """
        try:
            # Try to create a model with default/empty values
            example = {}
            for field_name, field_info in response_model.model_fields.items():
                field_type = field_info.annotation

                # Generate example values based on type
                if field_type == str:
                    example[field_name] = f"example_{field_name}"
                elif field_type == int:
                    example[field_name] = 0
                elif field_type == float:
                    example[field_name] = 0.0
                elif field_type == bool:
                    example[field_name] = False
                elif field_type == list:
                    example[field_name] = []
                elif field_type == dict:
                    example[field_name] = {}
                else:
                    # For complex types, try to get a string representation
                    example[field_name] = f"<{field_name}>"

            return json.dumps(example, indent=2)
        except Exception as e:
            logger.warning(f"Could not create example from model: {e}")
            # Fall back to schema if example generation fails
            return json.dumps(response_model.model_json_schema(), indent=2)

    def sanitize_response(self, raw_output: str, response_model: Optional[type[BaseModel]] = None) -> Dict[str, Any]:
        """
        Sanitize Ollama's messy output to extract valid JSON.

        This method implements the sanitization pipeline specified in the requirements:
        1. Remove code fences and trim whitespace
        2. Remove <think> sections
        3. Locate and extract JSON objects
        4. Validate against response model if provided

        Args:
            raw_output: The raw response from Ollama
            response_model: Optional Pydantic model to validate against

        Returns:
            The extracted and validated JSON object

        Raises:
            ValueError: If no valid JSON can be extracted
        """
        self.sanitization_stats['total_attempts'] += 1
        original_length = len(raw_output)

        # Step 1: Basic cleanup - remove leading/trailing whitespace
        cleaned = raw_output.strip()

        # Step 2: Remove thinking blocks (e.g., <think>...</think>)
        cleaned = self.SANITIZATION_PATTERNS['think_blocks'].sub('', cleaned)

        # Step 3: Extract from markdown code blocks if present
        markdown_match = self.SANITIZATION_PATTERNS['markdown_json'].search(cleaned)
        if markdown_match:
            cleaned = markdown_match.group(1).strip()

        # Step 4: Remove any schema echo (Ollama sometimes repeats the schema)
        if '"type"' in cleaned and '"properties"' in cleaned:
            # Try to find the actual response after the schema
            parts = cleaned.split('\n\n')
            for part in parts[1:]:  # Skip first part which might be schema
                if '{' in part:
                    cleaned = part
                    break

        # Step 5: Try direct JSON parsing
        try:
            result = json.loads(cleaned)
            if response_model:
                # Validate against model
                response_model.model_validate(result)
            self.sanitization_stats['recoverable'] += 1
            logger.debug(f"Sanitized {original_length} chars to {len(json.dumps(result))} chars")
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Direct parsing failed: {e}")

        # Step 6: Try to extract JSON object using regex
        json_matches = self.SANITIZATION_PATTERNS['json_object'].findall(cleaned)
        for match in json_matches:
            try:
                result = json.loads(match)
                if response_model:
                    response_model.model_validate(result)
                self.sanitization_stats['recoverable'] += 1
                logger.debug(f"Extracted JSON from {original_length} chars to {len(match)} chars")
                return result
            except (json.JSONDecodeError, ValidationError):
                continue

        # Step 7: Last resort - try to find JSON by looking for opening brace
        start_idx = cleaned.find('{')
        if start_idx != -1:
            # Try to balance braces
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(cleaned[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx > start_idx:
                try:
                    potential_json = cleaned[start_idx:end_idx]
                    result = json.loads(potential_json)
                    if response_model:
                        response_model.model_validate(result)
                    self.sanitization_stats['recoverable'] += 1
                    logger.debug(f"Extracted JSON via brace matching from {original_length} chars")
                    return result
                except (json.JSONDecodeError, ValidationError):
                    pass

        # If we get here, sanitization failed
        self.sanitization_stats['unrecoverable'] += 1
        logger.error(f"Failed to extract valid JSON from {original_length} chars of output")
        logger.debug(f"First 500 chars of failed output: {raw_output[:500]}")
        raise ValueError(f"Could not extract valid JSON from Ollama response. Stats: {self.sanitization_stats}")

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: Optional[type[BaseModel]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> Dict[str, Any]:
        """
        Generate a response from Ollama and sanitize the output.

        Args:
            messages: List of conversation messages
            response_model: Optional Pydantic model for response validation
            max_tokens: Maximum tokens for response
            model_size: Model size preference (ignored, uses configured model)

        Returns:
            The sanitized and validated JSON response
        """
        # Convert messages to Ollama format
        ollama_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ['user', 'system']:
                ollama_messages.append({'role': m.role, 'content': m.content})

        # Build request payload
        payload = {
            'model': self.model,
            'messages': ollama_messages,
            'temperature': self.temperature,
            'max_tokens': max_tokens,
            'stream': False,  # Don't stream to avoid partial JSON
        }

        # Add format hint for JSON
        if response_model:
            # Use simplified example instead of full schema
            example_json = self._create_example_from_model(response_model)
            payload['messages'][-1]['content'] += (
                f"\n\nRespond with a JSON object following this example structure:\n\n{example_json}"
                "\n\nProvide ONLY the JSON object, no explanation or markdown formatting."
            )

        try:
            # Make API request
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
            )
            response.raise_for_status()

            # Extract content from response
            result = response.json()
            raw_content = result['choices'][0]['message']['content']

            # Sanitize and return
            return self.sanitize_response(raw_content, response_model)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Ollama rate limit exceeded") from e
            logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating Ollama response: {e}")
            raise

    async def create_embedding(self, text: str) -> list[float]:
        """
        Create embeddings using Ollama's embedding models.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        try:
            response = await self.client.post(
                f"{self.embed_base_url}/embeddings",
                json={
                    'model': self.embed_model,
                    'input': text,
                },
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
            )
            response.raise_for_status()

            result = response.json()
            embedding = result['data'][0]['embedding']

            # Retry once if we get an empty vector
            if not embedding or all(v == 0 for v in embedding):
                logger.warning("Received empty embedding vector, retrying...")
                response = await self.client.post(
                    f"{self.embed_base_url}/embeddings",
                    json={
                        'model': self.embed_model,
                        'input': text,
                    },
                    headers={
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json',
                    },
                )
                response.raise_for_status()
                result = response.json()
                embedding = result['data'][0]['embedding']

            return embedding

        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    async def rerank(self, query: str, documents: list[str]) -> Optional[list[float]]:
        """
        Rerank documents using Ollama (if reranking endpoint is available).

        Args:
            query: The query to rerank against
            documents: List of documents to rerank

        Returns:
            Reranking scores or None if not available
        """
        try:
            # Check if rerank endpoint exists
            response = await self.client.get(f"{self.base_url}/rerank")
            if response.status_code == 404:
                logger.debug("Rerank endpoint not available in Ollama")
                return None

            # If available, make rerank request
            response = await self.client.post(
                f"{self.base_url}/rerank",
                json={
                    'model': self.model,
                    'query': query,
                    'documents': documents,
                },
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
            )
            response.raise_for_status()

            result = response.json()
            return result.get('scores', None)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("Rerank endpoint not available")
                return None
            logger.error(f"Error reranking: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reranking: {e}")
            raise

    def get_sanitization_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about sanitization success/failure rates.

        Returns:
            Dictionary with sanitization statistics
        """
        total = self.sanitization_stats['total_attempts']
        if total == 0:
            return {
                'total_attempts': 0,
                'success_rate': 0,
                'recoverable': 0,
                'unrecoverable': 0,
            }

        return {
            'total_attempts': total,
            'success_rate': self.sanitization_stats['recoverable'] / total,
            'recoverable': self.sanitization_stats['recoverable'],
            'unrecoverable': self.sanitization_stats['unrecoverable'],
        }