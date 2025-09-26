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

import logging
import os
from typing import Optional

from .client import LLMClient
from .config import LLMConfig
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .openai_generic_client import OpenAIGenericClient

logger = logging.getLogger(__name__)


def create_llm_client(
    config: Optional[LLMConfig] = None,
    cache: bool = False,
) -> LLMClient:
    """
    Factory function to create the appropriate LLM client based on environment variables.

    This function detects which LLM provider to use based on environment configuration:
    1. If OLLAMA_COMPLETIONS_BASE_URL or OLLAMA_BASE_URL is set -> OllamaClient
    2. If custom config with base_url is provided -> OpenAIGenericClient
    3. Otherwise -> OpenAIClient (default)

    Args:
        config: Optional LLMConfig to use for initialization
        cache: Whether to enable caching (not supported by all clients)

    Returns:
        An appropriate LLMClient instance
    """
    # Check for Ollama environment variables
    ollama_base = os.getenv('OLLAMA_COMPLETIONS_BASE_URL') or os.getenv('OLLAMA_BASE_URL')

    if ollama_base:
        logger.info("Detected Ollama configuration, using OllamaClient")
        return OllamaClient(config=config, cache=cache)

    # Check if config has a custom base_url (for OpenAI-compatible endpoints)
    if config and config.base_url:
        logger.info(f"Using OpenAIGenericClient with base_url: {config.base_url}")
        return OpenAIGenericClient(config=config, cache=cache)

    # Default to standard OpenAI client
    logger.info("Using default OpenAIClient")
    return OpenAIClient(config=config, cache=cache)


def create_llm_config_from_env() -> Optional[LLMConfig]:
    """
    Create an LLMConfig from environment variables.

    Checks for various LLM provider configurations in environment:
    - OLLAMA_* variables for Ollama
    - KIMI_K2_* variables for Moonshot Kimi
    - Standard OpenAI variables

    Returns:
        LLMConfig if environment variables are found, None otherwise
    """
    # Check for Kimi/Moonshot configuration (cloud fallback)
    kimi_api_key = os.getenv('KIMI_K2_API_KEY')
    if kimi_api_key:
        return LLMConfig(
            api_key=kimi_api_key,
            model=os.getenv('KIMI_K2_MODEL', 'kimi-k2-0905-preview'),
            small_model=os.getenv('KIMI_K2_SMALL_MODEL', 'kimi-k2-0905-preview'),
            base_url=os.getenv('KIMI_K2_BASE_URL', 'https://api.moonshot.ai/v1'),
        )

    # Check for Ollama configuration
    ollama_base = os.getenv('OLLAMA_BASE_URL')
    if ollama_base:
        return LLMConfig(
            api_key=os.getenv('OLLAMA_API_KEY', 'ollama'),
            model=os.getenv('OLLAMA_MODEL', 'deepseek-r1:7b'),
            small_model=os.getenv('OLLAMA_MODEL', 'deepseek-r1:7b'),
            base_url=ollama_base,
        )

    # Check for standard OpenAI configuration
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        return LLMConfig(
            api_key=openai_key,
            model=os.getenv('OPENAI_MODEL', 'gpt-4'),
            small_model=os.getenv('OPENAI_SMALL_MODEL', 'gpt-3.5-turbo'),
            base_url=os.getenv('OPENAI_BASE_URL'),  # None if not set
        )

    return None