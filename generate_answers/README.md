# Multi-Provider LLM Client Library

A comprehensive code for interacting with multiple Large Language Model providers through a unified interface. Supports OpenRouter, OpenAI, Google Gemini, Anthropic Claude, Azure OpenAI, Google Vertex AI, and Anthropic Vertex AI.

## Features

- **7 LLM Providers**: Support for all major LLM APIs in one library
- **Unified Interface**: Same code works across all providers
- **Environment Variable Config**: Secure API key management
- **Error Handling**: Comprehensive error messages for missing configurations
- **Type Safety**: Full type hints and dataclasses
- **Extensible**: Easy to add new providers

## Installation

```bash
pip install openai google-generativeai anthropic azure-openai python-dotenv
```

For Vertex AI support:
```bash
pip install google-cloud-aiplatform
```

## Quick Start

### Method 1: Direct Configuration

```python
from llm_clients import LLMPipeline, LLMConfig, LLMProvider

# Configure for OpenAI
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    api_key="sk-your-openai-key",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2048
)

pipeline = LLMPipeline(config)
response = pipeline.answer_question("What is machine learning?")
print(response)
```

### Method 2: Environment Variables (Recommended)

Create a `.env` file in your project root:

```env
# For OpenAI
OPENAI_API_KEY=sk-your-openai-key
LLM_MODEL_NAME=gpt-4

# For Gemini
GEMINI_API_KEY=your-gemini-key
LLM_MODEL_NAME=gemini-1.5-pro

# For Claude
ANTHROPIC_API_KEY=sk-ant-your-claude-key
LLM_MODEL_NAME=claude-3-5-sonnet-20241022

# Global settings
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
```

Then use the helper function:

```python
from llm_clients import load_config_from_env, LLMPipeline

# Load configuration from environment
config = load_config_from_env("openai")  # or "gemini", "claude", etc.
pipeline = LLMPipeline(config)

response = pipeline.answer_question("Explain quantum computing")
```

## Supported Providers

### 1. OpenRouter

**Environment Variables:**
```env
OPENROUTER_API_KEY=sk-or-v1-xxxxx
LLM_MODEL_NAME=anthropic/claude-3.5-sonnet
```

**Code Example:**
```python
from llm_clients import load_config_from_env, LLMPipeline

config = load_config_from_env("openrouter")
pipeline = LLMPipeline(config)
response = pipeline.answer_question("What is machine learning?")
```

### 2. OpenAI

**Environment Variables:**
```env
OPENAI_API_KEY=sk-xxxxx
LLM_MODEL_NAME=gpt-4
```

**Code Example:**
```python
config = load_config_from_env("openai")
pipeline = LLMPipeline(config)
response = pipeline.answer_question("Explain neural networks")
```

### 3. Google Gemini

**Environment Variables:**
```env
GEMINI_API_KEY=your-gemini-api-key
LLM_MODEL_NAME=gemini-1.5-pro
```

**Code Example:**
```python
config = load_config_from_env("gemini")
pipeline = LLMPipeline(config)
response = pipeline.answer_question("What is deep learning?")
```

### 4. Anthropic Claude

**Environment Variables:**
```env
ANTHROPIC_API_KEY=sk-ant-xxxxx
LLM_MODEL_NAME=claude-3-5-sonnet-20241022
```

**Code Example:**
```python
config = load_config_from_env("claude")
pipeline = LLMPipeline(config)
response = pipeline.answer_question("Explain transformers architecture")
```

### 5. Azure OpenAI

**Environment Variables:**
```env
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
OPENAI_API_VERSION=2024-02-15-preview
LLM_MODEL_NAME=gpt-4
```

**Code Example:**
```python
config = load_config_from_env("azure")
pipeline = LLMPipeline(config)
response = pipeline.answer_question("What is reinforcement learning?")
```

### 6. Google Vertex AI

**Environment Variables:**
```env
GOOGLE_CLOUD_PROJECT_ID=your-gcp-project-id
GOOGLE_CLOUD_PRIVATE_KEY_ID=your-private-key-id
GOOGLE_CLOUD_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...
GOOGLE_CLOUD_CLIENT_EMAIL=your-service-account@project.iam.gserviceaccount.com
GOOGLE_CLOUD_CLIENT_X509_CERT_URL=https://...
VERTEX_LOCATION=us-central1
LLM_MODEL_NAME=gemini-1.5-pro
```

**Code Example:**
```python
config = load_config_from_env("vertex_ai")
pipeline = LLMPipeline(config)
response = pipeline.answer_question("Explain computer vision")
```

### 7. Anthropic Vertex AI

**Environment Variables:**
```env
# Same as Vertex AI above, plus:
ANTHROPIC_VERTEX_REGION=us-east5
LLM_MODEL_NAME=claude-sonnet-4@20250514
```

**Code Example:**
```python
config = load_config_from_env("anthropic_vertex")
pipeline = LLMPipeline(config)
response = pipeline.answer_question("What is natural language processing?")
```

## Advanced Usage

### Custom Temperature and Token Limits

```python
# Override defaults per request
response = pipeline.answer_question(
    "Complex question here",
    temperature=0.1,  # More focused
    max_tokens=4096   # Longer response
)
```

### Direct Client Usage

```python
from llm_clients import create_client

# Get direct client instance
client = create_client(LLMProvider.OPENAI, "your-key", {})

# Use directly
response = client.generate("Question here", config)
```

### Error Handling

The library provides detailed error messages for missing configurations:

```python
try:
    config = load_config_from_env("openai")
    pipeline = LLMPipeline(config)
except ValueError as e:
    print(f"Configuration error: {e}")
```

## API Reference

### Classes

- **`LLMProvider`**: Enum of supported providers
- **`LLMConfig`**: Configuration dataclass
- **`BaseLLMClient`**: Abstract base class for clients
- **`LLMPipeline`**: Main pipeline class

### Functions

- **`create_client(provider, api_key, extra_params)`**: Factory function for clients
- **`load_config_from_env(provider)`**: Load config from environment variables
- **`build_vertex_ai_credentials()`**: Build GCP credentials for Vertex AI

### Environment Variables

**Global:**
- `LLM_MODEL_NAME`: Default model name
- `LLM_TEMPERATURE`: Default temperature (0.0-1.0)
- `LLM_MAX_TOKENS`: Default max tokens

**Provider-specific variables are documented above.**

## Contributing

To add a new provider:

1. Add provider to `LLMProvider` enum
2. Create a client class inheriting from `BaseLLMClient`
3. Implement the `generate()` method
4. Add provider case to `create_client()` function
5. Add environment variable loading in `load_config_from_env()`

## License

This library is part of the DSBC Evaluation project.