"""Simple Multi-Provider LLM Pipeline"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os


class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    AZURE = "azure"
    VERTEX_AI = "vertex_ai"
    ANTHROPIC_VERTEX = "anthropic_vertex"


@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: str
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 8192
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, config: LLMConfig) -> str:
        pass


class OpenRouterClient(BaseLLMClient):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    
    def generate(self, prompt: str, config: LLMConfig) -> str:
        completion = self.client.chat.completions.create(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, config: LLMConfig) -> str:
        completion = self.client.chat.completions.create(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
    
    def generate(self, prompt: str, config: LLMConfig) -> str:
        model = self.genai.GenerativeModel(config.model_name)
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
            }
        )
        return response.text


class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, config: LLMConfig) -> str:
        completion = self.client.messages.create(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        return completion.content[0].text


class AzureOpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, endpoint: str, api_version: str = "2024-02-15-preview", deployment: str = None):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment = deployment
    
    def generate(self, prompt: str, config: LLMConfig) -> str:
        model_to_use = self.deployment if self.deployment else config.model_name
        completion = self.client.chat.completions.create(
            model=model_to_use,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content


class VertexAIClient(BaseLLMClient):
    def __init__(self, project_id: str, location: str = "us-central1", credentials_dict: Optional[Dict] = None):
        import vertexai
        from vertexai.generative_models import GenerativeModel
        from google.oauth2 import service_account
        
        if credentials_dict:
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            vertexai.init(project=project_id, location=location, credentials=credentials)
        else:
            vertexai.init(project=project_id, location=location)
        
        self.GenerativeModel = GenerativeModel
    
    def generate(self, prompt: str, config: LLMConfig) -> str:
        model = self.GenerativeModel(config.model_name)
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
            }
        )
        return response.text


class AnthropicVertexClient(BaseLLMClient):
    def __init__(self, project_id: str, region: str = "us-east5", credentials_dict: Optional[Dict] = None):
        from anthropic import AnthropicVertex
        from google.oauth2 import service_account
        
        if credentials_dict:
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.client = AnthropicVertex(
                project_id=project_id,
                region=region,
                credentials=credentials
            )
        else:
            self.client = AnthropicVertex(
                project_id=project_id,
                region=region
            )
    
    def generate(self, prompt: str, config: LLMConfig) -> str:
        message = self.client.messages.create(
            model=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text


def create_client(provider: LLMProvider, api_key: str, extra_params: Dict) -> BaseLLMClient:
    if provider == LLMProvider.OPENROUTER:
        return OpenRouterClient(api_key)
    elif provider == LLMProvider.OPENAI:
        return OpenAIClient(api_key)
    elif provider == LLMProvider.GEMINI:
        return GeminiClient(api_key)
    elif provider == LLMProvider.CLAUDE:
        return ClaudeClient(api_key)
    elif provider == LLMProvider.AZURE:
        endpoint = extra_params.get('endpoint')
        if not endpoint:
            raise ValueError("Azure requires 'endpoint' in extra_params")
        api_version = extra_params.get('api_version', '2024-02-15-preview')
        deployment = extra_params.get('deployment')
        return AzureOpenAIClient(api_key, endpoint, api_version, deployment)
    elif provider == LLMProvider.VERTEX_AI:
        project_id = extra_params.get('project_id')
        if not project_id:
            raise ValueError("Vertex AI requires 'project_id' in extra_params")
        location = extra_params.get('location', 'us-central1')
        credentials_dict = extra_params.get('credentials_dict')
        return VertexAIClient(project_id, location, credentials_dict)
    elif provider == LLMProvider.ANTHROPIC_VERTEX:
        project_id = extra_params.get('project_id')
        if not project_id:
            raise ValueError("Anthropic Vertex AI requires 'project_id' in extra_params")
        region = extra_params.get('region', 'us-east5')
        credentials_dict = extra_params.get('credentials_dict')
        return AnthropicVertexClient(project_id, region, credentials_dict)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class LLMPipeline:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = create_client(config.provider, config.api_key, config.extra_params)
    
    def answer_question(self, system_prompt: str, temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        temp_config = LLMConfig(
            provider=self.config.provider,
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            extra_params=self.config.extra_params
        )
        return self.client.generate(system_prompt, temp_config)


# HELPER FUNCTIONS

def build_vertex_ai_credentials() -> Dict[str, str]:
    """
    Build Vertex AI credentials dictionary from environment variables.
    Raises detailed error if required variables are missing.
    """
    required_vars = {
        'GOOGLE_CLOUD_PROJECT_ID': 'GCP Project ID',
        'GOOGLE_CLOUD_PRIVATE_KEY_ID': 'Service Account Private Key ID',
        'GOOGLE_CLOUD_PRIVATE_KEY': 'Service Account Private Key',
        'GOOGLE_CLOUD_CLIENT_EMAIL': 'Service Account Email',
        'GOOGLE_CLOUD_CLIENT_X509_CERT_URL': 'Client Certificate URL'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  - {var} ({description})")
    
    if missing_vars:
        error_msg = (
            "\n VERTEX AI CONFIGURATION ERROR \n"
            "Missing required environment variables for Vertex AI:\n"
            + "\n".join(missing_vars) +
            "\n\nPlease add these to your .env file:\n"
            "GOOGLE_CLOUD_PROJECT_ID=your-project-id\n"
            "GOOGLE_CLOUD_PRIVATE_KEY_ID=your-key-id\n"
            'GOOGLE_CLOUD_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"\n'
            "GOOGLE_CLOUD_CLIENT_EMAIL=your-service-account@project.iam.gserviceaccount.com\n"
            "GOOGLE_CLOUD_CLIENT_X509_CERT_URL=https://...\n"
        )
        raise ValueError(error_msg)
    
    return {
        "type": "service_account",
        "project_id": os.getenv('GOOGLE_CLOUD_PROJECT_ID'),
        "private_key_id": os.getenv('GOOGLE_CLOUD_PRIVATE_KEY_ID'),
        "private_key": os.getenv('GOOGLE_CLOUD_PRIVATE_KEY', '').replace('\\n', '\n'),
        "client_email": os.getenv('GOOGLE_CLOUD_CLIENT_EMAIL'),
        "auth_uri": os.getenv('GOOGLE_CLOUD_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
        "token_uri": os.getenv('GOOGLE_CLOUD_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
        "auth_provider_x509_cert_url": os.getenv('GOOGLE_CLOUD_AUTH_PROVIDER_X509_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
        "client_x509_cert_url": os.getenv('GOOGLE_CLOUD_CLIENT_X509_CERT_URL'),
        "universe_domain": os.getenv('GOOGLE_CLOUD_UNIVERSE_DOMAIN', 'googleapis.com')
    }


def load_config_from_env(provider: str) -> LLMConfig:
    """
    Load LLM configuration from environment variables with proper error handling.
    
    Args:
        provider: Provider name (openrouter, openai, gemini, claude, azure, vertex_ai, anthropic_vertex)
    
    Returns:
        LLMConfig object ready to use
    
    Raises:
        ValueError: If required environment variables are missing
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    provider_enum = LLMProvider(provider.lower())
    extra_params = {}
    
    if provider_enum == LLMProvider.OPENROUTER:
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        if not api_key:
            raise ValueError(" Missing OPENROUTER_API_KEY in .env file")
        model_name = os.getenv('LLM_MODEL_NAME', 'anthropic/claude-3.5-sonnet')
    
    elif provider_enum == LLMProvider.OPENAI:
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env file")
        model_name = os.getenv('LLM_MODEL_NAME', 'gpt-4')
    
    elif provider_enum == LLMProvider.GEMINI:
        api_key = os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file")
        model_name = os.getenv('LLM_MODEL_NAME', 'gemini-1.5-pro')
    
    elif provider_enum == LLMProvider.CLAUDE:
        api_key = os.getenv('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in .env file")
        model_name = os.getenv('LLM_MODEL_NAME', 'claude-3-5-sonnet-20241022')
    
    elif provider_enum == LLMProvider.AZURE:
        api_key = os.getenv('AZURE_OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY in .env file")
        
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '')
        deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', '')
        
        if not endpoint or not deployment:
            raise ValueError(
                " Azure OpenAI requires:\n"
                "  - AZURE_OPENAI_ENDPOINT\n"
                "  - AZURE_OPENAI_DEPLOYMENT\n"
                "Please add these to your .env file"
            )
        
        model_name = os.getenv('LLM_MODEL_NAME', 'gpt-4')
        extra_params = {
            'endpoint': endpoint,
            'api_version': os.getenv('OPENAI_API_VERSION', '2024-02-15-preview'),
            'deployment': deployment
        }
    
    elif provider_enum == LLMProvider.VERTEX_AI:
        api_key = ''  # Vertex AI uses service account
        model_name = os.getenv('LLM_MODEL_NAME', 'gemini-1.5-pro')
        
        # Build credentials with error checking
        credentials_dict = build_vertex_ai_credentials()
        
        extra_params = {
            'project_id': os.getenv('GOOGLE_CLOUD_PROJECT_ID'),
            'location': os.getenv('VERTEX_LOCATION', 'us-central1'),
            'credentials_dict': credentials_dict
        }
    
    elif provider_enum == LLMProvider.ANTHROPIC_VERTEX:
        api_key = ''  # Anthropic Vertex uses service account
        model_name = os.getenv('LLM_MODEL_NAME', 'claude-sonnet-4-20250514')
        
        # Build credentials with error checking
        credentials_dict = build_vertex_ai_credentials()
        
        extra_params = {
            'project_id': os.getenv('GOOGLE_CLOUD_PROJECT_ID'),
            'region': os.getenv('ANTHROPIC_VERTEX_REGION', 'us-east5'),
            'credentials_dict': credentials_dict
        }
    
    return LLMConfig(
        provider=provider_enum,
        api_key=api_key,
        model_name=model_name,
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.3')),
        max_tokens=int(os.getenv('LLM_MAX_TOKENS', '8192')),
        extra_params=extra_params
    )