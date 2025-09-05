"""LLM Integration Module for StudSar V4

Provides direct integration with Large Language Models for enhanced semantic understanding,
sentiment analysis, and intelligent text processing.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from enum import Enum

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

@dataclass
class SemanticAnalysis:
    """Result of semantic analysis"""
    sentiment: float  # -1.0 to 1.0
    emotion: str
    topics: List[str]
    keywords: List[str]
    complexity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    summary: str
    embeddings: Optional[List[float]] = None

@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    tokens_used: int
    model: str
    provider: str
    metadata: Dict[str, Any]

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('model', 'default')
        
    @abstractmethod
    async def analyze_text(self, text: str) -> SemanticAnalysis:
        """Analyze text for semantic understanding"""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        pass
    
    @abstractmethod
    async def summarize(self, text: str, max_length: int = 100) -> str:
        """Generate summary of text"""
        pass
    
    @abstractmethod
    async def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords from text"""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        
        self.client = openai.AsyncOpenAI(
            api_key=config.get('api_key'),
            base_url=config.get('base_url')
        )
        self.model = config.get('model', 'gpt-3.5-turbo')
    
    async def analyze_text(self, text: str) -> SemanticAnalysis:
        """Analyze text using OpenAI GPT"""
        try:
            prompt = f"""
Analyze the following text and provide a JSON response with:
- sentiment: float between -1.0 (negative) and 1.0 (positive)
- emotion: primary emotion (joy, sadness, anger, fear, surprise, disgust, neutral)
- topics: list of main topics (max 5)
- keywords: list of important keywords (max 10)
- complexity: text complexity from 0.0 (simple) to 1.0 (complex)
- confidence: analysis confidence from 0.0 to 1.0
- summary: brief summary (max 50 words)

Text: {text[:2000]}  # Limit text length

Respond only with valid JSON.
"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Generate embeddings separately
            embeddings = await self.generate_embeddings(text)
            
            return SemanticAnalysis(
                sentiment=result.get('sentiment', 0.0),
                emotion=result.get('emotion', 'neutral'),
                topics=result.get('topics', []),
                keywords=result.get('keywords', []),
                complexity=result.get('complexity', 0.5),
                confidence=result.get('confidence', 0.8),
                summary=result.get('summary', ''),
                embeddings=embeddings
            )
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._fallback_analysis(text)
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI"""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:8000]  # Limit text length
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embeddings failed: {e}")
            return []
    
    async def summarize(self, text: str, max_length: int = 100) -> str:
        """Summarize text using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"Summarize this text in {max_length} words or less:\n\n{text[:3000]}"
                }],
                temperature=0.3,
                max_tokens=max_length * 2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI summarization failed: {e}")
            return text[:max_length] + "..."
    
    async def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"Extract {num_keywords} most important keywords from this text. Return only the keywords separated by commas:\n\n{text[:2000]}"
                }],
                temperature=0.1,
                max_tokens=100
            )
            keywords = response.choices[0].message.content.strip().split(',')
            return [kw.strip() for kw in keywords[:num_keywords]]
        except Exception as e:
            logger.error(f"OpenAI keyword extraction failed: {e}")
            return []
    
    def _fallback_analysis(self, text: str) -> SemanticAnalysis:
        """Fallback analysis when API fails"""
        words = text.split()
        return SemanticAnalysis(
            sentiment=0.0,
            emotion='neutral',
            topics=[],
            keywords=words[:5] if words else [],
            complexity=min(len(words) / 100, 1.0),
            confidence=0.3,
            summary=text[:100] + "..." if len(text) > 100 else text
        )

class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face Transformers integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        self.sentiment_pipeline = None
        self.summarization_pipeline = None
        self.embedding_model = None
        self.embedding_tokenizer = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Hugging Face models"""
        try:
            # Sentiment analysis
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
            )
            
            # Summarization
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.config.get('summarization_model', 'facebook/bart-large-cnn')
            )
            
            # Embeddings
            embedding_model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face models: {e}")
    
    async def analyze_text(self, text: str) -> SemanticAnalysis:
        """Analyze text using Hugging Face models"""
        try:
            # Sentiment analysis
            sentiment_result = self.sentiment_pipeline(text[:512])[0]
            sentiment_score = sentiment_result['score']
            if sentiment_result['label'] == 'NEGATIVE':
                sentiment_score = -sentiment_score
            
            # Extract basic features
            words = text.split()
            keywords = self._extract_keywords_simple(text)
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(text)
            
            return SemanticAnalysis(
                sentiment=sentiment_score,
                emotion=self._map_sentiment_to_emotion(sentiment_result['label']),
                topics=self._extract_topics_simple(text),
                keywords=keywords,
                complexity=min(len(words) / 100, 1.0),
                confidence=sentiment_result['score'],
                summary=await self.summarize(text, 50),
                embeddings=embeddings
            )
            
        except Exception as e:
            logger.error(f"Hugging Face analysis failed: {e}")
            return self._fallback_analysis(text)
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Hugging Face"""
        try:
            if self.embedding_model is None:
                return []
            
            inputs = self.embedding_tokenizer(text[:512], return_tensors='pt', truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            
            return embeddings
        except Exception as e:
            logger.error(f"Hugging Face embeddings failed: {e}")
            return []
    
    async def summarize(self, text: str, max_length: int = 100) -> str:
        """Summarize text using Hugging Face"""
        try:
            if self.summarization_pipeline is None:
                return text[:max_length] + "..."
            
            result = self.summarization_pipeline(
                text[:1024],
                max_length=max_length,
                min_length=max_length // 4,
                do_sample=False
            )
            return result[0]['summary_text']
        except Exception as e:
            logger.error(f"Hugging Face summarization failed: {e}")
            return text[:max_length] + "..."
    
    async def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords using simple frequency analysis"""
        return self._extract_keywords_simple(text, num_keywords)
    
    def _extract_keywords_simple(self, text: str, num_keywords: int = 5) -> List[str]:
        """Simple keyword extraction based on frequency"""
        words = text.lower().split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:num_keywords]]
    
    def _extract_topics_simple(self, text: str) -> List[str]:
        """Simple topic extraction"""
        keywords = self._extract_keywords_simple(text, 3)
        return keywords
    
    def _map_sentiment_to_emotion(self, sentiment_label: str) -> str:
        """Map sentiment to emotion"""
        mapping = {
            'POSITIVE': 'joy',
            'NEGATIVE': 'sadness',
            'NEUTRAL': 'neutral'
        }
        return mapping.get(sentiment_label, 'neutral')
    
    def _fallback_analysis(self, text: str) -> SemanticAnalysis:
        """Fallback analysis when models fail"""
        words = text.split()
        return SemanticAnalysis(
            sentiment=0.0,
            emotion='neutral',
            topics=[],
            keywords=words[:5] if words else [],
            complexity=min(len(words) / 100, 1.0),
            confidence=0.3,
            summary=text[:100] + "..." if len(text) > 100 else text
        )

class LLMIntegrationManager:
    """Main manager for LLM integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider = config.get('default_provider', 'huggingface')
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        provider_configs = self.config.get('providers', {})
        
        # Initialize OpenAI if available and configured
        if OPENAI_AVAILABLE and 'openai' in provider_configs:
            try:
                self.providers['openai'] = OpenAIProvider(provider_configs['openai'])
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize Hugging Face if available
        if TRANSFORMERS_AVAILABLE:
            try:
                hf_config = provider_configs.get('huggingface', {})
                self.providers['huggingface'] = HuggingFaceProvider(hf_config)
                logger.info("Hugging Face provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face provider: {e}")
        
        if not self.providers:
            logger.warning("No LLM providers available")
    
    def get_provider(self, provider_name: Optional[str] = None) -> Optional[BaseLLMProvider]:
        """Get LLM provider by name"""
        if provider_name is None:
            provider_name = self.default_provider
        
        return self.providers.get(provider_name)
    
    async def analyze_text(self, text: str, provider: Optional[str] = None) -> Optional[SemanticAnalysis]:
        """Analyze text using specified or default provider"""
        llm_provider = self.get_provider(provider)
        if llm_provider is None:
            logger.warning(f"Provider {provider or self.default_provider} not available")
            return None
        
        return await llm_provider.analyze_text(text)
    
    async def generate_embeddings(self, text: str, provider: Optional[str] = None) -> List[float]:
        """Generate embeddings using specified or default provider"""
        llm_provider = self.get_provider(provider)
        if llm_provider is None:
            return []
        
        return await llm_provider.generate_embeddings(text)
    
    async def summarize(self, text: str, max_length: int = 100, provider: Optional[str] = None) -> str:
        """Summarize text using specified or default provider"""
        llm_provider = self.get_provider(provider)
        if llm_provider is None:
            return text[:max_length] + "..."
        
        return await llm_provider.summarize(text, max_length)
    
    async def extract_keywords(self, text: str, num_keywords: int = 5, provider: Optional[str] = None) -> List[str]:
        """Extract keywords using specified or default provider"""
        llm_provider = self.get_provider(provider)
        if llm_provider is None:
            return []
        
        return await llm_provider.extract_keywords(text, num_keywords)
    
    def is_available(self, provider: Optional[str] = None) -> bool:
        """Check if provider is available"""
        return self.get_provider(provider) is not None
    
    def list_providers(self) -> List[str]:
        """List available providers"""
        return list(self.providers.keys())

# Default configuration
DEFAULT_LLM_CONFIG = {
    'default_provider': 'huggingface',
    'providers': {
        'huggingface': {
            'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'summarization_model': 'facebook/bart-large-cnn',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'openai': {
            'model': 'gpt-3.5-turbo',
            'api_key': None,  # Should be set via environment variable
            'base_url': None
        }
    }
}

def create_llm_manager(config: Optional[Dict[str, Any]] = None) -> LLMIntegrationManager:
    """Create LLM integration manager with default or custom config"""
    if config is None:
        config = DEFAULT_LLM_CONFIG
    
    return LLMIntegrationManager(config)