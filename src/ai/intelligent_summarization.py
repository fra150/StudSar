"""Intelligent Summarization Module for StudSar V4
Implements AI-powered summarization capabilities for memories, episodes,
and content using LLM integration and advanced NLP techniques.
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import json
from collections import defaultdict, Counter
import math

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    KMeans = None
    cosine_similarity = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except ImportError:
    nltk = None
    sent_tokenize = None
    word_tokenize = None
    stopwords = None
    PorterStemmer = None

logger = logging.getLogger(__name__)

class SummaryType(Enum):
    """Types of summaries"""
    EXTRACTIVE = "extractive"  # Extract key sentences
    ABSTRACTIVE = "abstractive"  # Generate new text
    HYBRID = "hybrid"  # Combination of both
    BULLET_POINTS = "bullet_points"  # Key points format
    TIMELINE = "timeline"  # Chronological summary
    THEMATIC = "thematic"  # Organized by themes
    STATISTICAL = "statistical"  # Data-driven summary

class SummaryLength(Enum):
    """Summary length options"""
    BRIEF = "brief"  # 1-2 sentences
    SHORT = "short"  # 3-5 sentences
    MEDIUM = "medium"  # 1-2 paragraphs
    LONG = "long"  # Multiple paragraphs
    DETAILED = "detailed"  # Comprehensive summary

class ContentType(Enum):
    """Types of content to summarize"""
    MEMORY = "memory"
    EPISODE = "episode"
    SEARCH_RESULTS = "search_results"
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    NETWORK_ANALYSIS = "network_analysis"
    TEMPORAL_SEQUENCE = "temporal_sequence"

@dataclass
class SummaryRequest:
    """Request for summarization"""
    content: Any
    content_type: ContentType
    summary_type: SummaryType = SummaryType.HYBRID
    summary_length: SummaryLength = SummaryLength.MEDIUM
    focus_areas: Optional[List[str]] = None
    exclude_topics: Optional[List[str]] = None
    target_audience: Optional[str] = None  # technical, general, academic
    include_statistics: bool = False
    include_timeline: bool = False
    include_key_entities: bool = True
    language: str = "en"
    custom_instructions: Optional[str] = None

@dataclass
class SummaryResult:
    """Result of summarization"""
    summary_text: str
    summary_type: SummaryType
    summary_length: SummaryLength
    key_points: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    statistics: Optional[Dict[str, Any]] = None
    timeline: Optional[List[Dict[str, Any]]] = None
    confidence_score: float = 0.0
    processing_time: float = 0.0
    word_count: int = 0
    compression_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractiveSentence:
    """Sentence with extraction score"""
    text: str
    score: float
    position: int
    length: int
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

class BaseSummarizer(ABC):
    """Base class for summarizers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Generate summary"""
        pass
    
    def _calculate_compression_ratio(self, original_text: str, summary_text: str) -> float:
        """Calculate compression ratio"""
        original_words = len(original_text.split())
        summary_words = len(summary_text.split())
        
        if original_words == 0:
            return 0.0
        
        return summary_words / original_words
    
    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        # Simple entity extraction using patterns
        entities = []
        
        # Capitalized words (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-z]+\b'
        capitalized_words = re.findall(capitalized_pattern, text)
        entities.extend(capitalized_words)
        
        # Numbers and dates
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, text)
        entities.extend(numbers)
        
        # Remove duplicates and common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'Or'}
        entities = list(set(entities) - common_words)
        
        return entities[:10]  # Limit to top 10
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract main themes from text"""
        # Simple theme extraction using keyword frequency
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        if stopwords and nltk:
            try:
                stop_words = set(stopwords.words('english'))
                words = [w for w in words if w not in stop_words and len(w) > 3]
            except:
                words = [w for w in words if len(w) > 3]
        else:
            # Basic stopword list
            basic_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [w for w in words if w not in basic_stopwords and len(w) > 3]
        
        # Count word frequency
        word_counts = Counter(words)
        
        # Get top themes
        themes = [word for word, count in word_counts.most_common(5)]
        return themes

class ExtractiveSummarizer(BaseSummarizer):
    """Extractive summarization using sentence ranking"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stemmer = PorterStemmer() if PorterStemmer else None
    
    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Generate extractive summary"""
        start_time = time.time()
        
        # Convert content to text
        text = self._extract_text_from_content(request.content, request.content_type)
        
        if not text or len(text.strip()) < 50:
            return SummaryResult(
                summary_text="Content too short to summarize.",
                summary_type=SummaryType.EXTRACTIVE,
                summary_length=request.summary_length,
                processing_time=time.time() - start_time
            )
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 3:
            summary_text = " ".join(sentences)
        else:
            # Score sentences
            scored_sentences = self._score_sentences(sentences, text)
            
            # Select top sentences based on length requirement
            target_count = self._get_target_sentence_count(request.summary_length, len(sentences))
            
            # Sort by score and select top sentences
            top_sentences = sorted(scored_sentences, key=lambda x: x.score, reverse=True)[:target_count]
            
            # Sort selected sentences by original position
            top_sentences.sort(key=lambda x: x.position)
            
            summary_text = " ".join([s.text for s in top_sentences])
        
        # Extract additional information
        key_points = self._extract_key_points(text)
        key_entities = self._extract_key_entities(text)
        themes = self._extract_themes(text)
        
        # Calculate statistics
        statistics = {
            'original_sentences': len(sentences),
            'summary_sentences': len(summary_text.split('. ')),
            'original_words': len(text.split()),
            'summary_words': len(summary_text.split())
        }
        
        processing_time = time.time() - start_time
        
        return SummaryResult(
            summary_text=summary_text,
            summary_type=SummaryType.EXTRACTIVE,
            summary_length=request.summary_length,
            key_points=key_points,
            key_entities=key_entities,
            themes=themes,
            statistics=statistics,
            confidence_score=self._calculate_confidence(scored_sentences if len(sentences) > 3 else []),
            processing_time=processing_time,
            word_count=len(summary_text.split()),
            compression_ratio=self._calculate_compression_ratio(text, summary_text)
        )
    
    def _extract_text_from_content(self, content: Any, content_type: ContentType) -> str:
        """Extract text from various content types"""
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            if content_type == ContentType.MEMORY:
                return content.get('content', '') + ' ' + content.get('context', '')
            elif content_type == ContentType.EPISODE:
                return (content.get('description', '') + ' ' + 
                       content.get('outcome', '') + ' ' +
                       ' '.join(content.get('lessons_learned', [])))
            elif content_type == ContentType.SEARCH_RESULTS:
                texts = []
                for result in content.get('results', []):
                    if isinstance(result, dict):
                        texts.append(result.get('content', ''))
                    else:
                        texts.append(str(result))
                return ' '.join(texts)
            else:
                # Generic dict handling
                return ' '.join(str(v) for v in content.values() if v)
        
        if isinstance(content, list):
            return ' '.join(str(item) for item in content)
        
        return str(content)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if sent_tokenize:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _score_sentences(self, sentences: List[str], full_text: str) -> List[ExtractiveSentence]:
        """Score sentences for extraction"""
        scored_sentences = []
        
        # Calculate word frequencies
        words = re.findall(r'\b\w+\b', full_text.lower())
        word_freq = Counter(words)
        
        # Remove stopwords from frequency calculation
        if stopwords and nltk:
            try:
                stop_words = set(stopwords.words('english'))
                word_freq = {w: f for w, f in word_freq.items() if w not in stop_words}
            except:
                pass
        
        for i, sentence in enumerate(sentences):
            score = self._calculate_sentence_score(sentence, word_freq, i, len(sentences))
            
            scored_sentence = ExtractiveSentence(
                text=sentence,
                score=score,
                position=i,
                length=len(sentence.split()),
                keywords=self._extract_sentence_keywords(sentence, word_freq),
                entities=self._extract_key_entities(sentence)
            )
            
            scored_sentences.append(scored_sentence)
        
        return scored_sentences
    
    def _calculate_sentence_score(self, sentence: str, word_freq: Dict[str, int], 
                                position: int, total_sentences: int) -> float:
        """Calculate score for a sentence"""
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        if not words:
            return 0.0
        
        # Word frequency score
        freq_score = sum(word_freq.get(word, 0) for word in words) / len(words)
        
        # Position score (first and last sentences often important)
        if position == 0 or position == total_sentences - 1:
            position_score = 1.5
        elif position < total_sentences * 0.3:  # First third
            position_score = 1.2
        else:
            position_score = 1.0
        
        # Length score (prefer medium-length sentences)
        length = len(words)
        if 10 <= length <= 25:
            length_score = 1.2
        elif 5 <= length <= 35:
            length_score = 1.0
        else:
            length_score = 0.8
        
        # Keyword density score
        unique_words = len(set(words))
        diversity_score = unique_words / len(words) if words else 0
        
        # Numerical content score (numbers often indicate important facts)
        numerical_score = 1.1 if re.search(r'\d+', sentence) else 1.0
        
        # Combine scores
        total_score = freq_score * position_score * length_score * (1 + diversity_score) * numerical_score
        
        return total_score
    
    def _extract_sentence_keywords(self, sentence: str, word_freq: Dict[str, int]) -> List[str]:
        """Extract keywords from sentence"""
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Get words with high frequency
        keywords = []
        for word in words:
            if word in word_freq and word_freq[word] > 1:
                keywords.append(word)
        
        return list(set(keywords))[:5]
    
    def _get_target_sentence_count(self, length: SummaryLength, total_sentences: int) -> int:
        """Get target number of sentences based on length requirement"""
        if length == SummaryLength.BRIEF:
            return min(2, max(1, total_sentences // 10))
        elif length == SummaryLength.SHORT:
            return min(5, max(2, total_sentences // 5))
        elif length == SummaryLength.MEDIUM:
            return min(8, max(3, total_sentences // 3))
        elif length == SummaryLength.LONG:
            return min(12, max(5, total_sentences // 2))
        else:  # DETAILED
            return min(20, max(8, total_sentences * 2 // 3))
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text"""
        sentences = self._split_sentences(text)
        
        # Look for sentences with indicators of importance
        key_indicators = [
            r'\b(important|significant|key|main|primary|crucial|essential)\b',
            r'\b(result|conclusion|finding|outcome)\b',
            r'\b(first|second|third|finally|lastly)\b',
            r'\b(because|therefore|thus|hence|consequently)\b'
        ]
        
        key_points = []
        for sentence in sentences:
            for pattern in key_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    key_points.append(sentence.strip())
                    break
        
        return key_points[:5]
    
    def _calculate_confidence(self, scored_sentences: List[ExtractiveSentence]) -> float:
        """Calculate confidence in the summary"""
        if not scored_sentences:
            return 0.5
        
        scores = [s.score for s in scored_sentences]
        
        if len(scores) < 2:
            return 0.7
        
        # Calculate score variance (lower variance = higher confidence)
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        # Normalize confidence (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + variance)
        
        return min(1.0, max(0.0, confidence))

class LLMSummarizer(BaseSummarizer):
    """LLM-based abstractive summarization"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_provider = None
        
        # Try to import LLM integration
        try:
            from .llm_integration import create_llm_provider
            self.llm_provider = create_llm_provider(config.get('llm_config', {}))
        except ImportError:
            logger.warning("LLM integration not available for abstractive summarization")
    
    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Generate abstractive summary using LLM"""
        start_time = time.time()
        
        if not self.llm_provider:
            # Fallback to extractive summarization
            extractive = ExtractiveSummarizer(self.config)
            result = await extractive.summarize(request)
            result.summary_type = SummaryType.ABSTRACTIVE
            result.metadata['fallback'] = 'extractive'
            return result
        
        # Extract text from content
        text = self._extract_text_from_content(request.content, request.content_type)
        
        if not text or len(text.strip()) < 50:
            return SummaryResult(
                summary_text="Content too short to summarize.",
                summary_type=SummaryType.ABSTRACTIVE,
                summary_length=request.summary_length,
                processing_time=time.time() - start_time
            )
        
        # Build prompt for LLM
        prompt = self._build_summarization_prompt(text, request)
        
        try:
            # Generate summary using LLM
            response = await self.llm_provider.generate_text(
                prompt=prompt,
                max_tokens=self._get_max_tokens(request.summary_length),
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            summary_text = response.text.strip()
            
            # Extract additional information
            key_points = self._extract_key_points_from_summary(summary_text)
            key_entities = self._extract_key_entities(text)
            themes = self._extract_themes(text)
            
            # Calculate statistics
            statistics = {
                'original_words': len(text.split()),
                'summary_words': len(summary_text.split()),
                'llm_tokens_used': response.tokens_used,
                'llm_model': response.model_name
            }
            
            processing_time = time.time() - start_time
            
            return SummaryResult(
                summary_text=summary_text,
                summary_type=SummaryType.ABSTRACTIVE,
                summary_length=request.summary_length,
                key_points=key_points,
                key_entities=key_entities,
                themes=themes,
                statistics=statistics,
                confidence_score=response.confidence,
                processing_time=processing_time,
                word_count=len(summary_text.split()),
                compression_ratio=self._calculate_compression_ratio(text, summary_text)
            )
            
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            
            # Fallback to extractive
            extractive = ExtractiveSummarizer(self.config)
            result = await extractive.summarize(request)
            result.summary_type = SummaryType.ABSTRACTIVE
            result.metadata['fallback'] = 'extractive'
            result.metadata['llm_error'] = str(e)
            return result
    
    def _extract_text_from_content(self, content: Any, content_type: ContentType) -> str:
        """Extract text from various content types"""
        # Reuse extractive summarizer's method
        extractive = ExtractiveSummarizer(self.config)
        return extractive._extract_text_from_content(content, content_type)
    
    def _build_summarization_prompt(self, text: str, request: SummaryRequest) -> str:
        """Build prompt for LLM summarization"""
        # Base prompt
        prompt_parts = []
        
        # Task description
        if request.summary_type == SummaryType.BULLET_POINTS:
            prompt_parts.append("Create a bullet-point summary of the following text:")
        elif request.summary_type == SummaryType.TIMELINE:
            prompt_parts.append("Create a chronological timeline summary of the following text:")
        elif request.summary_type == SummaryType.THEMATIC:
            prompt_parts.append("Create a thematic summary organized by main topics of the following text:")
        else:
            prompt_parts.append("Create a concise summary of the following text:")
        
        # Length instruction
        length_instructions = {
            SummaryLength.BRIEF: "Keep it to 1-2 sentences.",
            SummaryLength.SHORT: "Keep it to 3-5 sentences.",
            SummaryLength.MEDIUM: "Write 1-2 paragraphs.",
            SummaryLength.LONG: "Write 2-3 paragraphs.",
            SummaryLength.DETAILED: "Provide a comprehensive summary with multiple paragraphs."
        }
        
        prompt_parts.append(length_instructions.get(request.summary_length, "Keep it concise."))
        
        # Focus areas
        if request.focus_areas:
            prompt_parts.append(f"Focus particularly on: {', '.join(request.focus_areas)}.")
        
        # Exclude topics
        if request.exclude_topics:
            prompt_parts.append(f"Avoid mentioning: {', '.join(request.exclude_topics)}.")
        
        # Target audience
        if request.target_audience:
            audience_instructions = {
                'technical': "Use technical language appropriate for experts.",
                'general': "Use simple, accessible language for general audiences.",
                'academic': "Use formal academic language with precise terminology."
            }
            instruction = audience_instructions.get(request.target_audience, "")
            if instruction:
                prompt_parts.append(instruction)
        
        # Additional requirements
        if request.include_statistics:
            prompt_parts.append("Include relevant statistics and numbers.")
        
        if request.include_key_entities:
            prompt_parts.append("Mention key entities, names, and important terms.")
        
        # Custom instructions
        if request.custom_instructions:
            prompt_parts.append(request.custom_instructions)
        
        # Combine prompt parts
        prompt = " ".join(prompt_parts) + "\n\nText to summarize:\n" + text
        
        return prompt
    
    def _get_max_tokens(self, length: SummaryLength) -> int:
        """Get maximum tokens for summary length"""
        token_limits = {
            SummaryLength.BRIEF: 50,
            SummaryLength.SHORT: 150,
            SummaryLength.MEDIUM: 300,
            SummaryLength.LONG: 500,
            SummaryLength.DETAILED: 800
        }
        
        return token_limits.get(length, 300)
    
    def _extract_key_points_from_summary(self, summary: str) -> List[str]:
        """Extract key points from generated summary"""
        # Split by bullet points if present
        if '•' in summary or '*' in summary:
            points = re.split(r'[•*]', summary)
            return [p.strip() for p in points if p.strip()]
        
        # Split by numbered points
        numbered_pattern = r'\d+\.'  
        if re.search(numbered_pattern, summary):
            points = re.split(numbered_pattern, summary)
            return [p.strip() for p in points if p.strip()]
        
        # Split by sentences as fallback
        sentences = re.split(r'[.!?]+', summary)
        return [s.strip() for s in sentences if s.strip()][:5]

class HybridSummarizer(BaseSummarizer):
    """Hybrid summarization combining extractive and abstractive approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.extractive = ExtractiveSummarizer(config)
        self.llm_summarizer = LLMSummarizer(config)
    
    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Generate hybrid summary"""
        start_time = time.time()
        
        # First, get extractive summary to identify key content
        extractive_request = SummaryRequest(
            content=request.content,
            content_type=request.content_type,
            summary_type=SummaryType.EXTRACTIVE,
            summary_length=SummaryLength.MEDIUM,  # Get more content for LLM processing
            focus_areas=request.focus_areas,
            exclude_topics=request.exclude_topics
        )
        
        extractive_result = await self.extractive.summarize(extractive_request)
        
        # If extractive summary is short enough, use LLM to refine it
        if len(extractive_result.summary_text.split()) < 500:
            llm_request = SummaryRequest(
                content=extractive_result.summary_text,
                content_type=ContentType.DOCUMENT,
                summary_type=request.summary_type,
                summary_length=request.summary_length,
                focus_areas=request.focus_areas,
                exclude_topics=request.exclude_topics,
                target_audience=request.target_audience,
                include_statistics=request.include_statistics,
                include_timeline=request.include_timeline,
                include_key_entities=request.include_key_entities,
                custom_instructions=request.custom_instructions
            )
            
            llm_result = await self.llm_summarizer.summarize(llm_request)
            
            # Combine results
            hybrid_result = SummaryResult(
                summary_text=llm_result.summary_text,
                summary_type=SummaryType.HYBRID,
                summary_length=request.summary_length,
                key_points=extractive_result.key_points + llm_result.key_points,
                key_entities=list(set(extractive_result.key_entities + llm_result.key_entities)),
                themes=list(set(extractive_result.themes + llm_result.themes)),
                statistics={
                    'extractive_stats': extractive_result.statistics,
                    'llm_stats': llm_result.statistics
                },
                confidence_score=(extractive_result.confidence_score + llm_result.confidence_score) / 2,
                processing_time=time.time() - start_time,
                word_count=len(llm_result.summary_text.split()),
                compression_ratio=llm_result.compression_ratio,
                metadata={
                    'approach': 'hybrid',
                    'extractive_confidence': extractive_result.confidence_score,
                    'llm_confidence': llm_result.confidence_score
                }
            )
            
            return hybrid_result
        
        else:
            # Content too long for LLM, return enhanced extractive summary
            extractive_result.summary_type = SummaryType.HYBRID
            extractive_result.metadata = {'approach': 'extractive_only', 'reason': 'content_too_long'}
            return extractive_result

class TimelineSummarizer(BaseSummarizer):
    """Specialized summarizer for temporal content"""
    
    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Generate timeline summary"""
        start_time = time.time()
        
        # Extract temporal information
        timeline_data = self._extract_timeline_data(request.content, request.content_type)
        
        if not timeline_data:
            return SummaryResult(
                summary_text="No temporal information found to create timeline.",
                summary_type=SummaryType.TIMELINE,
                summary_length=request.summary_length,
                processing_time=time.time() - start_time
            )
        
        # Sort by timestamp
        timeline_data.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        # Create timeline summary
        timeline_text = self._format_timeline(timeline_data, request.summary_length)
        
        # Extract additional information
        full_text = " ".join([item.get('description', '') for item in timeline_data])
        key_entities = self._extract_key_entities(full_text)
        themes = self._extract_themes(full_text)
        
        # Create timeline for result
        timeline = [{
            'timestamp': item.get('timestamp', datetime.now()).isoformat(),
            'event': item.get('description', ''),
            'importance': item.get('importance', 0.5)
        } for item in timeline_data]
        
        processing_time = time.time() - start_time
        
        return SummaryResult(
            summary_text=timeline_text,
            summary_type=SummaryType.TIMELINE,
            summary_length=request.summary_length,
            key_entities=key_entities,
            themes=themes,
            timeline=timeline,
            statistics={
                'total_events': len(timeline_data),
                'time_span': self._calculate_time_span(timeline_data)
            },
            confidence_score=0.8,  # Timeline summaries are generally reliable
            processing_time=processing_time,
            word_count=len(timeline_text.split())
        )
    
    def _extract_timeline_data(self, content: Any, content_type: ContentType) -> List[Dict[str, Any]]:
        """Extract timeline data from content"""
        timeline_data = []
        
        if content_type == ContentType.TEMPORAL_SEQUENCE:
            if isinstance(content, dict) and 'events' in content:
                for event in content['events']:
                    if isinstance(event, dict):
                        timeline_data.append({
                            'timestamp': datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat())),
                            'description': event.get('content', ''),
                            'importance': event.get('importance', 0.5)
                        })
        
        elif content_type == ContentType.EPISODE:
            if isinstance(content, dict):
                timeline_data.append({
                    'timestamp': datetime.fromisoformat(content.get('start_time', datetime.now().isoformat())),
                    'description': f"Episode started: {content.get('title', '')}",
                    'importance': content.get('importance_score', 0.5)
                })
                
                if content.get('end_time'):
                    timeline_data.append({
                        'timestamp': datetime.fromisoformat(content['end_time']),
                        'description': f"Episode ended: {content.get('outcome', '')}",
                        'importance': content.get('importance_score', 0.5)
                    })
        
        return timeline_data
    
    def _format_timeline(self, timeline_data: List[Dict[str, Any]], length: SummaryLength) -> str:
        """Format timeline data into text"""
        if not timeline_data:
            return "No timeline events found."
        
        # Determine how many events to include based on length
        max_events = {
            SummaryLength.BRIEF: 3,
            SummaryLength.SHORT: 5,
            SummaryLength.MEDIUM: 8,
            SummaryLength.LONG: 12,
            SummaryLength.DETAILED: len(timeline_data)
        }.get(length, 8)
        
        # Select most important events if we need to limit
        if len(timeline_data) > max_events:
            timeline_data = sorted(timeline_data, key=lambda x: x.get('importance', 0), reverse=True)[:max_events]
            timeline_data.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        # Format timeline
        timeline_parts = []
        for i, event in enumerate(timeline_data):
            timestamp = event.get('timestamp', datetime.now())
            description = event.get('description', '')
            
            if length in [SummaryLength.BRIEF, SummaryLength.SHORT]:
                # Compact format
                time_str = timestamp.strftime("%Y-%m-%d %H:%M")
                timeline_parts.append(f"{time_str}: {description}")
            else:
                # Detailed format
                time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                timeline_parts.append(f"[{time_str}] {description}")
        
        return "\n".join(timeline_parts)
    
    def _calculate_time_span(self, timeline_data: List[Dict[str, Any]]) -> str:
        """Calculate time span of timeline"""
        if len(timeline_data) < 2:
            return "Single event"
        
        timestamps = [item.get('timestamp', datetime.now()) for item in timeline_data]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        duration = end_time - start_time
        
        if duration.days > 0:
            return f"{duration.days} days"
        elif duration.seconds > 3600:
            hours = duration.seconds // 3600
            return f"{hours} hours"
        else:
            minutes = duration.seconds // 60
            return f"{minutes} minutes"

class IntelligentSummarizationManager:
    """Main manager for intelligent summarization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize summarizers
        self.extractive = ExtractiveSummarizer(config)
        self.llm_summarizer = LLMSummarizer(config)
        self.hybrid = HybridSummarizer(config)
        self.timeline = TimelineSummarizer(config)
        
        # Cache for recent summaries
        self.summary_cache: Dict[str, SummaryResult] = {}
        self.cache_max_size = config.get('cache_max_size', 1000)
        
        # Statistics
        self.total_summaries_generated = 0
        self.summarizer_usage = defaultdict(int)
        
    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Generate summary using appropriate summarizer"""
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.summary_cache:
            logger.debug(f"Returning cached summary for key: {cache_key[:50]}...")
            return self.summary_cache[cache_key]
        
        # Select appropriate summarizer
        summarizer = self._select_summarizer(request)
        
        # Generate summary
        result = await summarizer.summarize(request)
        
        # Cache result
        self._cache_summary(cache_key, result)
        
        # Update statistics
        self.total_summaries_generated += 1
        self.summarizer_usage[summarizer.name] += 1
        
        logger.info(f"Generated {request.summary_type.value} summary using {summarizer.name}")
        
        return result
    
    async def summarize_memories(self, memories: List[Dict[str, Any]], 
                               summary_type: SummaryType = SummaryType.HYBRID,
                               summary_length: SummaryLength = SummaryLength.MEDIUM) -> SummaryResult:
        """Summarize a collection of memories"""
        
        request = SummaryRequest(
            content={'memories': memories},
            content_type=ContentType.SEARCH_RESULTS,
            summary_type=summary_type,
            summary_length=summary_length,
            include_key_entities=True,
            include_statistics=True
        )
        
        return await self.summarize(request)
    
    async def summarize_episode(self, episode: Dict[str, Any],
                              summary_type: SummaryType = SummaryType.HYBRID,
                              summary_length: SummaryLength = SummaryLength.MEDIUM) -> SummaryResult:
        """Summarize an episode"""
        
        request = SummaryRequest(
            content=episode,
            content_type=ContentType.EPISODE,
            summary_type=summary_type,
            summary_length=summary_length,
            include_timeline=True,
            include_key_entities=True
        )
        
        return await self.summarize(request)
    
    async def create_temporal_summary(self, temporal_sequence: Dict[str, Any],
                                    summary_length: SummaryLength = SummaryLength.MEDIUM) -> SummaryResult:
        """Create timeline-based summary"""
        
        request = SummaryRequest(
            content=temporal_sequence,
            content_type=ContentType.TEMPORAL_SEQUENCE,
            summary_type=SummaryType.TIMELINE,
            summary_length=summary_length,
            include_timeline=True,
            include_statistics=True
        )
        
        return await self.summarize(request)
    
    async def create_thematic_summary(self, content: Any, content_type: ContentType,
                                    themes: List[str],
                                    summary_length: SummaryLength = SummaryLength.MEDIUM) -> SummaryResult:
        """Create theme-organized summary"""
        
        request = SummaryRequest(
            content=content,
            content_type=content_type,
            summary_type=SummaryType.THEMATIC,
            summary_length=summary_length,
            focus_areas=themes,
            include_key_entities=True
        )
        
        return await self.summarize(request)
    
    def get_summarization_statistics(self) -> Dict[str, Any]:
        """Get summarization statistics"""
        
        return {
            'total_summaries': self.total_summaries_generated,
            'summarizer_usage': dict(self.summarizer_usage),
            'cache_size': len(self.summary_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'available_summarizers': {
                'extractive': True,
                'llm': self.llm_summarizer.llm_provider is not None,
                'hybrid': True,
                'timeline': True
            }
        }
    
    def clear_cache(self):
        """Clear summary cache"""
        self.summary_cache.clear()
        logger.info("Summary cache cleared")
    
    # Private methods
    
    def _select_summarizer(self, request: SummaryRequest) -> BaseSummarizer:
        """Select appropriate summarizer based on request"""
        
        if request.summary_type == SummaryType.EXTRACTIVE:
            return self.extractive
        elif request.summary_type == SummaryType.ABSTRACTIVE:
            return self.llm_summarizer
        elif request.summary_type == SummaryType.TIMELINE:
            return self.timeline
        elif request.summary_type == SummaryType.HYBRID:
            return self.hybrid
        else:
            # Default to hybrid for best results
            return self.hybrid
    
    def _generate_cache_key(self, request: SummaryRequest) -> str:
        """Generate cache key for request"""
        # Create a hash of the request parameters
        key_data = {
            'content': str(request.content)[:1000],  # Limit content length for key
            'content_type': request.content_type.value,
            'summary_type': request.summary_type.value,
            'summary_length': request.summary_length.value,
            'focus_areas': request.focus_areas,
            'exclude_topics': request.exclude_topics,
            'target_audience': request.target_audience
        }
        
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _cache_summary(self, cache_key: str, result: SummaryResult):
        """Cache summary result"""
        # Remove oldest entries if cache is full
        if len(self.summary_cache) >= self.cache_max_size:
            # Remove 10% of oldest entries
            keys_to_remove = list(self.summary_cache.keys())[:self.cache_max_size // 10]
            for key in keys_to_remove:
                del self.summary_cache[key]
        
        self.summary_cache[cache_key] = result
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This would need to be tracked separately in a real implementation
        return 0.0  # Placeholder

# Default configuration
DEFAULT_SUMMARIZATION_CONFIG = {
    'cache_max_size': 1000,
    'llm_config': {
        'provider': 'openai',
        'model': 'gpt-3.5-turbo',
        'api_key': None  # Should be provided by user
    }
}

def create_summarization_manager(config: Optional[Dict[str, Any]] = None) -> IntelligentSummarizationManager:
    """Create summarization manager with default or custom config"""
    if config is None:
        config = DEFAULT_SUMMARIZATION_CONFIG
    
    return IntelligentSummarizationManager(config)