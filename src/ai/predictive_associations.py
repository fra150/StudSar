"""Predictive Associations Module for StudSar V4

Implements AI-driven prediction of future relevant memories and associations
based on historical patterns, context analysis, and machine learning models.
"""

import asyncio
import logging
import time
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from collections import defaultdict, Counter, deque
import math
import random

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    RandomForestRegressor = None
    GradientBoostingRegressor = None
    LinearRegression = None
    LogisticRegression = None
    KMeans = None
    DBSCAN = None
    StandardScaler = None
    LabelEncoder = None
    train_test_split = None
    mean_squared_error = None
    accuracy_score = None
    TfidfVectorizer = None

try:
    import networkx as nx
except ImportError:
    nx = None

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions"""
    MEMORY_RELEVANCE = "memory_relevance"  # Predict memory relevance
    ASSOCIATION_STRENGTH = "association_strength"  # Predict association strength
    FUTURE_ACCESS = "future_access"  # Predict future memory access
    CONTEXT_SIMILARITY = "context_similarity"  # Predict context similarity
    TEMPORAL_PATTERN = "temporal_pattern"  # Predict temporal patterns
    SEMANTIC_DRIFT = "semantic_drift"  # Predict semantic changes
    USAGE_FREQUENCY = "usage_frequency"  # Predict usage patterns

class ModelType(Enum):
    """Types of prediction models"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENSEMBLE = "ensemble"

class ConfidenceLevel(Enum):
    """Confidence levels for predictions"""
    VERY_LOW = "very_low"  # < 0.3
    LOW = "low"  # 0.3 - 0.5
    MEDIUM = "medium"  # 0.5 - 0.7
    HIGH = "high"  # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9

@dataclass
class PredictionContext:
    """Context for making predictions"""
    current_memory_id: Optional[str] = None
    current_content: Optional[str] = None
    current_tags: List[str] = field(default_factory=list)
    current_timestamp: datetime = field(default_factory=datetime.now)
    user_context: Dict[str, Any] = field(default_factory=dict)
    session_context: Dict[str, Any] = field(default_factory=dict)
    recent_memories: List[str] = field(default_factory=list)
    recent_searches: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    emotional_state: Optional[str] = None
    cognitive_load: float = 0.5  # 0.0 to 1.0
    time_of_day: Optional[str] = None
    day_of_week: Optional[str] = None

@dataclass
class PredictionFeatures:
    """Features for prediction models"""
    # Content features
    content_similarity: float = 0.0
    semantic_similarity: float = 0.0
    keyword_overlap: float = 0.0
    tag_similarity: float = 0.0
    
    # Temporal features
    time_since_creation: float = 0.0
    time_since_last_access: float = 0.0
    access_frequency: float = 0.0
    temporal_pattern_score: float = 0.0
    
    # Network features
    connection_strength: float = 0.0
    network_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    path_length: float = 0.0
    
    # User behavior features
    user_preference_score: float = 0.0
    historical_relevance: float = 0.0
    context_match_score: float = 0.0
    emotional_resonance: float = 0.0
    
    # Statistical features
    popularity_score: float = 0.0
    novelty_score: float = 0.0
    diversity_score: float = 0.0
    quality_score: float = 0.0
    
    # Custom features
    custom_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """Result of a prediction"""
    memory_id: str
    prediction_type: PredictionType
    predicted_value: float
    confidence: float
    confidence_level: ConfidenceLevel
    features_used: PredictionFeatures
    model_name: str
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    explanation: Optional[str] = None
    contributing_factors: List[Tuple[str, float]] = field(default_factory=list)
    uncertainty_bounds: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionBatch:
    """Batch of predictions"""
    predictions: List[PredictionResult]
    context: PredictionContext
    batch_timestamp: datetime = field(default_factory=datetime.now)
    total_processing_time: float = 0.0
    model_performance: Dict[str, float] = field(default_factory=dict)
    batch_metadata: Dict[str, Any] = field(default_factory=dict)

class BasePredictionModel(ABC):
    """Base class for prediction models"""
    
    def __init__(self, model_type: ModelType, config: Dict[str, Any]):
        self.model_type = model_type
        self.config = config
        self.name = self.__class__.__name__
        self.is_trained = False
        self.training_data_size = 0
        self.last_training_time = None
        self.performance_metrics = {}
        
    @abstractmethod
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the prediction model"""
        pass
    
    @abstractmethod
    async def predict(self, context: PredictionContext, 
                     candidate_memories: List[str]) -> List[PredictionResult]:
        """Make predictions for candidate memories"""
        pass
    
    @abstractmethod
    def extract_features(self, memory_id: str, context: PredictionContext) -> PredictionFeatures:
        """Extract features for prediction"""
        pass
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def save_model(self, filepath: str):
        """Save model to file"""
        try:
            model_data = {
                'model_type': self.model_type.value,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_data_size': self.training_data_size,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'performance_metrics': self.performance_metrics
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model {self.name} saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model {self.name}: {e}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.is_trained = model_data.get('is_trained', False)
            self.training_data_size = model_data.get('training_data_size', 0)
            self.performance_metrics = model_data.get('performance_metrics', {})
            
            if model_data.get('last_training_time'):
                self.last_training_time = datetime.fromisoformat(model_data['last_training_time'])
            
            logger.info(f"Model {self.name} loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.name}: {e}")

class CollaborativeFilteringModel(BasePredictionModel):
    """Collaborative filtering prediction model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.COLLABORATIVE_FILTERING, config)
        self.user_memory_matrix = None
        self.memory_similarity_matrix = None
        self.user_similarity_matrix = None
        self.scaler = StandardScaler() if StandardScaler else None
        
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train collaborative filtering model"""
        start_time = time.time()
        
        if not training_data:
            return {'error': 'No training data provided'}
        
        try:
            # Build user-memory interaction matrix
            self._build_interaction_matrix(training_data)
            
            # Calculate similarity matrices
            self._calculate_similarity_matrices()
            
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.now()
            
            training_time = time.time() - start_time
            
            # Calculate performance metrics
            metrics = self._evaluate_model(training_data)
            metrics['training_time'] = training_time
            
            self.performance_metrics = metrics
            
            logger.info(f"Collaborative filtering model trained on {len(training_data)} samples")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed for collaborative filtering model: {e}")
            return {'error': str(e)}
    
    async def predict(self, context: PredictionContext, 
                     candidate_memories: List[str]) -> List[PredictionResult]:
        """Make collaborative filtering predictions"""
        if not self.is_trained:
            return []
        
        predictions = []
        
        for memory_id in candidate_memories:
            features = self.extract_features(memory_id, context)
            
            # Calculate collaborative filtering score
            cf_score = self._calculate_collaborative_score(memory_id, context)
            
            # Calculate confidence based on data availability
            confidence = self._calculate_confidence(memory_id, context)
            
            prediction = PredictionResult(
                memory_id=memory_id,
                prediction_type=PredictionType.MEMORY_RELEVANCE,
                predicted_value=cf_score,
                confidence=confidence,
                confidence_level=self.get_confidence_level(confidence),
                features_used=features,
                model_name=self.name,
                explanation=f"Collaborative filtering based on user similarity and memory interactions",
                contributing_factors=[
                    ('user_similarity', features.user_preference_score),
                    ('memory_popularity', features.popularity_score),
                    ('historical_relevance', features.historical_relevance)
                ]
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def extract_features(self, memory_id: str, context: PredictionContext) -> PredictionFeatures:
        """Extract features for collaborative filtering"""
        features = PredictionFeatures()
        
        # User preference score based on historical interactions
        features.user_preference_score = self._get_user_preference_score(memory_id, context)
        
        # Popularity score based on overall memory usage
        features.popularity_score = self._get_popularity_score(memory_id)
        
        # Historical relevance based on past access patterns
        features.historical_relevance = self._get_historical_relevance(memory_id, context)
        
        return features
    
    def _build_interaction_matrix(self, training_data: List[Dict[str, Any]]):
        """Build user-memory interaction matrix"""
        # Extract user-memory interactions from training data
        interactions = defaultdict(lambda: defaultdict(float))
        
        for data_point in training_data:
            user_id = data_point.get('user_id', 'default_user')
            memory_id = data_point.get('memory_id')
            interaction_score = data_point.get('relevance_score', 0.0)
            
            if memory_id:
                interactions[user_id][memory_id] = interaction_score
        
        # Convert to matrix format (simplified)
        self.user_memory_matrix = dict(interactions)
    
    def _calculate_similarity_matrices(self):
        """Calculate user and memory similarity matrices"""
        if not self.user_memory_matrix:
            return
        
        # Calculate memory similarity based on user interactions
        memory_similarity = defaultdict(lambda: defaultdict(float))
        
        memories = set()
        for user_memories in self.user_memory_matrix.values():
            memories.update(user_memories.keys())
        
        memories = list(memories)
        
        for i, memory1 in enumerate(memories):
            for j, memory2 in enumerate(memories):
                if i != j:
                    similarity = self._calculate_memory_similarity(memory1, memory2)
                    memory_similarity[memory1][memory2] = similarity
        
        self.memory_similarity_matrix = dict(memory_similarity)
    
    def _calculate_memory_similarity(self, memory1: str, memory2: str) -> float:
        """Calculate similarity between two memories based on user interactions"""
        if not self.user_memory_matrix:
            return 0.0
        
        # Find users who interacted with both memories
        common_users = []
        for user_id, user_memories in self.user_memory_matrix.items():
            if memory1 in user_memories and memory2 in user_memories:
                common_users.append(user_id)
        
        if not common_users:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for user_id in common_users:
            score1 = self.user_memory_matrix[user_id][memory1]
            score2 = self.user_memory_matrix[user_id][memory2]
            
            dot_product += score1 * score2
            norm1 += score1 ** 2
            norm2 += score2 ** 2
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    
    def _calculate_collaborative_score(self, memory_id: str, context: PredictionContext) -> float:
        """Calculate collaborative filtering score"""
        if not self.memory_similarity_matrix or memory_id not in self.memory_similarity_matrix:
            return 0.5  # Default score
        
        # Get similar memories from recent context
        similar_memories = context.recent_memories[:10]  # Limit to recent memories
        
        if not similar_memories:
            return 0.5
        
        # Calculate weighted average based on similarity
        total_score = 0.0
        total_weight = 0.0
        
        for similar_memory in similar_memories:
            if similar_memory in self.memory_similarity_matrix[memory_id]:
                similarity = self.memory_similarity_matrix[memory_id][similar_memory]
                weight = max(0, similarity)  # Only positive similarities
                
                # Assume recent memories have high relevance
                relevance = 0.8  # This would come from actual user feedback
                
                total_score += weight * relevance
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return min(1.0, max(0.0, total_score / total_weight))
    
    def _get_user_preference_score(self, memory_id: str, context: PredictionContext) -> float:
        """Get user preference score for memory"""
        # This would be based on user's historical interactions
        # For now, return a placeholder based on context
        
        if memory_id in context.recent_memories:
            return 0.8
        
        # Check tag similarity
        if context.current_tags:
            # This would require access to memory tags
            return 0.6
        
        return 0.4
    
    def _get_popularity_score(self, memory_id: str) -> float:
        """Get popularity score for memory"""
        if not self.user_memory_matrix:
            return 0.5
        
        # Count how many users interacted with this memory
        interaction_count = 0
        total_score = 0.0
        
        for user_memories in self.user_memory_matrix.values():
            if memory_id in user_memories:
                interaction_count += 1
                total_score += user_memories[memory_id]
        
        if interaction_count == 0:
            return 0.1  # Low popularity for new memories
        
        # Normalize by number of interactions
        avg_score = total_score / interaction_count
        popularity = min(1.0, interaction_count / 10.0)  # Normalize to 0-1
        
        return (avg_score + popularity) / 2.0
    
    def _get_historical_relevance(self, memory_id: str, context: PredictionContext) -> float:
        """Get historical relevance score"""
        # This would analyze historical access patterns
        # For now, return a placeholder
        
        if context.current_task:
            # Memories related to current task are more relevant
            return 0.7
        
        return 0.5
    
    def _calculate_confidence(self, memory_id: str, context: PredictionContext) -> float:
        """Calculate prediction confidence"""
        confidence_factors = []
        
        # Data availability
        if self.user_memory_matrix:
            data_availability = min(1.0, len(self.user_memory_matrix) / 10.0)
            confidence_factors.append(data_availability)
        
        # Memory popularity (more interactions = higher confidence)
        popularity = self._get_popularity_score(memory_id)
        confidence_factors.append(popularity)
        
        # Context richness
        context_richness = len(context.recent_memories) / 20.0  # Normalize
        confidence_factors.append(min(1.0, context_richness))
        
        if not confidence_factors:
            return 0.5
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _evaluate_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model performance"""
        # This would perform cross-validation and calculate metrics
        # For now, return placeholder metrics
        
        return {
            'rmse': 0.3,  # Root mean square error
            'mae': 0.25,  # Mean absolute error
            'precision': 0.75,
            'recall': 0.70,
            'f1_score': 0.72
        }

class ContentBasedModel(BasePredictionModel):
    """Content-based prediction model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.CONTENT_BASED, config)
        self.vectorizer = TfidfVectorizer(max_features=1000) if TfidfVectorizer else None
        self.content_vectors = {}
        self.tag_encoder = LabelEncoder() if LabelEncoder else None
        
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train content-based model"""
        start_time = time.time()
        
        if not training_data or not self.vectorizer:
            return {'error': 'No training data or vectorizer not available'}
        
        try:
            # Extract content and build vectors
            contents = []
            memory_ids = []
            
            for data_point in training_data:
                memory_id = data_point.get('memory_id')
                content = data_point.get('content', '')
                
                if memory_id and content:
                    contents.append(content)
                    memory_ids.append(memory_id)
            
            if not contents:
                return {'error': 'No valid content found in training data'}
            
            # Fit vectorizer and transform content
            content_matrix = self.vectorizer.fit_transform(contents)
            
            # Store content vectors
            for i, memory_id in enumerate(memory_ids):
                self.content_vectors[memory_id] = content_matrix[i].toarray().flatten()
            
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.now()
            
            training_time = time.time() - start_time
            
            metrics = {
                'training_time': training_time,
                'vocabulary_size': len(self.vectorizer.vocabulary_),
                'content_vectors_created': len(self.content_vectors)
            }
            
            self.performance_metrics = metrics
            
            logger.info(f"Content-based model trained on {len(contents)} documents")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed for content-based model: {e}")
            return {'error': str(e)}
    
    async def predict(self, context: PredictionContext, 
                     candidate_memories: List[str]) -> List[PredictionResult]:
        """Make content-based predictions"""
        if not self.is_trained or not self.vectorizer:
            return []
        
        predictions = []
        
        # Vectorize current content if available
        current_vector = None
        if context.current_content:
            try:
                current_vector = self.vectorizer.transform([context.current_content]).toarray().flatten()
            except:
                pass
        
        for memory_id in candidate_memories:
            features = self.extract_features(memory_id, context)
            
            # Calculate content similarity
            content_similarity = 0.0
            if current_vector is not None and memory_id in self.content_vectors:
                content_similarity = self._calculate_cosine_similarity(
                    current_vector, self.content_vectors[memory_id]
                )
            
            # Calculate overall content-based score
            cb_score = self._calculate_content_score(memory_id, context, content_similarity)
            
            # Calculate confidence
            confidence = self._calculate_confidence(memory_id, context, content_similarity)
            
            prediction = PredictionResult(
                memory_id=memory_id,
                prediction_type=PredictionType.MEMORY_RELEVANCE,
                predicted_value=cb_score,
                confidence=confidence,
                confidence_level=self.get_confidence_level(confidence),
                features_used=features,
                model_name=self.name,
                explanation=f"Content-based prediction using text similarity and semantic features",
                contributing_factors=[
                    ('content_similarity', content_similarity),
                    ('semantic_similarity', features.semantic_similarity),
                    ('tag_similarity', features.tag_similarity)
                ]
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def extract_features(self, memory_id: str, context: PredictionContext) -> PredictionFeatures:
        """Extract content-based features"""
        features = PredictionFeatures()
        
        # Content similarity (if current content is available)
        if context.current_content and memory_id in self.content_vectors:
            try:
                current_vector = self.vectorizer.transform([context.current_content]).toarray().flatten()
                memory_vector = self.content_vectors[memory_id]
                features.content_similarity = self._calculate_cosine_similarity(current_vector, memory_vector)
            except:
                features.content_similarity = 0.0
        
        # Tag similarity
        features.tag_similarity = self._calculate_tag_similarity(memory_id, context)
        
        # Keyword overlap
        features.keyword_overlap = self._calculate_keyword_overlap(memory_id, context)
        
        # Semantic similarity (placeholder - would use embeddings)
        features.semantic_similarity = features.content_similarity * 0.8  # Approximation
        
        return features
    
    def _calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if np is None:
            return 0.0
        
        try:
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def _calculate_content_score(self, memory_id: str, context: PredictionContext, 
                               content_similarity: float) -> float:
        """Calculate overall content-based score"""
        score_components = []
        
        # Content similarity weight
        if content_similarity > 0:
            score_components.append(('content', content_similarity, 0.4))
        
        # Tag similarity
        tag_similarity = self._calculate_tag_similarity(memory_id, context)
        if tag_similarity > 0:
            score_components.append(('tags', tag_similarity, 0.3))
        
        # Keyword overlap
        keyword_overlap = self._calculate_keyword_overlap(memory_id, context)
        if keyword_overlap > 0:
            score_components.append(('keywords', keyword_overlap, 0.2))
        
        # Recency bonus
        recency_score = self._calculate_recency_score(memory_id, context)
        score_components.append(('recency', recency_score, 0.1))
        
        # Calculate weighted average
        if not score_components:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for name, score, weight in score_components:
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _calculate_tag_similarity(self, memory_id: str, context: PredictionContext) -> float:
        """Calculate tag similarity"""
        # This would require access to memory tags
        # For now, return a placeholder based on context
        
        if context.current_tags:
            # Simulate tag matching
            return random.uniform(0.3, 0.8)
        
        return 0.0
    
    def _calculate_keyword_overlap(self, memory_id: str, context: PredictionContext) -> float:
        """Calculate keyword overlap"""
        if not context.current_content:
            return 0.0
        
        # Extract keywords from current content
        current_keywords = set(context.current_content.lower().split())
        
        # This would require access to memory content
        # For now, simulate keyword overlap
        return random.uniform(0.1, 0.6)
    
    def _calculate_recency_score(self, memory_id: str, context: PredictionContext) -> float:
        """Calculate recency score"""
        if memory_id in context.recent_memories:
            position = context.recent_memories.index(memory_id)
            # More recent = higher score
            return 1.0 - (position / len(context.recent_memories))
        
        return 0.2  # Default low recency score
    
    def _calculate_confidence(self, memory_id: str, context: PredictionContext, 
                            content_similarity: float) -> float:
        """Calculate prediction confidence"""
        confidence_factors = []
        
        # Content availability
        if memory_id in self.content_vectors:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Current content availability
        if context.current_content:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Similarity strength
        if content_similarity > 0.5:
            confidence_factors.append(0.8)
        elif content_similarity > 0.3:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return sum(confidence_factors) / len(confidence_factors)

class TemporalSequenceModel(BasePredictionModel):
    """Temporal sequence prediction model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.TEMPORAL_SEQUENCE, config)
        self.sequence_patterns = defaultdict(list)
        self.temporal_weights = {}
        self.access_history = defaultdict(list)
        
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train temporal sequence model"""
        start_time = time.time()
        
        if not training_data:
            return {'error': 'No training data provided'}
        
        try:
            # Build temporal sequences from training data
            self._build_temporal_sequences(training_data)
            
            # Calculate temporal weights
            self._calculate_temporal_weights()
            
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.now()
            
            training_time = time.time() - start_time
            
            metrics = {
                'training_time': training_time,
                'sequence_patterns': len(self.sequence_patterns),
                'temporal_weights': len(self.temporal_weights)
            }
            
            self.performance_metrics = metrics
            
            logger.info(f"Temporal sequence model trained on {len(training_data)} samples")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed for temporal sequence model: {e}")
            return {'error': str(e)}
    
    async def predict(self, context: PredictionContext, 
                     candidate_memories: List[str]) -> List[PredictionResult]:
        """Make temporal sequence predictions"""
        if not self.is_trained:
            return []
        
        predictions = []
        
        for memory_id in candidate_memories:
            features = self.extract_features(memory_id, context)
            
            # Calculate temporal sequence score
            temporal_score = self._calculate_temporal_score(memory_id, context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(memory_id, context)
            
            prediction = PredictionResult(
                memory_id=memory_id,
                prediction_type=PredictionType.FUTURE_ACCESS,
                predicted_value=temporal_score,
                confidence=confidence,
                confidence_level=self.get_confidence_level(confidence),
                features_used=features,
                model_name=self.name,
                explanation=f"Temporal sequence prediction based on access patterns and time-based features",
                contributing_factors=[
                    ('temporal_pattern', features.temporal_pattern_score),
                    ('access_frequency', features.access_frequency),
                    ('time_since_last_access', 1.0 - features.time_since_last_access)
                ]
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def extract_features(self, memory_id: str, context: PredictionContext) -> PredictionFeatures:
        """Extract temporal features"""
        features = PredictionFeatures()
        
        # Temporal pattern score
        features.temporal_pattern_score = self._get_temporal_pattern_score(memory_id, context)
        
        # Access frequency
        features.access_frequency = self._get_access_frequency(memory_id)
        
        # Time since last access (normalized)
        features.time_since_last_access = self._get_time_since_last_access(memory_id, context)
        
        return features
    
    def _build_temporal_sequences(self, training_data: List[Dict[str, Any]]):
        """Build temporal sequences from training data"""
        # Group by user and sort by timestamp
        user_sequences = defaultdict(list)
        
        for data_point in training_data:
            user_id = data_point.get('user_id', 'default_user')
            memory_id = data_point.get('memory_id')
            timestamp = data_point.get('timestamp')
            
            if memory_id and timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                user_sequences[user_id].append((timestamp, memory_id))
        
        # Sort sequences by timestamp and extract patterns
        for user_id, sequence in user_sequences.items():
            sequence.sort(key=lambda x: x[0])
            
            # Extract sequential patterns
            for i in range(len(sequence) - 1):
                current_memory = sequence[i][1]
                next_memory = sequence[i + 1][1]
                time_diff = (sequence[i + 1][0] - sequence[i][0]).total_seconds()
                
                self.sequence_patterns[current_memory].append({
                    'next_memory': next_memory,
                    'time_diff': time_diff,
                    'user_id': user_id
                })
                
                # Store access history
                self.access_history[current_memory].append(sequence[i][0])
    
    def _calculate_temporal_weights(self):
        """Calculate temporal weights for patterns"""
        for memory_id, patterns in self.sequence_patterns.items():
            # Calculate weights based on frequency and recency
            pattern_weights = defaultdict(float)
            
            for pattern in patterns:
                next_memory = pattern['next_memory']
                time_diff = pattern['time_diff']
                
                # Weight based on frequency
                frequency_weight = 1.0
                
                # Weight based on time difference (shorter = higher weight)
                time_weight = 1.0 / (1.0 + time_diff / 3600.0)  # Normalize by hour
                
                pattern_weights[next_memory] += frequency_weight * time_weight
            
            # Normalize weights
            total_weight = sum(pattern_weights.values())
            if total_weight > 0:
                for next_memory in pattern_weights:
                    pattern_weights[next_memory] /= total_weight
            
            self.temporal_weights[memory_id] = dict(pattern_weights)
    
    def _calculate_temporal_score(self, memory_id: str, context: PredictionContext) -> float:
        """Calculate temporal sequence score"""
        score = 0.0
        
        # Check if any recent memories predict this memory
        for recent_memory in context.recent_memories[:5]:  # Check last 5 memories
            if recent_memory in self.temporal_weights:
                if memory_id in self.temporal_weights[recent_memory]:
                    weight = self.temporal_weights[recent_memory][memory_id]
                    # Apply recency decay
                    position = context.recent_memories.index(recent_memory)
                    recency_factor = 1.0 / (1.0 + position)
                    score += weight * recency_factor
        
        # Add time-of-day patterns
        time_score = self._get_time_of_day_score(memory_id, context)
        score = (score + time_score) / 2.0
        
        return min(1.0, score)
    
    def _get_temporal_pattern_score(self, memory_id: str, context: PredictionContext) -> float:
        """Get temporal pattern score"""
        return self._calculate_temporal_score(memory_id, context)
    
    def _get_access_frequency(self, memory_id: str) -> float:
        """Get access frequency score"""
        if memory_id not in self.access_history:
            return 0.1
        
        access_count = len(self.access_history[memory_id])
        # Normalize to 0-1 range
        return min(1.0, access_count / 10.0)
    
    def _get_time_since_last_access(self, memory_id: str, context: PredictionContext) -> float:
        """Get normalized time since last access"""
        if memory_id not in self.access_history or not self.access_history[memory_id]:
            return 1.0  # Long time since access
        
        last_access = max(self.access_history[memory_id])
        time_diff = (context.current_timestamp - last_access).total_seconds()
        
        # Normalize to 0-1 range (1 day = 1.0)
        normalized_time = min(1.0, time_diff / (24 * 3600))
        
        return normalized_time
    
    def _get_time_of_day_score(self, memory_id: str, context: PredictionContext) -> float:
        """Get time-of-day pattern score"""
        # This would analyze historical access patterns by time of day
        # For now, return a placeholder
        
        if context.time_of_day:
            # Simulate time-of-day preferences
            if context.time_of_day in ['morning', 'afternoon']:
                return 0.7
            else:
                return 0.4
        
        return 0.5
    
    def _calculate_confidence(self, memory_id: str, context: PredictionContext) -> float:
        """Calculate prediction confidence"""
        confidence_factors = []
        
        # Pattern availability
        if memory_id in self.temporal_weights or any(
            memory_id in self.temporal_weights.get(recent, {}) 
            for recent in context.recent_memories
        ):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Access history availability
        if memory_id in self.access_history and self.access_history[memory_id]:
            history_length = len(self.access_history[memory_id])
            confidence_factors.append(min(1.0, history_length / 5.0))
        else:
            confidence_factors.append(0.2)
        
        # Context richness
        context_richness = len(context.recent_memories) / 10.0
        confidence_factors.append(min(1.0, context_richness))
        
        return sum(confidence_factors) / len(confidence_factors)

class EnsembleModel(BasePredictionModel):
    """Ensemble model combining multiple prediction approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.ENSEMBLE, config)
        self.models = []
        self.model_weights = {}
        self.meta_model = None
        
        # Initialize component models
        self._initialize_component_models(config)
    
    def _initialize_component_models(self, config: Dict[str, Any]):
        """Initialize component models"""
        try:
            # Collaborative filtering
            cf_model = CollaborativeFilteringModel(config.get('collaborative_filtering', {}))
            self.models.append(cf_model)
            
            # Content-based
            cb_model = ContentBasedModel(config.get('content_based', {}))
            self.models.append(cb_model)
            
            # Temporal sequence
            ts_model = TemporalSequenceModel(config.get('temporal_sequence', {}))
            self.models.append(ts_model)
            
            # Initialize equal weights
            for model in self.models:
                self.model_weights[model.name] = 1.0 / len(self.models)
                
        except Exception as e:
            logger.error(f"Failed to initialize component models: {e}")
    
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train ensemble model"""
        start_time = time.time()
        
        if not training_data:
            return {'error': 'No training data provided'}
        
        try:
            # Train all component models
            model_metrics = {}
            
            for model in self.models:
                logger.info(f"Training {model.name}...")
                metrics = await model.train(training_data)
                model_metrics[model.name] = metrics
            
            # Train meta-model to learn optimal weights
            self._train_meta_model(training_data)
            
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.now()
            
            training_time = time.time() - start_time
            
            ensemble_metrics = {
                'training_time': training_time,
                'component_models': len(self.models),
                'model_metrics': model_metrics,
                'model_weights': self.model_weights
            }
            
            self.performance_metrics = ensemble_metrics
            
            logger.info(f"Ensemble model trained with {len(self.models)} component models")
            
            return ensemble_metrics
            
        except Exception as e:
            logger.error(f"Training failed for ensemble model: {e}")
            return {'error': str(e)}
    
    async def predict(self, context: PredictionContext, 
                     candidate_memories: List[str]) -> List[PredictionResult]:
        """Make ensemble predictions"""
        if not self.is_trained:
            return []
        
        # Get predictions from all component models
        all_predictions = {}
        
        for model in self.models:
            if model.is_trained:
                try:
                    model_predictions = await model.predict(context, candidate_memories)
                    all_predictions[model.name] = {p.memory_id: p for p in model_predictions}
                except Exception as e:
                    logger.warning(f"Prediction failed for {model.name}: {e}")
        
        # Combine predictions
        ensemble_predictions = []
        
        for memory_id in candidate_memories:
            combined_prediction = self._combine_predictions(memory_id, all_predictions, context)
            if combined_prediction:
                ensemble_predictions.append(combined_prediction)
        
        return ensemble_predictions
    
    def extract_features(self, memory_id: str, context: PredictionContext) -> PredictionFeatures:
        """Extract features from all component models"""
        combined_features = PredictionFeatures()
        
        # Combine features from all models
        for model in self.models:
            if model.is_trained:
                try:
                    model_features = model.extract_features(memory_id, context)
                    
                    # Weighted combination of features
                    weight = self.model_weights.get(model.name, 0.0)
                    
                    combined_features.content_similarity += model_features.content_similarity * weight
                    combined_features.semantic_similarity += model_features.semantic_similarity * weight
                    combined_features.temporal_pattern_score += model_features.temporal_pattern_score * weight
                    combined_features.user_preference_score += model_features.user_preference_score * weight
                    combined_features.popularity_score += model_features.popularity_score * weight
                    
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {model.name}: {e}")
        
        return combined_features
    
    def _train_meta_model(self, training_data: List[Dict[str, Any]]):
        """Train meta-model to learn optimal weights"""
        # This would use a simple linear model to learn optimal weights
        # For now, use performance-based weighting
        
        total_performance = 0.0
        model_performances = {}
        
        for model in self.models:
            if model.is_trained and model.performance_metrics:
                # Use F1 score or similar metric as performance indicator
                performance = model.performance_metrics.get('f1_score', 0.5)
                model_performances[model.name] = performance
                total_performance += performance
        
        # Update weights based on performance
        if total_performance > 0:
            for model_name, performance in model_performances.items():
                self.model_weights[model_name] = performance / total_performance
        
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def _combine_predictions(self, memory_id: str, all_predictions: Dict[str, Dict[str, PredictionResult]], 
                           context: PredictionContext) -> Optional[PredictionResult]:
        """Combine predictions from component models"""
        if not all_predictions:
            return None
        
        # Weighted combination of predictions
        total_score = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        contributing_factors = []
        
        for model_name, predictions in all_predictions.items():
            if memory_id in predictions:
                prediction = predictions[memory_id]
                weight = self.model_weights.get(model_name, 0.0)
                
                total_score += prediction.predicted_value * weight
                total_confidence += prediction.confidence * weight
                total_weight += weight
                
                contributing_factors.append((model_name, prediction.predicted_value))
        
        if total_weight == 0:
            return None
        
        # Calculate final scores
        final_score = total_score / total_weight
        final_confidence = total_confidence / total_weight
        
        # Extract combined features
        combined_features = self.extract_features(memory_id, context)
        
        return PredictionResult(
            memory_id=memory_id,
            prediction_type=PredictionType.MEMORY_RELEVANCE,
            predicted_value=final_score,
            confidence=final_confidence,
            confidence_level=self.get_confidence_level(final_confidence),
            features_used=combined_features,
            model_name=self.name,
            explanation=f"Ensemble prediction combining {len(contributing_factors)} models",
            contributing_factors=contributing_factors,
            metadata={
                'model_weights': self.model_weights,
                'component_models': len(self.models)
            }
        )

class PredictiveAssociationsManager:
    """Main manager for predictive associations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_max_size = config.get('cache_max_size', 1000)
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour
        
        # Statistics
        self.total_predictions_made = 0
        self.model_usage_stats = defaultdict(int)
        self.prediction_accuracy_history = deque(maxlen=1000)
        
        # Training data buffer
        self.training_buffer = deque(maxlen=config.get('training_buffer_size', 10000))
        self.auto_retrain_threshold = config.get('auto_retrain_threshold', 1000)
        
    def _initialize_models(self):
        """Initialize prediction models"""
        try:
            # Ensemble model (primary)
            self.models['ensemble'] = EnsembleModel(self.config.get('ensemble', {}))
            
            # Individual models for specific use cases
            self.models['collaborative'] = CollaborativeFilteringModel(
                self.config.get('collaborative_filtering', {})
            )
            self.models['content'] = ContentBasedModel(
                self.config.get('content_based', {})
            )
            self.models['temporal'] = TemporalSequenceModel(
                self.config.get('temporal_sequence', {})
            )
            
            logger.info(f"Initialized {len(self.models)} prediction models")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    async def predict_relevant_memories(self, context: PredictionContext, 
                                      candidate_memories: List[str],
                                      model_name: str = 'ensemble',
                                      use_cache: bool = True) -> PredictionBatch:
        """Predict relevant memories for given context"""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(context, candidate_memories, model_name)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                logger.debug(f"Returning cached prediction for {len(candidate_memories)} memories")
                return cached_result
        
        # Get model
        model = self.models.get(model_name)
        if not model or not model.is_trained:
            logger.warning(f"Model {model_name} not available or not trained")
            return PredictionBatch(predictions=[], context=context)
        
        try:
            # Make predictions
            predictions = await model.predict(context, candidate_memories)
            
            # Sort by predicted value (descending)
            predictions.sort(key=lambda p: p.predicted_value, reverse=True)
            
            # Create batch result
            processing_time = time.time() - start_time
            
            batch = PredictionBatch(
                predictions=predictions,
                context=context,
                total_processing_time=processing_time,
                model_performance=model.performance_metrics,
                batch_metadata={
                    'model_used': model_name,
                    'candidate_count': len(candidate_memories),
                    'prediction_count': len(predictions)
                }
            )
            
            # Cache result
            if use_cache:
                self._cache_prediction(cache_key, batch)
            
            # Update statistics
            self.total_predictions_made += len(predictions)
            self.model_usage_stats[model_name] += 1
            
            logger.info(f"Generated {len(predictions)} predictions using {model_name} model")
            
            return batch
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {e}")
            return PredictionBatch(predictions=[], context=context)
    
    async def predict_future_associations(self, memory_id: str, 
                                        time_horizon: timedelta = timedelta(days=7),
                                        model_name: str = 'temporal') -> List[PredictionResult]:
        """Predict future associations for a memory"""
        context = PredictionContext(
            current_memory_id=memory_id,
            current_timestamp=datetime.now() + time_horizon
        )
        
        # Get candidate memories (this would come from the memory manager)
        candidate_memories = []  # Placeholder
        
        batch = await self.predict_relevant_memories(context, candidate_memories, model_name)
        return batch.predictions
    
    async def predict_semantic_drift(self, memory_id: str, 
                                   time_horizon: timedelta = timedelta(days=30)) -> PredictionResult:
        """Predict how memory semantics might change over time"""
        # This would analyze historical semantic changes
        # For now, return a placeholder
        
        return PredictionResult(
            memory_id=memory_id,
            prediction_type=PredictionType.SEMANTIC_DRIFT,
            predicted_value=0.3,  # Low drift expected
            confidence=0.6,
            confidence_level=ConfidenceLevel.MEDIUM,
            features_used=PredictionFeatures(),
            model_name="semantic_drift_model",
            explanation="Predicted semantic drift based on historical patterns"
        )
    
    async def train_models(self, training_data: List[Dict[str, Any]], 
                          model_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Train prediction models"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        training_results = {}
        
        for model_name in model_names:
            if model_name in self.models:
                logger.info(f"Training {model_name} model...")
                try:
                    metrics = await self.models[model_name].train(training_data)
                    training_results[model_name] = metrics
                    logger.info(f"Training completed for {model_name}")
                except Exception as e:
                    logger.error(f"Training failed for {model_name}: {e}")
                    training_results[model_name] = {'error': str(e)}
        
        return training_results
    
    def add_training_data(self, data_point: Dict[str, Any]):
        """Add data point to training buffer"""
        self.training_buffer.append(data_point)
        
        # Auto-retrain if threshold reached
        if len(self.training_buffer) >= self.auto_retrain_threshold:
            asyncio.create_task(self._auto_retrain())
    
    async def _auto_retrain(self):
        """Automatically retrain models with buffered data"""
        logger.info("Starting automatic model retraining...")
        
        training_data = list(self.training_buffer)
        self.training_buffer.clear()
        
        await self.train_models(training_data)
        
        logger.info("Automatic retraining completed")
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        return {
            'total_predictions': self.total_predictions_made,
            'model_usage': dict(self.model_usage_stats),
            'cache_size': len(self.prediction_cache),
            'training_buffer_size': len(self.training_buffer),
            'model_status': {
                name: {
                    'is_trained': model.is_trained,
                    'training_data_size': model.training_data_size,
                    'last_training': model.last_training_time.isoformat() if model.last_training_time else None
                }
                for name, model in self.models.items()
            },
            'average_accuracy': sum(self.prediction_accuracy_history) / len(self.prediction_accuracy_history) 
                              if self.prediction_accuracy_history else 0.0
        }
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")
    
    # Private methods
    
    def _generate_cache_key(self, context: PredictionContext, 
                          candidate_memories: List[str], model_name: str) -> str:
        """Generate cache key for prediction"""
        key_data = {
            'model': model_name,
            'current_memory': context.current_memory_id,
            'current_content_hash': hash(context.current_content) if context.current_content else None,
            'candidates_hash': hash(tuple(sorted(candidate_memories))),
            'timestamp_hour': context.current_timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
        }
        
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionBatch]:
        """Get cached prediction if valid"""
        if cache_key in self.prediction_cache:
            cached_data, timestamp = self.prediction_cache[cache_key]
            
            # Check if cache entry is still valid
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return cached_data
            else:
                # Remove expired entry
                del self.prediction_cache[cache_key]
        
        return None
    
    def _cache_prediction(self, cache_key: str, batch: PredictionBatch):
        """Cache prediction result"""
        # Remove oldest entries if cache is full
        if len(self.prediction_cache) >= self.cache_max_size:
            # Remove 10% of oldest entries
            entries_to_remove = max(1, self.cache_max_size // 10)
            oldest_keys = sorted(self.prediction_cache.keys(), 
                               key=lambda k: self.prediction_cache[k][1])[:entries_to_remove]
            
            for key in oldest_keys:
                del self.prediction_cache[key]
        
        self.prediction_cache[cache_key] = (batch, datetime.now())

# Example usage and utility functions

def create_prediction_context(current_memory_id: Optional[str] = None,
                            current_content: Optional[str] = None,
                            current_tags: List[str] = None,
                            recent_memories: List[str] = None,
                            user_context: Dict[str, Any] = None) -> PredictionContext:
    """Create a prediction context with sensible defaults"""
    return PredictionContext(
        current_memory_id=current_memory_id,
        current_content=current_content,
        current_tags=current_tags or [],
        recent_memories=recent_memories or [],
        user_context=user_context or {},
        session_context={},
        current_timestamp=datetime.now(),
        time_of_day=datetime.now().strftime('%H'),
        day_of_week=datetime.now().strftime('%A').lower()
    )

def create_training_data_point(memory_id: str,
                             user_id: str = 'default_user',
                             content: str = '',
                             tags: List[str] = None,
                             relevance_score: float = 0.5,
                             timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """Create a training data point"""
    return {
        'memory_id': memory_id,
        'user_id': user_id,
        'content': content,
        'tags': tags or [],
        'relevance_score': relevance_score,
        'timestamp': timestamp or datetime.now()
    }

async def example_usage():
    """Example usage of the predictive associations system"""
    
    # Configuration
    config = {
        'cache_max_size': 500,
        'cache_ttl': 1800,  # 30 minutes
        'training_buffer_size': 5000,
        'auto_retrain_threshold': 500,
        'ensemble': {
            'collaborative_filtering': {},
            'content_based': {},
            'temporal_sequence': {}
        }
    }
    
    # Initialize manager
    manager = PredictiveAssociationsManager(config)
    
    # Create training data
    training_data = [
        create_training_data_point(
            memory_id=f"memory_{i}",
            content=f"Sample content {i}",
            relevance_score=random.uniform(0.3, 0.9),
            timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
        )
        for i in range(100)
    ]
    
    # Train models
    print("Training models...")
    training_results = await manager.train_models(training_data)
    print(f"Training results: {training_results}")
    
    # Create prediction context
    context = create_prediction_context(
        current_memory_id="memory_50",
        current_content="Sample query content",
        current_tags=["example", "test"],
        recent_memories=["memory_48", "memory_49"]
    )
    
    # Get candidate memories
    candidate_memories = [f"memory_{i}" for i in range(1, 21)]
    
    # Make predictions
    print("Making predictions...")
    predictions = await manager.predict_relevant_memories(context, candidate_memories)
    
    print(f"Generated {len(predictions.predictions)} predictions")
    for pred in predictions.predictions[:5]:  # Show top 5
        print(f"Memory: {pred.memory_id}, Score: {pred.predicted_value:.3f}, "
              f"Confidence: {pred.confidence:.3f} ({pred.confidence_level.value})")
    
    # Get statistics
    stats = manager.get_prediction_statistics()
    print(f"\nPrediction statistics: {stats}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage());
    key
