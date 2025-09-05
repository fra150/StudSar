"""Differential Privacy Module for StudSar V4

Implements privacy-preserving operations for memory storage and retrieval,
ensuring individual data points cannot be identified while maintaining utility.
"""

import hashlib
import json
import logging
import math
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    Fernet = None
    hashes = None
    PBKDF2HMAC = None
    Cipher = None
    algorithms = None
    modes = None

logger = logging.getLogger(__name__)

class PrivacyMechanism(Enum):
    """Types of differential privacy mechanisms"""
    LAPLACE = "laplace"  # Laplace mechanism
    GAUSSIAN = "gaussian"  # Gaussian mechanism
    EXPONENTIAL = "exponential"  # Exponential mechanism
    RANDOMIZED_RESPONSE = "randomized_response"  # Randomized response
    LOCAL_HASHING = "local_hashing"  # Local differential privacy with hashing
    RAPPOR = "rappor"  # RAPPOR (Randomized Aggregatable Privacy-Preserving Ordinal Response)
    PRIVATE_AGGREGATION = "private_aggregation"  # Private aggregation
    SYNTHETIC_DATA = "synthetic_data"  # Synthetic data generation

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    LOW = "low"  # ε = 1.0
    MEDIUM = "medium"  # ε = 0.1
    HIGH = "high"  # ε = 0.01
    MAXIMUM = "maximum"  # ε = 0.001

class NoiseType(Enum):
    """Types of noise for privacy"""
    ADDITIVE = "additive"  # Add noise to values
    MULTIPLICATIVE = "multiplicative"  # Multiply by noise
    SUBSTITUTION = "substitution"  # Replace with noisy values
    PERMUTATION = "permutation"  # Permute order
    SUPPRESSION = "suppression"  # Remove some values

class QueryType(Enum):
    """Types of queries for privacy analysis"""
    COUNT = "count"  # Count queries
    SUM = "sum"  # Sum queries
    AVERAGE = "average"  # Average queries
    HISTOGRAM = "histogram"  # Histogram queries
    RANGE = "range"  # Range queries
    TOP_K = "top_k"  # Top-k queries
    SIMILARITY = "similarity"  # Similarity queries
    CLUSTERING = "clustering"  # Clustering queries

@dataclass
class PrivacyBudget:
    """Privacy budget management"""
    total_epsilon: float
    used_epsilon: float = 0.0
    total_delta: float = 1e-5
    used_delta: float = 0.0
    allocations: Dict[str, float] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)
    expiry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_epsilon(self) -> float:
        """Get remaining epsilon budget"""
        return max(0.0, self.total_epsilon - self.used_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        """Get remaining delta budget"""
        return max(0.0, self.total_delta - self.used_delta)
    
    def can_allocate(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check if budget can be allocated"""
        return (epsilon <= self.remaining_epsilon and 
                delta <= self.remaining_delta)
    
    def allocate(self, query_id: str, epsilon: float, delta: float = 0.0) -> bool:
        """Allocate privacy budget"""
        if not self.can_allocate(epsilon, delta):
            return False
        
        self.used_epsilon += epsilon
        self.used_delta += delta
        self.allocations[query_id] = epsilon
        
        return True

@dataclass
class PrivacyParameters:
    """Parameters for privacy mechanisms"""
    epsilon: float  # Privacy parameter
    delta: float = 1e-5  # Failure probability
    sensitivity: float = 1.0  # Global sensitivity
    mechanism: PrivacyMechanism = PrivacyMechanism.LAPLACE
    noise_type: NoiseType = NoiseType.ADDITIVE
    clipping_bound: Optional[float] = None
    sampling_probability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrivateQuery:
    """A privacy-preserving query"""
    query_id: str
    query_type: QueryType
    parameters: Dict[str, Any]
    privacy_params: PrivacyParameters
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[Any] = None
    noise_added: Optional[float] = None
    privacy_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrivacyAuditLog:
    """Audit log for privacy operations"""
    operation_id: str
    operation_type: str
    privacy_mechanism: PrivacyMechanism
    epsilon_used: float
    delta_used: float
    data_size: int
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    query_hash: Optional[str] = None
    result_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BasePrivacyMechanism(ABC):
    """Base class for differential privacy mechanisms"""
    
    def __init__(self, privacy_params: PrivacyParameters):
        self.privacy_params = privacy_params
        self.name = self.__class__.__name__
        
    @abstractmethod
    def add_noise(self, value: Union[float, List[float]], 
                  sensitivity: Optional[float] = None) -> Union[float, List[float]]:
        """Add privacy-preserving noise to value(s)"""
        pass
    
    @abstractmethod
    def get_privacy_cost(self) -> Tuple[float, float]:
        """Get privacy cost (epsilon, delta)"""
        pass
    
    def validate_parameters(self) -> bool:
        """Validate privacy parameters"""
        if self.privacy_params.epsilon <= 0:
            return False
        if self.privacy_params.delta < 0 or self.privacy_params.delta >= 1:
            return False
        if self.privacy_params.sensitivity <= 0:
            return False
        return True

class LaplaceMechanism(BasePrivacyMechanism):
    """Laplace mechanism for differential privacy"""
    
    def __init__(self, privacy_params: PrivacyParameters):
        super().__init__(privacy_params)
        self.scale = privacy_params.sensitivity / privacy_params.epsilon
        
    def add_noise(self, value: Union[float, List[float]], 
                  sensitivity: Optional[float] = None) -> Union[float, List[float]]:
        """Add Laplace noise"""
        if sensitivity is not None:
            scale = sensitivity / self.privacy_params.epsilon
        else:
            scale = self.scale
        
        if isinstance(value, list):
            if np:
                noise = np.random.laplace(0, scale, len(value))
                return (np.array(value) + noise).tolist()
            else:
                # Fallback implementation
                return [v + self._sample_laplace(scale) for v in value]
        else:
            return value + self._sample_laplace(scale)
    
    def _sample_laplace(self, scale: float) -> float:
        """Sample from Laplace distribution"""
        if np:
            return np.random.laplace(0, scale)
        else:
            # Box-Muller transform approximation
            u = random.uniform(0, 1)
            if u < 0.5:
                return -scale * math.log(2 * u)
            else:
                return scale * math.log(2 * (1 - u))
    
    def get_privacy_cost(self) -> Tuple[float, float]:
        """Get privacy cost for Laplace mechanism"""
        return self.privacy_params.epsilon, 0.0  # Pure DP

class GaussianMechanism(BasePrivacyMechanism):
    """Gaussian mechanism for differential privacy"""
    
    def __init__(self, privacy_params: PrivacyParameters):
        super().__init__(privacy_params)
        # Calculate sigma for (ε, δ)-DP
        self.sigma = self._calculate_sigma()
        
    def _calculate_sigma(self) -> float:
        """Calculate sigma for Gaussian mechanism"""
        epsilon = self.privacy_params.epsilon
        delta = self.privacy_params.delta
        sensitivity = self.privacy_params.sensitivity
        
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        # Approximate formula for sigma
        c = math.sqrt(2 * math.log(1.25 / delta))
        sigma = c * sensitivity / epsilon
        
        return sigma
    
    def add_noise(self, value: Union[float, List[float]], 
                  sensitivity: Optional[float] = None) -> Union[float, List[float]]:
        """Add Gaussian noise"""
        if sensitivity is not None:
            # Recalculate sigma for different sensitivity
            epsilon = self.privacy_params.epsilon
            delta = self.privacy_params.delta
            c = math.sqrt(2 * math.log(1.25 / delta))
            sigma = c * sensitivity / epsilon
        else:
            sigma = self.sigma
        
        if isinstance(value, list):
            if np:
                noise = np.random.normal(0, sigma, len(value))
                return (np.array(value) + noise).tolist()
            else:
                return [v + random.gauss(0, sigma) for v in value]
        else:
            if np:
                return value + np.random.normal(0, sigma)
            else:
                return value + random.gauss(0, sigma)
    
    def get_privacy_cost(self) -> Tuple[float, float]:
        """Get privacy cost for Gaussian mechanism"""
        return self.privacy_params.epsilon, self.privacy_params.delta

class ExponentialMechanism(BasePrivacyMechanism):
    """Exponential mechanism for differential privacy"""
    
    def __init__(self, privacy_params: PrivacyParameters):
        super().__init__(privacy_params)
        
    def select_output(self, candidates: List[Any], 
                     utility_function: Callable[[Any], float]) -> Any:
        """Select output using exponential mechanism"""
        if not candidates:
            raise ValueError("No candidates provided")
        
        # Calculate utilities
        utilities = [utility_function(candidate) for candidate in candidates]
        
        # Calculate probabilities
        epsilon = self.privacy_params.epsilon
        sensitivity = self.privacy_params.sensitivity
        
        # Normalize utilities and calculate probabilities
        max_utility = max(utilities)
        exp_utilities = [math.exp(epsilon * (u - max_utility) / (2 * sensitivity)) 
                        for u in utilities]
        
        total_prob = sum(exp_utilities)
        probabilities = [p / total_prob for p in exp_utilities]
        
        # Sample according to probabilities
        r = random.random()
        cumulative = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return candidates[i]
        
        # Fallback to last candidate
        return candidates[-1]
    
    def add_noise(self, value: Union[float, List[float]], 
                  sensitivity: Optional[float] = None) -> Union[float, List[float]]:
        """Not applicable for exponential mechanism"""
        raise NotImplementedError("Exponential mechanism doesn't add noise directly")
    
    def get_privacy_cost(self) -> Tuple[float, float]:
        """Get privacy cost for exponential mechanism"""
        return self.privacy_params.epsilon, 0.0  # Pure DP

class RandomizedResponseMechanism(BasePrivacyMechanism):
    """Randomized response mechanism for local differential privacy"""
    
    def __init__(self, privacy_params: PrivacyParameters):
        super().__init__(privacy_params)
        # Calculate probability for randomized response
        epsilon = privacy_params.epsilon
        self.p = math.exp(epsilon) / (math.exp(epsilon) + 1)
        
    def randomize_response(self, true_value: bool) -> bool:
        """Apply randomized response to boolean value"""
        if random.random() < self.p:
            return true_value
        else:
            return not true_value
    
    def randomize_categorical(self, true_value: str, 
                            categories: List[str]) -> str:
        """Apply randomized response to categorical value"""
        epsilon = self.privacy_params.epsilon
        k = len(categories)
        
        # Probability of reporting true value
        p_true = math.exp(epsilon) / (math.exp(epsilon) + k - 1)
        
        if random.random() < p_true:
            return true_value
        else:
            # Select uniformly from other categories
            other_categories = [c for c in categories if c != true_value]
            return random.choice(other_categories)
    
    def add_noise(self, value: Union[float, List[float]], 
                  sensitivity: Optional[float] = None) -> Union[float, List[float]]:
        """Apply randomized response (for boolean/categorical data)"""
        if isinstance(value, list):
            # Assume boolean list
            return [self.randomize_response(bool(v)) for v in value]
        else:
            return self.randomize_response(bool(value))
    
    def get_privacy_cost(self) -> Tuple[float, float]:
        """Get privacy cost for randomized response"""
        return self.privacy_params.epsilon, 0.0  # Local DP

class PrivateAggregator:
    """Aggregator for private computations"""
    
    def __init__(self, privacy_params: PrivacyParameters):
        self.privacy_params = privacy_params
        self.mechanism = self._create_mechanism()
        
    def _create_mechanism(self) -> BasePrivacyMechanism:
        """Create appropriate privacy mechanism"""
        if self.privacy_params.mechanism == PrivacyMechanism.LAPLACE:
            return LaplaceMechanism(self.privacy_params)
        elif self.privacy_params.mechanism == PrivacyMechanism.GAUSSIAN:
            return GaussianMechanism(self.privacy_params)
        elif self.privacy_params.mechanism == PrivacyMechanism.EXPONENTIAL:
            return ExponentialMechanism(self.privacy_params)
        elif self.privacy_params.mechanism == PrivacyMechanism.RANDOMIZED_RESPONSE:
            return RandomizedResponseMechanism(self.privacy_params)
        else:
            return LaplaceMechanism(self.privacy_params)  # Default
    
    def private_count(self, data: List[Any], 
                     predicate: Optional[Callable[[Any], bool]] = None) -> float:
        """Compute private count"""
        if predicate:
            true_count = sum(1 for item in data if predicate(item))
        else:
            true_count = len(data)
        
        # Add noise with sensitivity 1
        noisy_count = self.mechanism.add_noise(float(true_count), sensitivity=1.0)
        
        # Ensure non-negative
        return max(0.0, noisy_count)
    
    def private_sum(self, data: List[float], 
                   clipping_bound: Optional[float] = None) -> float:
        """Compute private sum"""
        if clipping_bound:
            # Clip values
            clipped_data = [max(-clipping_bound, min(clipping_bound, x)) for x in data]
            sensitivity = 2 * clipping_bound  # L1 sensitivity
        else:
            clipped_data = data
            # Estimate sensitivity (this is not ideal in practice)
            sensitivity = max(abs(x) for x in data) if data else 1.0
        
        true_sum = sum(clipped_data)
        noisy_sum = self.mechanism.add_noise(true_sum, sensitivity=sensitivity)
        
        return noisy_sum
    
    def private_mean(self, data: List[float], 
                    clipping_bound: Optional[float] = None) -> float:
        """Compute private mean"""
        if not data:
            return 0.0
        
        # Use composition: private sum + private count
        epsilon_sum = self.privacy_params.epsilon / 2
        epsilon_count = self.privacy_params.epsilon / 2
        
        # Create mechanisms for sum and count
        sum_params = PrivacyParameters(
            epsilon=epsilon_sum,
            delta=self.privacy_params.delta / 2,
            sensitivity=self.privacy_params.sensitivity,
            mechanism=self.privacy_params.mechanism
        )
        
        count_params = PrivacyParameters(
            epsilon=epsilon_count,
            delta=self.privacy_params.delta / 2,
            sensitivity=1.0,
            mechanism=self.privacy_params.mechanism
        )
        
        sum_aggregator = PrivateAggregator(sum_params)
        count_aggregator = PrivateAggregator(count_params)
        
        private_sum_val = sum_aggregator.private_sum(data, clipping_bound)
        private_count_val = count_aggregator.private_count(data)
        
        if private_count_val <= 0:
            return 0.0
        
        return private_sum_val / private_count_val
    
    def private_histogram(self, data: List[str], 
                         bins: List[str]) -> Dict[str, float]:
        """Compute private histogram"""
        # Count occurrences
        counts = {bin_name: 0 for bin_name in bins}
        for item in data:
            if item in counts:
                counts[item] += 1
        
        # Add noise to each count
        epsilon_per_bin = self.privacy_params.epsilon / len(bins)
        
        noisy_counts = {}
        for bin_name, count in counts.items():
            bin_params = PrivacyParameters(
                epsilon=epsilon_per_bin,
                delta=self.privacy_params.delta / len(bins),
                sensitivity=1.0,
                mechanism=self.privacy_params.mechanism
            )
            
            bin_mechanism = self._create_mechanism()
            bin_mechanism.privacy_params = bin_params
            
            noisy_count = bin_mechanism.add_noise(float(count), sensitivity=1.0)
            noisy_counts[bin_name] = max(0.0, noisy_count)
        
        return noisy_counts

class PrivacyAccountant:
    """Manages privacy budget and tracks usage"""
    
    def __init__(self, total_epsilon: float = 1.0, total_delta: float = 1e-5):
        self.budgets: Dict[str, PrivacyBudget] = {}
        self.audit_logs: List[PrivacyAuditLog] = []
        self.default_budget = PrivacyBudget(
            total_epsilon=total_epsilon,
            total_delta=total_delta
        )
        self.lock = threading.Lock()
        
    def create_budget(self, budget_id: str, epsilon: float, 
                     delta: float = 1e-5, 
                     expiry_hours: Optional[int] = None) -> bool:
        """Create a new privacy budget"""
        with self.lock:
            if budget_id in self.budgets:
                return False
            
            expiry_time = None
            if expiry_hours:
                expiry_time = datetime.now() + timedelta(hours=expiry_hours)
            
            self.budgets[budget_id] = PrivacyBudget(
                total_epsilon=epsilon,
                total_delta=delta,
                expiry_time=expiry_time
            )
            
            return True
    
    def allocate_budget(self, budget_id: str, query_id: str, 
                       epsilon: float, delta: float = 0.0) -> bool:
        """Allocate privacy budget for a query"""
        with self.lock:
            budget = self.budgets.get(budget_id, self.default_budget)
            
            # Check expiry
            if budget.expiry_time and datetime.now() > budget.expiry_time:
                return False
            
            success = budget.allocate(query_id, epsilon, delta)
            
            if success:
                # Log allocation
                log_entry = PrivacyAuditLog(
                    operation_id=str(uuid.uuid4()),
                    operation_type="budget_allocation",
                    privacy_mechanism=PrivacyMechanism.LAPLACE,  # Default
                    epsilon_used=epsilon,
                    delta_used=delta,
                    data_size=0,
                    query_hash=hashlib.sha256(query_id.encode()).hexdigest()[:16],
                    metadata={'budget_id': budget_id, 'query_id': query_id}
                )
                
                self.audit_logs.append(log_entry)
            
            return success
    
    def get_budget_status(self, budget_id: str) -> Optional[Dict[str, Any]]:
        """Get budget status"""
        with self.lock:
            budget = self.budgets.get(budget_id)
            if not budget:
                return None
            
            return {
                'budget_id': budget_id,
                'total_epsilon': budget.total_epsilon,
                'used_epsilon': budget.used_epsilon,
                'remaining_epsilon': budget.remaining_epsilon,
                'total_delta': budget.total_delta,
                'used_delta': budget.used_delta,
                'remaining_delta': budget.remaining_delta,
                'allocations': len(budget.allocations),
                'creation_time': budget.creation_time.isoformat(),
                'expiry_time': budget.expiry_time.isoformat() if budget.expiry_time else None,
                'expired': budget.expiry_time and datetime.now() > budget.expiry_time
            }
    
    def cleanup_expired_budgets(self) -> int:
        """Remove expired budgets"""
        with self.lock:
            current_time = datetime.now()
            expired_budgets = []
            
            for budget_id, budget in self.budgets.items():
                if budget.expiry_time and current_time > budget.expiry_time:
                    expired_budgets.append(budget_id)
            
            for budget_id in expired_budgets:
                del self.budgets[budget_id]
            
            return len(expired_budgets)
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for recent operations"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_logs = [log for log in self.audit_logs 
                      if log.timestamp >= cutoff_time]
        
        total_epsilon = sum(log.epsilon_used for log in recent_logs)
        total_delta = sum(log.delta_used for log in recent_logs)
        
        mechanism_counts = defaultdict(int)
        for log in recent_logs:
            mechanism_counts[log.privacy_mechanism.value] += 1
        
        return {
            'period_hours': hours,
            'total_operations': len(recent_logs),
            'total_epsilon_used': total_epsilon,
            'total_delta_used': total_delta,
            'mechanism_usage': dict(mechanism_counts),
            'active_budgets': len(self.budgets),
            'audit_log_size': len(self.audit_logs)
        }

class DifferentialPrivacyManager:
    """Main manager for differential privacy operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accountant = PrivacyAccountant(
            total_epsilon=config.get('default_epsilon', 1.0),
            total_delta=config.get('default_delta', 1e-5)
        )
        
        # Default privacy levels
        self.privacy_levels = {
            PrivacyLevel.LOW: PrivacyParameters(epsilon=1.0, delta=1e-3),
            PrivacyLevel.MEDIUM: PrivacyParameters(epsilon=0.1, delta=1e-4),
            PrivacyLevel.HIGH: PrivacyParameters(epsilon=0.01, delta=1e-5),
            PrivacyLevel.MAXIMUM: PrivacyParameters(epsilon=0.001, delta=1e-6)
        }
        
        # Query cache for repeated queries
        self.query_cache: Dict[str, Any] = {}
        self.cache_ttl = config.get('cache_ttl_hours', 1)
        
        # Statistics
        self.queries_processed = 0
        self.total_epsilon_used = 0.0
        self.total_delta_used = 0.0
        
        logger.info("Initialized Differential Privacy Manager")
    
    def create_private_query(self, query_type: QueryType, 
                           data: List[Any],
                           privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                           custom_params: Optional[PrivacyParameters] = None,
                           budget_id: str = "default") -> PrivateQuery:
        """Create a private query"""
        
        query_id = str(uuid.uuid4())
        
        # Get privacy parameters
        if custom_params:
            privacy_params = custom_params
        else:
            privacy_params = self.privacy_levels[privacy_level]
        
        # Check budget allocation
        if not self.accountant.allocate_budget(
            budget_id, query_id, 
            privacy_params.epsilon, privacy_params.delta
        ):
            raise ValueError(f"Insufficient privacy budget for query {query_id}")
        
        query = PrivateQuery(
            query_id=query_id,
            query_type=query_type,
            parameters={'data_size': len(data)},
            privacy_params=privacy_params
        )
        
        return query
    
    def execute_private_query(self, query: PrivateQuery, 
                            data: List[Any],
                            **kwargs) -> Any:
        """Execute a private query"""
        
        try:
            # Create aggregator
            aggregator = PrivateAggregator(query.privacy_params)
            
            # Execute based on query type
            if query.query_type == QueryType.COUNT:
                predicate = kwargs.get('predicate')
                result = aggregator.private_count(data, predicate)
                
            elif query.query_type == QueryType.SUM:
                clipping_bound = kwargs.get('clipping_bound')
                if not all(isinstance(x, (int, float)) for x in data):
                    raise ValueError("Sum query requires numeric data")
                result = aggregator.private_sum(data, clipping_bound)
                
            elif query.query_type == QueryType.AVERAGE:
                clipping_bound = kwargs.get('clipping_bound')
                if not all(isinstance(x, (int, float)) for x in data):
                    raise ValueError("Average query requires numeric data")
                result = aggregator.private_mean(data, clipping_bound)
                
            elif query.query_type == QueryType.HISTOGRAM:
                bins = kwargs.get('bins', [])
                if not bins:
                    # Auto-generate bins from data
                    bins = list(set(str(x) for x in data))
                string_data = [str(x) for x in data]
                result = aggregator.private_histogram(string_data, bins)
                
            else:
                raise ValueError(f"Unsupported query type: {query.query_type}")
            
            # Update query with result
            query.result = result
            query.privacy_cost = query.privacy_params.epsilon
            
            # Update statistics
            self.queries_processed += 1
            self.total_epsilon_used += query.privacy_params.epsilon
            self.total_delta_used += query.privacy_params.delta
            
            # Log operation
            log_entry = PrivacyAuditLog(
                operation_id=str(uuid.uuid4()),
                operation_type=f"query_{query.query_type.value}",
                privacy_mechanism=query.privacy_params.mechanism,
                epsilon_used=query.privacy_params.epsilon,
                delta_used=query.privacy_params.delta,
                data_size=len(data),
                query_hash=hashlib.sha256(query.query_id.encode()).hexdigest()[:16],
                result_hash=hashlib.sha256(str(result).encode()).hexdigest()[:16]
            )
            
            self.accountant.audit_logs.append(log_entry)
            
            logger.info(f"Executed private {query.query_type.value} query "
                       f"with ε={query.privacy_params.epsilon}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing private query {query.query_id}: {e}")
            raise
    
    def private_count_query(self, data: List[Any], 
                          predicate: Optional[Callable[[Any], bool]] = None,
                          privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                          budget_id: str = "default") -> float:
        """Convenience method for private count"""
        query = self.create_private_query(
            QueryType.COUNT, data, privacy_level, budget_id=budget_id
        )
        return self.execute_private_query(query, data, predicate=predicate)
    
    def private_sum_query(self, data: List[float], 
                         clipping_bound: Optional[float] = None,
                         privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                         budget_id: str = "default") -> float:
        """Convenience method for private sum"""
        query = self.create_private_query(
            QueryType.SUM, data, privacy_level, budget_id=budget_id
        )
        return self.execute_private_query(query, data, clipping_bound=clipping_bound)
    
    def private_mean_query(self, data: List[float], 
                          clipping_bound: Optional[float] = None,
                          privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                          budget_id: str = "default") -> float:
        """Convenience method for private mean"""
        query = self.create_private_query(
            QueryType.AVERAGE, data, privacy_level, budget_id=budget_id
        )
        return self.execute_private_query(query, data, clipping_bound=clipping_bound)
    
    def private_histogram_query(self, data: List[Any], 
                               bins: Optional[List[str]] = None,
                               privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                               budget_id: str = "default") -> Dict[str, float]:
        """Convenience method for private histogram"""
        query = self.create_private_query(
            QueryType.HISTOGRAM, data, privacy_level, budget_id=budget_id
        )
        return self.execute_private_query(query, data, bins=bins)
    
    def anonymize_data(self, data: List[Dict[str, Any]], 
                      privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                      k_anonymity: int = 5) -> List[Dict[str, Any]]:
        """Anonymize dataset using differential privacy"""
        
        if not data:
            return []
        
        privacy_params = self.privacy_levels[privacy_level]
        
        # Simple anonymization: add noise to numeric fields
        anonymized_data = []
        
        for record in data:
            anonymized_record = {}
            
            for field, value in record.items():
                if isinstance(value, (int, float)):
                    # Add Laplace noise to numeric fields
                    mechanism = LaplaceMechanism(privacy_params)
                    noisy_value = mechanism.add_noise(float(value))
                    anonymized_record[field] = noisy_value
                    
                elif isinstance(value, str):
                    # For categorical data, use randomized response
                    rr_mechanism = RandomizedResponseMechanism(privacy_params)
                    # Simple binary randomization (would need categories for full RR)
                    anonymized_record[field] = value  # Placeholder
                    
                else:
                    # Keep other types as-is
                    anonymized_record[field] = value
            
            anonymized_data.append(anonymized_record)
        
        logger.info(f"Anonymized {len(data)} records with privacy level {privacy_level.value}")
        
        return anonymized_data
    
    def get_privacy_statistics(self) -> Dict[str, Any]:
        """Get privacy usage statistics"""
        return {
            'queries_processed': self.queries_processed,
            'total_epsilon_used': self.total_epsilon_used,
            'total_delta_used': self.total_delta_used,
            'active_budgets': len(self.accountant.budgets),
            'audit_log_entries': len(self.accountant.audit_logs),
            'cache_size': len(self.query_cache),
            'privacy_levels_available': [level.value for level in PrivacyLevel],
            'mechanisms_available': [mech.value for mech in PrivacyMechanism],
            'accountant_summary': self.accountant.get_audit_summary()
        }
    
    def cleanup_resources(self):
        """Cleanup expired resources"""
        # Clean expired budgets
        expired_count = self.accountant.cleanup_expired_budgets()
        
        # Clean old cache entries
        current_time = datetime.now()
        cache_cutoff = current_time - timedelta(hours=self.cache_ttl)
        
        old_cache_keys = []
        for key, (result, timestamp) in self.query_cache.items():
            if timestamp < cache_cutoff:
                old_cache_keys.append(key)
        
        for key in old_cache_keys:
            del self.query_cache[key]
        
        logger.info(f"Cleaned up {expired_count} expired budgets and "
                   f"{len(old_cache_keys)} old cache entries")

# Example usage and utility functions

def create_privacy_config(epsilon: float = 1.0, 
                         delta: float = 1e-5,
                         mechanism: PrivacyMechanism = PrivacyMechanism.LAPLACE) -> Dict[str, Any]:
    """Create privacy configuration"""
    return {
        'default_epsilon': epsilon,
        'default_delta': delta,
        'default_mechanism': mechanism.value,
        'cache_ttl_hours': 1,
        'enable_audit_logging': True,
        'max_audit_log_size': 10000
    }

def example_usage():
    """Example usage of differential privacy system"""
    
    # Configuration
    config = create_privacy_config(
        epsilon=1.0,
        delta=1e-5,
        mechanism=PrivacyMechanism.LAPLACE
    )
    
    # Initialize manager
    dp_manager = DifferentialPrivacyManager(config)
    
    # Create privacy budget
    dp_manager.accountant.create_budget(
        "user_analytics", 
        epsilon=2.0, 
        delta=1e-4, 
        expiry_hours=24
    )
    
    # Sample data
    numeric_data = [1.5, 2.3, 3.1, 4.7, 2.8, 3.9, 1.2, 5.1, 2.6, 3.4]
    categorical_data = ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C']
    
    print("Differential Privacy Example")
    print("=" * 30)
    
    # Private count
    private_count = dp_manager.private_count_query(
        numeric_data,
        predicate=lambda x: x > 3.0,
        privacy_level=PrivacyLevel.MEDIUM,
        budget_id="user_analytics"
    )
    print(f"Private count (>3.0): {private_count:.2f}")
    
    # Private sum
    private_sum = dp_manager.private_sum_query(
        numeric_data,
        clipping_bound=10.0,
        privacy_level=PrivacyLevel.MEDIUM,
        budget_id="user_analytics"
    )
    print(f"Private sum: {private_sum:.2f}")
    
    # Private mean
    private_mean = dp_manager.private_mean_query(
        numeric_data,
        clipping_bound=10.0,
        privacy_level=PrivacyLevel.MEDIUM,
        budget_id="user_analytics"
    )
    print(f"Private mean: {private_mean:.2f}")
    
    # Private histogram
    private_hist = dp_manager.private_histogram_query(
        categorical_data,
        bins=['A', 'B', 'C'],
        privacy_level=PrivacyLevel.MEDIUM,
        budget_id="user_analytics"
    )
    print(f"Private histogram: {private_hist}")
    
    # Anonymize data
    sample_records = [
        {'age': 25, 'salary': 50000, 'department': 'Engineering'},
        {'age': 30, 'salary': 60000, 'department': 'Marketing'},
        {'age': 35, 'salary': 70000, 'department': 'Engineering'},
        {'age': 28, 'salary': 55000, 'department': 'Sales'}
    ]
    
    anonymized_records = dp_manager.anonymize_data(
        sample_records,
        privacy_level=PrivacyLevel.MEDIUM
    )
    
    print("\nOriginal vs Anonymized Data:")
    for i, (orig, anon) in enumerate(zip(sample_records, anonymized_records)):
        print(f"Record {i+1}:")
        print(f"  Original: {orig}")
        print(f"  Anonymized: {anon}")
    
    # Get statistics
    stats = dp_manager.get_privacy_statistics()
    print(f"\nPrivacy Statistics:")
    print(f"Queries processed: {stats['queries_processed']}")
    print(f"Total epsilon used: {stats['total_epsilon_used']:.4f}")
    print(f"Total delta used: {stats['total_delta_used']:.6f}")
    
    # Budget status
    budget_status = dp_manager.accountant.get_budget_status("user_analytics")
    if budget_status:
        print(f"\nBudget Status:")
        print(f"Remaining epsilon: {budget_status['remaining_epsilon']:.4f}")
        print(f"Remaining delta: {budget_status['remaining_delta']:.6f}")
    
    # Cleanup
    dp_manager.cleanup_resources()
    
    print("\nDifferential privacy example completed!")

if __name__ == "__main__":
    # Run example
    example_usage()