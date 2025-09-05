"""Federated Learning Module for StudSar V4

Implements collaborative learning between multiple StudSar instances,
allowing knowledge sharing while preserving privacy and autonomy.
"""

import asyncio
import hashlib
import json
import logging
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
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    Fernet = None
    hashes = None
    PBKDF2HMAC = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import websockets
except ImportError:
    websockets = None

logger = logging.getLogger(__name__)

class NodeRole(Enum):
    """Roles in federated network"""
    COORDINATOR = "coordinator"  # Central coordination node
    PARTICIPANT = "participant"  # Regular participating node
    OBSERVER = "observer"  # Read-only observer
    VALIDATOR = "validator"  # Validates updates
    AGGREGATOR = "aggregator"  # Aggregates model updates

class LearningStrategy(Enum):
    """Federated learning strategies"""
    FEDERATED_AVERAGING = "federated_averaging"  # FedAvg
    FEDERATED_SGD = "federated_sgd"  # FedSGD
    PERSONALIZED = "personalized"  # Personalized FL
    HIERARCHICAL = "hierarchical"  # Hierarchical FL
    ASYNCHRONOUS = "asynchronous"  # Async updates
    DIFFERENTIAL_PRIVATE = "differential_private"  # DP-FL

class MessageType(Enum):
    """Types of federated messages"""
    MODEL_UPDATE = "model_update"
    GRADIENT_UPDATE = "gradient_update"
    AGGREGATION_RESULT = "aggregation_result"
    COORDINATION = "coordination"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    CONSENSUS = "consensus"
    KNOWLEDGE_SHARE = "knowledge_share"
    PRIVACY_BUDGET = "privacy_budget"

class NodeStatus(Enum):
    """Node status in federation"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SYNCHRONIZING = "synchronizing"
    UPDATING = "updating"
    VALIDATING = "validating"
    DISCONNECTED = "disconnected"
    BANNED = "banned"

class AggregationMethod(Enum):
    """Methods for aggregating updates"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    BYZANTINE_ROBUST = "byzantine_robust"
    SECURE_AGGREGATION = "secure_aggregation"

@dataclass
class NodeInfo:
    """Information about a federated node"""
    node_id: str
    role: NodeRole
    status: NodeStatus
    endpoint: str
    public_key: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    last_seen: datetime = field(default_factory=datetime.now)
    trust_score: float = 1.0
    contribution_score: float = 0.0
    data_size: int = 0
    model_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelUpdate:
    """Model update from a node"""
    update_id: str
    node_id: str
    model_version: str
    parameters: Dict[str, Any]
    gradients: Optional[Dict[str, Any]] = None
    data_size: int = 0
    training_loss: float = 0.0
    validation_accuracy: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    signature: Optional[str] = None
    privacy_budget_used: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FederatedMessage:
    """Message in federated communication"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: int = 3600  # Time to live in seconds
    priority: int = 1  # 1=low, 5=high
    encrypted: bool = False
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregationResult:
    """Result of model aggregation"""
    aggregation_id: str
    round_number: int
    participating_nodes: List[str]
    aggregated_parameters: Dict[str, Any]
    aggregation_method: AggregationMethod
    convergence_metric: float
    quality_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FederationConfig:
    """Configuration for federated learning"""
    federation_id: str
    learning_strategy: LearningStrategy
    aggregation_method: AggregationMethod
    min_participants: int = 2
    max_participants: int = 100
    round_duration: int = 300  # seconds
    convergence_threshold: float = 0.001
    privacy_budget: float = 1.0
    trust_threshold: float = 0.5
    max_rounds: int = 1000
    enable_encryption: bool = True
    enable_validation: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseCommunicationProtocol(ABC):
    """Base class for communication protocols"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def send_message(self, message: FederatedMessage, target: str) -> bool:
        """Send message to target node"""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: FederatedMessage) -> List[str]:
        """Broadcast message to all nodes"""
        pass
    
    @abstractmethod
    async def receive_messages(self) -> List[FederatedMessage]:
        """Receive pending messages"""
        pass
    
    @abstractmethod
    async def discover_nodes(self) -> List[NodeInfo]:
        """Discover available nodes"""
        pass

class HTTPCommunicationProtocol(BaseCommunicationProtocol):
    """HTTP-based communication protocol"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session = None
        self.server_port = config.get('port', 8080)
        self.timeout = config.get('timeout', 30)
        
    async def initialize(self):
        """Initialize HTTP session"""
        if aiohttp:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: FederatedMessage, target: str) -> bool:
        """Send HTTP message"""
        if not self.session:
            await self.initialize()
        
        try:
            url = f"http://{target}/federated/message"
            data = {
                'message_id': message.message_id,
                'sender_id': message.sender_id,
                'message_type': message.message_type.value,
                'payload': message.payload,
                'timestamp': message.timestamp.isoformat(),
                'signature': message.signature
            }
            
            async with self.session.post(url, json=data) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to send message to {target}: {e}")
            return False
    
    async def broadcast_message(self, message: FederatedMessage) -> List[str]:
        """Broadcast to known nodes"""
        # This would need a registry of known nodes
        # For now, return empty list
        return []
    
    async def receive_messages(self) -> List[FederatedMessage]:
        """Receive messages (would be implemented with a server)"""
        # This would be implemented with an HTTP server
        # For now, return empty list
        return []
    
    async def discover_nodes(self) -> List[NodeInfo]:
        """Discover nodes via HTTP"""
        # This would implement node discovery
        # For now, return empty list
        return []

class WebSocketCommunicationProtocol(BaseCommunicationProtocol):
    """WebSocket-based communication protocol"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.connections = {}
        self.server_port = config.get('port', 8081)
        
    async def send_message(self, message: FederatedMessage, target: str) -> bool:
        """Send WebSocket message"""
        if not websockets:
            logger.warning("WebSockets not available")
            return False
        
        try:
            if target not in self.connections:
                # Establish connection
                uri = f"ws://{target}:{self.server_port}/federated"
                self.connections[target] = await websockets.connect(uri)
            
            websocket = self.connections[target]
            
            data = json.dumps({
                'message_id': message.message_id,
                'sender_id': message.sender_id,
                'message_type': message.message_type.value,
                'payload': message.payload,
                'timestamp': message.timestamp.isoformat()
            })
            
            await websocket.send(data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket message to {target}: {e}")
            if target in self.connections:
                del self.connections[target]
            return False
    
    async def broadcast_message(self, message: FederatedMessage) -> List[str]:
        """Broadcast via WebSocket"""
        successful_sends = []
        
        for target in list(self.connections.keys()):
            if await self.send_message(message, target):
                successful_sends.append(target)
        
        return successful_sends
    
    async def receive_messages(self) -> List[FederatedMessage]:
        """Receive WebSocket messages"""
        messages = []
        
        for target, websocket in list(self.connections.items()):
            try:
                # Non-blocking receive
                data = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                message_data = json.loads(data)
                
                message = FederatedMessage(
                    message_id=message_data['message_id'],
                    sender_id=message_data['sender_id'],
                    receiver_id=None,
                    message_type=MessageType(message_data['message_type']),
                    payload=message_data['payload'],
                    timestamp=datetime.fromisoformat(message_data['timestamp'])
                )
                
                messages.append(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error receiving from {target}: {e}")
                del self.connections[target]
        
        return messages
    
    async def discover_nodes(self) -> List[NodeInfo]:
        """Discover nodes via WebSocket"""
        # Implementation would depend on discovery mechanism
        return []

class BaseAggregator(ABC):
    """Base class for model aggregators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def aggregate_updates(self, updates: List[ModelUpdate]) -> AggregationResult:
        """Aggregate model updates"""
        pass
    
    @abstractmethod
    def validate_update(self, update: ModelUpdate) -> bool:
        """Validate a model update"""
        pass

class FederatedAveragingAggregator(BaseAggregator):
    """Federated Averaging (FedAvg) aggregator"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.round_number = 0
        
    def aggregate_updates(self, updates: List[ModelUpdate]) -> AggregationResult:
        """Aggregate using federated averaging"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        self.round_number += 1
        
        # Calculate weights based on data size
        total_data_size = sum(update.data_size for update in updates)
        weights = [update.data_size / total_data_size for update in updates]
        
        # Aggregate parameters
        aggregated_params = {}
        
        # Get all parameter keys from first update
        if updates[0].parameters:
            for param_name in updates[0].parameters.keys():
                if np:
                    # Use numpy for efficient aggregation
                    param_values = []
                    param_weights = []
                    
                    for i, update in enumerate(updates):
                        if param_name in update.parameters:
                            param_values.append(np.array(update.parameters[param_name]))
                            param_weights.append(weights[i])
                    
                    if param_values:
                        # Weighted average
                        weighted_sum = sum(w * v for w, v in zip(param_weights, param_values))
                        aggregated_params[param_name] = weighted_sum.tolist()
                else:
                    # Fallback to simple averaging
                    param_values = [update.parameters[param_name] for update in updates 
                                  if param_name in update.parameters]
                    if param_values and isinstance(param_values[0], (int, float)):
                        aggregated_params[param_name] = sum(param_values) / len(param_values)
                    else:
                        aggregated_params[param_name] = param_values[0]  # Take first
        
        # Calculate convergence metric (simplified)
        convergence_metric = self._calculate_convergence(updates)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(updates)
        
        return AggregationResult(
            aggregation_id=str(uuid.uuid4()),
            round_number=self.round_number,
            participating_nodes=[update.node_id for update in updates],
            aggregated_parameters=aggregated_params,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            convergence_metric=convergence_metric,
            quality_score=quality_score,
            metadata={
                'total_data_size': total_data_size,
                'num_participants': len(updates),
                'aggregator': self.name
            }
        )
    
    def validate_update(self, update: ModelUpdate) -> bool:
        """Validate model update"""
        # Basic validation
        if not update.node_id or not update.parameters:
            return False
        
        # Check data size is reasonable
        if update.data_size <= 0 or update.data_size > 1000000:
            return False
        
        # Check training loss is reasonable
        if update.training_loss < 0 or update.training_loss > 1000:
            return False
        
        # Check timestamp is recent
        time_diff = (datetime.now() - update.timestamp).total_seconds()
        if time_diff > 3600:  # 1 hour
            return False
        
        return True
    
    def _calculate_convergence(self, updates: List[ModelUpdate]) -> float:
        """Calculate convergence metric"""
        if len(updates) < 2:
            return 1.0
        
        # Simple convergence based on loss variance
        losses = [update.training_loss for update in updates if update.training_loss > 0]
        
        if not losses:
            return 1.0
        
        if len(losses) == 1:
            return 1.0
        
        mean_loss = sum(losses) / len(losses)
        variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
        
        # Lower variance indicates better convergence
        convergence = 1.0 / (1.0 + variance)
        
        return min(1.0, max(0.0, convergence))
    
    def _calculate_quality_score(self, updates: List[ModelUpdate]) -> float:
        """Calculate quality score for aggregation"""
        if not updates:
            return 0.0
        
        # Factors: validation accuracy, data size, freshness
        total_score = 0.0
        
        for update in updates:
            score = 0.0
            
            # Validation accuracy factor (0-1)
            if update.validation_accuracy > 0:
                score += min(1.0, update.validation_accuracy) * 0.4
            
            # Data size factor (normalized)
            data_factor = min(1.0, update.data_size / 10000) * 0.3
            score += data_factor
            
            # Freshness factor
            time_diff = (datetime.now() - update.timestamp).total_seconds()
            freshness = max(0.0, 1.0 - time_diff / 3600) * 0.3  # 1 hour decay
            score += freshness
            
            total_score += score
        
        return total_score / len(updates)

class SecureAggregator(BaseAggregator):
    """Secure aggregator with privacy preservation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.privacy_budget = config.get('privacy_budget', 1.0)
        self.noise_multiplier = config.get('noise_multiplier', 0.1)
        
    def aggregate_updates(self, updates: List[ModelUpdate]) -> AggregationResult:
        """Aggregate with differential privacy"""
        # First do standard aggregation
        fed_avg = FederatedAveragingAggregator(self.config)
        result = fed_avg.aggregate_updates(updates)
        
        # Add differential privacy noise
        if np and self.noise_multiplier > 0:
            for param_name, param_value in result.aggregated_parameters.items():
                if isinstance(param_value, list):
                    # Add Gaussian noise
                    noise = np.random.normal(0, self.noise_multiplier, len(param_value))
                    noisy_params = np.array(param_value) + noise
                    result.aggregated_parameters[param_name] = noisy_params.tolist()
        
        # Update metadata
        result.aggregation_method = AggregationMethod.SECURE_AGGREGATION
        result.metadata['privacy_noise_added'] = self.noise_multiplier
        result.metadata['privacy_budget_used'] = sum(u.privacy_budget_used for u in updates)
        
        return result
    
    def validate_update(self, update: ModelUpdate) -> bool:
        """Validate with privacy checks"""
        # Standard validation
        if not super().validate_update(update):
            return False
        
        # Privacy budget check
        if update.privacy_budget_used > self.privacy_budget:
            return False
        
        return True

class FederatedLearningNode:
    """A node in the federated learning network"""
    
    def __init__(self, node_id: str, role: NodeRole, config: Dict[str, Any]):
        self.node_id = node_id
        self.role = role
        self.config = config
        
        # Node state
        self.status = NodeStatus.INACTIVE
        self.federation_id = config.get('federation_id')
        self.model_version = "1.0"
        self.trust_score = 1.0
        
        # Communication
        protocol_type = config.get('communication_protocol', 'http')
        if protocol_type == 'websocket':
            self.communication = WebSocketCommunicationProtocol(config.get('communication', {}))
        else:
            self.communication = HTTPCommunicationProtocol(config.get('communication', {}))
        
        # Aggregation (for coordinator/aggregator nodes)
        if role in [NodeRole.COORDINATOR, NodeRole.AGGREGATOR]:
            aggregator_type = config.get('aggregator_type', 'federated_averaging')
            if aggregator_type == 'secure':
                self.aggregator = SecureAggregator(config.get('aggregation', {}))
            else:
                self.aggregator = FederatedAveragingAggregator(config.get('aggregation', {}))
        else:
            self.aggregator = None
        
        # Known nodes
        self.known_nodes: Dict[str, NodeInfo] = {}
        
        # Message queues
        self.incoming_messages = deque(maxlen=1000)
        self.outgoing_messages = deque(maxlen=1000)
        
        # Learning state
        self.current_round = 0
        self.local_model_params = {}
        self.pending_updates = []
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.rounds_participated = 0
        self.last_aggregation_time = None
        
        # Threading
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized federated node {node_id} with role {role.value}")
    
    async def start(self):
        """Start the federated node"""
        self.running = True
        self.status = NodeStatus.ACTIVE
        
        # Initialize communication
        if hasattr(self.communication, 'initialize'):
            await self.communication.initialize()
        
        # Start background tasks
        asyncio.create_task(self._message_processing_loop())
        asyncio.create_task(self._heartbeat_loop())
        
        if self.role == NodeRole.COORDINATOR:
            asyncio.create_task(self._coordination_loop())
        
        logger.info(f"Started federated node {self.node_id}")
    
    async def stop(self):
        """Stop the federated node"""
        self.running = False
        self.status = NodeStatus.INACTIVE
        
        # Cleanup communication
        if hasattr(self.communication, 'cleanup'):
            await self.communication.cleanup()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info(f"Stopped federated node {self.node_id}")
    
    async def join_federation(self, coordinator_endpoint: str) -> bool:
        """Join a federation"""
        try:
            # Send discovery message to coordinator
            discovery_message = FederatedMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.node_id,
                receiver_id=None,
                message_type=MessageType.DISCOVERY,
                payload={
                    'node_id': self.node_id,
                    'role': self.role.value,
                    'capabilities': list(self.config.get('capabilities', [])),
                    'model_version': self.model_version
                }
            )
            
            success = await self.communication.send_message(discovery_message, coordinator_endpoint)
            
            if success:
                logger.info(f"Sent join request to coordinator at {coordinator_endpoint}")
                return True
            else:
                logger.error(f"Failed to contact coordinator at {coordinator_endpoint}")
                return False
                
        except Exception as e:
            logger.error(f"Error joining federation: {e}")
            return False
    
    async def submit_model_update(self, parameters: Dict[str, Any], 
                                data_size: int, training_loss: float = 0.0,
                                validation_accuracy: float = 0.0) -> str:
        """Submit a model update"""
        
        update = ModelUpdate(
            update_id=str(uuid.uuid4()),
            node_id=self.node_id,
            model_version=self.model_version,
            parameters=parameters,
            data_size=data_size,
            training_loss=training_loss,
            validation_accuracy=validation_accuracy,
            privacy_budget_used=0.0  # Would be calculated based on DP mechanism
        )
        
        # Send to coordinator/aggregator
        coordinator_id = self._find_coordinator()
        if coordinator_id:
            message = FederatedMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.node_id,
                receiver_id=coordinator_id,
                message_type=MessageType.MODEL_UPDATE,
                payload={
                    'update': {
                        'update_id': update.update_id,
                        'parameters': update.parameters,
                        'data_size': update.data_size,
                        'training_loss': update.training_loss,
                        'validation_accuracy': update.validation_accuracy,
                        'timestamp': update.timestamp.isoformat()
                    }
                }
            )
            
            # Find coordinator endpoint
            coordinator_endpoint = self._get_node_endpoint(coordinator_id)
            if coordinator_endpoint:
                success = await self.communication.send_message(message, coordinator_endpoint)
                if success:
                    self.messages_sent += 1
                    logger.info(f"Submitted model update {update.update_id}")
                    return update.update_id
        
        logger.error("Failed to submit model update")
        return ""
    
    async def _message_processing_loop(self):
        """Process incoming messages"""
        while self.running:
            try:
                # Receive messages
                messages = await self.communication.receive_messages()
                
                for message in messages:
                    await self._handle_message(message)
                    self.messages_received += 1
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: FederatedMessage):
        """Handle incoming message"""
        try:
            if message.message_type == MessageType.MODEL_UPDATE:
                await self._handle_model_update(message)
            elif message.message_type == MessageType.AGGREGATION_RESULT:
                await self._handle_aggregation_result(message)
            elif message.message_type == MessageType.DISCOVERY:
                await self._handle_discovery(message)
            elif message.message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
            elif message.message_type == MessageType.COORDINATION:
                await self._handle_coordination(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message {message.message_id}: {e}")
    
    async def _handle_model_update(self, message: FederatedMessage):
        """Handle model update message"""
        if self.role not in [NodeRole.COORDINATOR, NodeRole.AGGREGATOR]:
            return
        
        try:
            update_data = message.payload['update']
            
            update = ModelUpdate(
                update_id=update_data['update_id'],
                node_id=message.sender_id,
                model_version=self.model_version,
                parameters=update_data['parameters'],
                data_size=update_data['data_size'],
                training_loss=update_data['training_loss'],
                validation_accuracy=update_data['validation_accuracy'],
                timestamp=datetime.fromisoformat(update_data['timestamp'])
            )
            
            # Validate update
            if self.aggregator and self.aggregator.validate_update(update):
                self.pending_updates.append(update)
                logger.info(f"Received valid model update from {message.sender_id}")
                
                # Check if we have enough updates to aggregate
                min_updates = self.config.get('min_updates_for_aggregation', 2)
                if len(self.pending_updates) >= min_updates:
                    await self._perform_aggregation()
            else:
                logger.warning(f"Invalid model update from {message.sender_id}")
                
        except Exception as e:
            logger.error(f"Error handling model update: {e}")
    
    async def _handle_aggregation_result(self, message: FederatedMessage):
        """Handle aggregation result"""
        try:
            result_data = message.payload['result']
            
            # Update local model with aggregated parameters
            self.local_model_params = result_data['aggregated_parameters']
            self.model_version = result_data.get('model_version', self.model_version)
            self.current_round = result_data.get('round_number', self.current_round)
            
            logger.info(f"Received aggregation result for round {self.current_round}")
            
        except Exception as e:
            logger.error(f"Error handling aggregation result: {e}")
    
    async def _handle_discovery(self, message: FederatedMessage):
        """Handle node discovery"""
        try:
            payload = message.payload
            
            # Add node to known nodes
            node_info = NodeInfo(
                node_id=payload['node_id'],
                role=NodeRole(payload['role']),
                status=NodeStatus.ACTIVE,
                endpoint="",  # Would be extracted from message metadata
                capabilities=set(payload.get('capabilities', [])),
                model_version=payload.get('model_version', "1.0")
            )
            
            self.known_nodes[node_info.node_id] = node_info
            
            logger.info(f"Discovered node {node_info.node_id} with role {node_info.role.value}")
            
        except Exception as e:
            logger.error(f"Error handling discovery: {e}")
    
    async def _handle_heartbeat(self, message: FederatedMessage):
        """Handle heartbeat message"""
        sender_id = message.sender_id
        
        if sender_id in self.known_nodes:
            self.known_nodes[sender_id].last_seen = datetime.now()
            self.known_nodes[sender_id].status = NodeStatus.ACTIVE
    
    async def _handle_coordination(self, message: FederatedMessage):
        """Handle coordination message"""
        # Implementation depends on coordination protocol
        pass
    
    async def _perform_aggregation(self):
        """Perform model aggregation"""
        if not self.aggregator or not self.pending_updates:
            return
        
        try:
            # Aggregate updates
            result = self.aggregator.aggregate_updates(self.pending_updates)
            
            # Update local state
            self.current_round = result.round_number
            self.last_aggregation_time = datetime.now()
            
            # Broadcast result to participants
            broadcast_message = FederatedMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.node_id,
                receiver_id=None,  # Broadcast
                message_type=MessageType.AGGREGATION_RESULT,
                payload={
                    'result': {
                        'aggregation_id': result.aggregation_id,
                        'round_number': result.round_number,
                        'aggregated_parameters': result.aggregated_parameters,
                        'convergence_metric': result.convergence_metric,
                        'quality_score': result.quality_score,
                        'model_version': self.model_version
                    }
                }
            )
            
            await self.communication.broadcast_message(broadcast_message)
            
            # Clear pending updates
            self.pending_updates.clear()
            
            logger.info(f"Performed aggregation for round {result.round_number} "
                       f"with {len(result.participating_nodes)} participants")
            
        except Exception as e:
            logger.error(f"Error performing aggregation: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        heartbeat_interval = self.config.get('heartbeat_interval', 30)  # seconds
        
        while self.running:
            try:
                heartbeat_message = FederatedMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.node_id,
                    receiver_id=None,  # Broadcast
                    message_type=MessageType.HEARTBEAT,
                    payload={
                        'status': self.status.value,
                        'model_version': self.model_version,
                        'current_round': self.current_round
                    }
                )
                
                await self.communication.broadcast_message(heartbeat_message)
                
                await asyncio.sleep(heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(heartbeat_interval)
    
    async def _coordination_loop(self):
        """Coordination loop for coordinator nodes"""
        coordination_interval = self.config.get('coordination_interval', 60)  # seconds
        
        while self.running:
            try:
                # Perform coordination tasks
                await self._cleanup_inactive_nodes()
                await self._check_federation_health()
                
                await asyncio.sleep(coordination_interval)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(coordination_interval)
    
    async def _cleanup_inactive_nodes(self):
        """Remove inactive nodes"""
        current_time = datetime.now()
        inactive_threshold = timedelta(minutes=5)
        
        inactive_nodes = []
        for node_id, node_info in self.known_nodes.items():
            if current_time - node_info.last_seen > inactive_threshold:
                inactive_nodes.append(node_id)
        
        for node_id in inactive_nodes:
            del self.known_nodes[node_id]
            logger.info(f"Removed inactive node {node_id}")
    
    async def _check_federation_health(self):
        """Check overall federation health"""
        active_nodes = sum(1 for node in self.known_nodes.values() 
                          if node.status == NodeStatus.ACTIVE)
        
        min_nodes = self.config.get('min_active_nodes', 2)
        
        if active_nodes < min_nodes:
            logger.warning(f"Federation health warning: only {active_nodes} active nodes "
                          f"(minimum: {min_nodes})")
    
    def _find_coordinator(self) -> Optional[str]:
        """Find coordinator node ID"""
        for node_id, node_info in self.known_nodes.items():
            if node_info.role == NodeRole.COORDINATOR:
                return node_id
        return None
    
    def _get_node_endpoint(self, node_id: str) -> Optional[str]:
        """Get endpoint for a node"""
        if node_id in self.known_nodes:
            return self.known_nodes[node_id].endpoint
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get node statistics"""
        return {
            'node_id': self.node_id,
            'role': self.role.value,
            'status': self.status.value,
            'current_round': self.current_round,
            'known_nodes': len(self.known_nodes),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'rounds_participated': self.rounds_participated,
            'pending_updates': len(self.pending_updates),
            'last_aggregation': self.last_aggregation_time.isoformat() if self.last_aggregation_time else None,
            'model_version': self.model_version,
            'trust_score': self.trust_score
        }

class FederatedLearningManager:
    """Manager for federated learning operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, FederatedLearningNode] = {}
        self.federations: Dict[str, FederationConfig] = {}
        
        # Statistics
        self.total_rounds_completed = 0
        self.total_updates_processed = 0
        self.federation_creation_time = datetime.now()
        
    async def create_federation(self, federation_config: FederationConfig) -> str:
        """Create a new federation"""
        self.federations[federation_config.federation_id] = federation_config
        
        logger.info(f"Created federation {federation_config.federation_id} "
                   f"with strategy {federation_config.learning_strategy.value}")
        
        return federation_config.federation_id
    
    async def create_node(self, node_id: str, role: NodeRole, 
                         federation_id: str, node_config: Dict[str, Any]) -> FederatedLearningNode:
        """Create a new federated node"""
        
        if federation_id not in self.federations:
            raise ValueError(f"Federation {federation_id} does not exist")
        
        # Merge federation config with node config
        full_config = {**self.config, **node_config}
        full_config['federation_id'] = federation_id
        
        node = FederatedLearningNode(node_id, role, full_config)
        self.nodes[node_id] = node
        
        logger.info(f"Created node {node_id} with role {role.value} in federation {federation_id}")
        
        return node
    
    async def start_node(self, node_id: str) -> bool:
        """Start a federated node"""
        if node_id not in self.nodes:
            return False
        
        await self.nodes[node_id].start()
        return True
    
    async def stop_node(self, node_id: str) -> bool:
        """Stop a federated node"""
        if node_id not in self.nodes:
            return False
        
        await self.nodes[node_id].stop()
        return True
    
    async def join_federation(self, node_id: str, coordinator_endpoint: str) -> bool:
        """Have a node join a federation"""
        if node_id not in self.nodes:
            return False
        
        return await self.nodes[node_id].join_federation(coordinator_endpoint)
    
    def get_federation_statistics(self, federation_id: str) -> Dict[str, Any]:
        """Get statistics for a federation"""
        if federation_id not in self.federations:
            return {}
        
        federation_nodes = [node for node in self.nodes.values() 
                          if node.federation_id == federation_id]
        
        active_nodes = sum(1 for node in federation_nodes 
                          if node.status == NodeStatus.ACTIVE)
        
        total_messages = sum(node.messages_sent + node.messages_received 
                           for node in federation_nodes)
        
        return {
            'federation_id': federation_id,
            'total_nodes': len(federation_nodes),
            'active_nodes': active_nodes,
            'total_messages': total_messages,
            'total_rounds': self.total_rounds_completed,
            'total_updates': self.total_updates_processed,
            'uptime': (datetime.now() - self.federation_creation_time).total_seconds(),
            'node_roles': {role.value: sum(1 for node in federation_nodes if node.role == role) 
                          for role in NodeRole}
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'total_federations': len(self.federations),
            'total_nodes': len(self.nodes),
            'active_nodes': sum(1 for node in self.nodes.values() 
                              if node.status == NodeStatus.ACTIVE),
            'total_rounds_completed': self.total_rounds_completed,
            'total_updates_processed': self.total_updates_processed,
            'federations': {fid: self.get_federation_statistics(fid) 
                          for fid in self.federations.keys()},
            'nodes': {nid: node.get_statistics() 
                     for nid, node in self.nodes.items()}
        }

# Example usage and utility functions

def create_federation_config(federation_id: str, 
                           learning_strategy: LearningStrategy = LearningStrategy.FEDERATED_AVERAGING,
                           min_participants: int = 2) -> FederationConfig:
    """Create federation configuration"""
    return FederationConfig(
        federation_id=federation_id,
        learning_strategy=learning_strategy,
        aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        min_participants=min_participants,
        max_participants=50,
        round_duration=300,
        convergence_threshold=0.001,
        privacy_budget=1.0,
        trust_threshold=0.5
    )

async def example_usage():
    """Example usage of federated learning system"""
    
    # Configuration
    config = {
        'communication_protocol': 'http',
        'aggregator_type': 'federated_averaging',
        'heartbeat_interval': 30,
        'coordination_interval': 60,
        'min_updates_for_aggregation': 2,
        'communication': {
            'port': 8080,
            'timeout': 30
        },
        'aggregation': {
            'privacy_budget': 1.0,
            'noise_multiplier': 0.1
        }
    }
    
    # Initialize manager
    manager = FederatedLearningManager(config)
    
    # Create federation
    federation_config = create_federation_config(
        "studsar_federation_v1",
        LearningStrategy.FEDERATED_AVERAGING,
        min_participants=3
    )
    
    federation_id = await manager.create_federation(federation_config)
    print(f"Created federation: {federation_id}")
    
    # Create nodes
    coordinator = await manager.create_node(
        "coordinator_1", 
        NodeRole.COORDINATOR, 
        federation_id, 
        {'capabilities': ['aggregation', 'coordination']}
    )
    
    participant1 = await manager.create_node(
        "participant_1", 
        NodeRole.PARTICIPANT, 
        federation_id, 
        {'capabilities': ['training']}
    )
    
    participant2 = await manager.create_node(
        "participant_2", 
        NodeRole.PARTICIPANT, 
        federation_id, 
        {'capabilities': ['training']}
    )
    
    # Start nodes
    await manager.start_node("coordinator_1")
    await manager.start_node("participant_1")
    await manager.start_node("participant_2")
    
    print("Started all nodes")
    
    # Simulate model updates
    await asyncio.sleep(2)  # Let nodes initialize
    
    # Participant 1 submits update
    update_id1 = await participant1.submit_model_update(
        parameters={'weights': [0.1, 0.2, 0.3], 'bias': [0.01]},
        data_size=1000,
        training_loss=0.5,
        validation_accuracy=0.85
    )
    
    # Participant 2 submits update
    update_id2 = await participant2.submit_model_update(
        parameters={'weights': [0.15, 0.25, 0.35], 'bias': [0.02]},
        data_size=800,
        training_loss=0.45,
        validation_accuracy=0.87
    )
    
    print(f"Submitted updates: {update_id1}, {update_id2}")
    
    # Wait for aggregation
    await asyncio.sleep(3)
    
    # Get statistics
    stats = manager.get_all_statistics()
    print(f"\nFederation Statistics:")
    print(f"Total federations: {stats['total_federations']}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Active nodes: {stats['active_nodes']}")
    
    # Stop nodes
    await manager.stop_node("coordinator_1")
    await manager.stop_node("participant_1")
    await manager.stop_node("participant_2")
    
    print("Stopped all nodes")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())