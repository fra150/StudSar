"""Attention Mechanisms Module for StudSar V4

Implements various attention mechanisms for cognitive focus and resource allocation:
- Selective Attention: Focus on relevant information
- Sustained Attention: Maintain focus over time
- Divided Attention: Manage multiple concurrent tasks
- Executive Attention: Control and coordinate attention
"""

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
import json
from collections import defaultdict, deque
import heapq

logger = logging.getLogger(__name__)

class AttentionType(Enum):
    """Types of attention mechanisms"""
    SELECTIVE = "selective"
    SUSTAINED = "sustained"
    DIVIDED = "divided"
    EXECUTIVE = "executive"

class AttentionState(Enum):
    """Current state of attention"""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    SWITCHING = "switching"
    OVERLOADED = "overloaded"
    IDLE = "idle"

class Priority(Enum):
    """Priority levels for attention allocation"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1

@dataclass
class AttentionTarget:
    """Object that can receive attention"""
    id: str
    content: Any
    priority: Priority
    attention_weight: float = 0.0
    last_attended: Optional[datetime] = None
    total_attention_time: float = 0.0
    activation_level: float = 0.0
    decay_rate: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_attention(self, attention_duration: float):
        """Update attention statistics"""
        self.last_attended = datetime.now()
        self.total_attention_time += attention_duration
        self.activation_level = min(1.0, self.activation_level + attention_duration * 0.1)
    
    def decay_activation(self, time_elapsed: float):
        """Decay activation over time"""
        decay_factor = math.exp(-self.decay_rate * time_elapsed)
        self.activation_level *= decay_factor
    
    def get_attention_score(self) -> float:
        """Calculate current attention score"""
        priority_weight = self.priority.value / 5.0
        recency_weight = 1.0
        
        if self.last_attended:
            time_since = (datetime.now() - self.last_attended).total_seconds() / 3600
            recency_weight = math.exp(-0.1 * time_since)
        
        return (priority_weight * 0.4 + 
                self.activation_level * 0.3 + 
                recency_weight * 0.2 + 
                self.attention_weight * 0.1)

@dataclass
class AttentionContext:
    """Context information for attention decisions"""
    current_task: Optional[str] = None
    active_targets: Set[str] = field(default_factory=set)
    cognitive_load: float = 0.0
    available_resources: float = 1.0
    interruption_threshold: float = 0.7
    focus_duration: float = 0.0
    last_switch_time: Optional[datetime] = None
    switch_cost: float = 0.1
    fatigue_level: float = 0.0

class BaseAttentionMechanism(ABC):
    """Abstract base class for attention mechanisms"""
    
    def __init__(self, attention_type: AttentionType, config: Dict[str, Any]):
        self.attention_type = attention_type
        self.config = config
        self.targets: Dict[str, AttentionTarget] = {}
        self.context = AttentionContext()
        self.state = AttentionState.IDLE
        self.attention_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, float] = defaultdict(float)
    
    @abstractmethod
    async def allocate_attention(self, targets: List[AttentionTarget]) -> Dict[str, float]:
        """Allocate attention across targets"""
        pass
    
    @abstractmethod
    async def update_attention(self, target_id: str, duration: float) -> bool:
        """Update attention for a specific target"""
        pass
    
    @abstractmethod
    async def should_switch_attention(self, new_target: AttentionTarget) -> bool:
        """Determine if attention should switch to new target"""
        pass
    
    def add_target(self, target: AttentionTarget):
        """Add attention target"""
        self.targets[target.id] = target
        logger.debug(f"Added attention target: {target.id}")
    
    def remove_target(self, target_id: str) -> Optional[AttentionTarget]:
        """Remove attention target"""
        return self.targets.pop(target_id, None)
    
    def get_target(self, target_id: str) -> Optional[AttentionTarget]:
        """Get attention target by ID"""
        return self.targets.get(target_id)
    
    def update_context(self, **kwargs):
        """Update attention context"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def get_attention_state(self) -> Dict[str, Any]:
        """Get current attention state"""
        return {
            'type': self.attention_type.value,
            'state': self.state.value,
            'active_targets': list(self.context.active_targets),
            'cognitive_load': self.context.cognitive_load,
            'available_resources': self.context.available_resources,
            'focus_duration': self.context.focus_duration,
            'fatigue_level': self.context.fatigue_level,
            'target_count': len(self.targets),
            'performance_metrics': dict(self.performance_metrics)
        }
    
    def _calculate_switch_cost(self, from_target: str, to_target: str) -> float:
        """Calculate cost of switching attention"""
        base_cost = self.context.switch_cost
        
        # Increase cost based on cognitive load
        load_multiplier = 1.0 + self.context.cognitive_load
        
        # Increase cost based on fatigue
        fatigue_multiplier = 1.0 + self.context.fatigue_level * 0.5
        
        # Decrease cost if targets are related
        similarity_discount = 0.0
        if from_target in self.targets and to_target in self.targets:
            # Simple similarity based on metadata
            from_meta = self.targets[from_target].metadata
            to_meta = self.targets[to_target].metadata
            common_keys = set(from_meta.keys()) & set(to_meta.keys())
            if common_keys:
                similarity_discount = len(common_keys) * 0.1
        
        return max(0.01, base_cost * load_multiplier * fatigue_multiplier - similarity_discount)
    
    def _update_fatigue(self, attention_duration: float):
        """Update fatigue level based on attention duration"""
        # Fatigue increases with sustained attention
        fatigue_increase = attention_duration * 0.01
        self.context.fatigue_level = min(1.0, self.context.fatigue_level + fatigue_increase)
        
        # Fatigue decreases over time when not attending
        if attention_duration == 0:
            recovery_rate = 0.05  # Recovery per time unit
            self.context.fatigue_level = max(0.0, self.context.fatigue_level - recovery_rate)

class SelectiveAttention(BaseAttentionMechanism):
    """Selective attention mechanism - focus on most relevant targets"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AttentionType.SELECTIVE, config)
        self.focus_threshold = config.get('focus_threshold', 0.6)
        self.max_concurrent_targets = config.get('max_concurrent_targets', 3)
        self.inhibition_strength = config.get('inhibition_strength', 0.8)
    
    async def allocate_attention(self, targets: List[AttentionTarget]) -> Dict[str, float]:
        """Allocate attention using selective mechanism"""
        if not targets:
            return {}
        
        # Calculate attention scores for all targets
        scored_targets = [(target, target.get_attention_score()) for target in targets]
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        
        # Select top targets based on threshold and max concurrent
        selected_targets = []
        total_score = sum(score for _, score in scored_targets)
        
        for target, score in scored_targets:
            if (len(selected_targets) < self.max_concurrent_targets and 
                score >= self.focus_threshold * total_score / len(scored_targets)):
                selected_targets.append((target, score))
        
        # Allocate attention proportionally among selected targets
        allocation = {}
        if selected_targets:
            selected_total = sum(score for _, score in selected_targets)
            available_attention = self.context.available_resources
            
            for target, score in selected_targets:
                attention_amount = (score / selected_total) * available_attention
                allocation[target.id] = attention_amount
                
                # Update target
                target.attention_weight = attention_amount
                if target.id not in self.context.active_targets:
                    self.context.active_targets.add(target.id)
        
        # Update cognitive load
        self.context.cognitive_load = len(selected_targets) / self.max_concurrent_targets
        
        # Update state
        if len(selected_targets) == 1:
            self.state = AttentionState.FOCUSED
        elif len(selected_targets) > 1:
            self.state = AttentionState.DIVIDED
        else:
            self.state = AttentionState.IDLE
        
        logger.debug(f"Selective attention allocated to {len(selected_targets)} targets")
        return allocation
    
    async def update_attention(self, target_id: str, duration: float) -> bool:
        """Update attention for specific target"""
        if target_id not in self.targets:
            return False
        
        target = self.targets[target_id]
        target.update_attention(duration)
        
        # Update context
        self.context.focus_duration += duration
        self._update_fatigue(duration)
        
        # Record attention event
        self.attention_history.append({
            'timestamp': datetime.now(),
            'target_id': target_id,
            'duration': duration,
            'attention_type': 'selective'
        })
        
        # Update performance metrics
        self.performance_metrics['total_attention_time'] += duration
        self.performance_metrics['target_switches'] += 1 if self.context.current_task != target_id else 0
        
        self.context.current_task = target_id
        return True
    
    async def should_switch_attention(self, new_target: AttentionTarget) -> bool:
        """Determine if attention should switch to new target"""
        if not self.context.active_targets:
            return True
        
        # Calculate switch cost
        current_target = self.context.current_task
        if current_target:
            switch_cost = self._calculate_switch_cost(current_target, new_target.id)
        else:
            switch_cost = 0.0
        
        # Calculate benefit of switching
        new_score = new_target.get_attention_score()
        current_scores = [self.targets[tid].get_attention_score() 
                         for tid in self.context.active_targets 
                         if tid in self.targets]
        
        max_current_score = max(current_scores) if current_scores else 0.0
        
        # Switch if benefit exceeds cost and threshold
        benefit = new_score - max_current_score
        should_switch = benefit > switch_cost + self.context.interruption_threshold
        
        if should_switch:
            self.context.last_switch_time = datetime.now()
            self.state = AttentionState.SWITCHING
        
        return should_switch

class SustainedAttention(BaseAttentionMechanism):
    """Sustained attention mechanism - maintain focus over time"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AttentionType.SUSTAINED, config)
        self.vigilance_threshold = config.get('vigilance_threshold', 0.5)
        self.fatigue_resistance = config.get('fatigue_resistance', 0.7)
        self.focus_boost_rate = config.get('focus_boost_rate', 0.05)
        self.distraction_resistance = config.get('distraction_resistance', 0.8)
    
    async def allocate_attention(self, targets: List[AttentionTarget]) -> Dict[str, float]:
        """Allocate sustained attention to primary target"""
        if not targets:
            return {}
        
        # Find primary target (highest priority or current focus)
        primary_target = None
        if self.context.current_task and self.context.current_task in [t.id for t in targets]:
            # Continue with current task if possible
            primary_target = next(t for t in targets if t.id == self.context.current_task)
        else:
            # Select highest priority target
            primary_target = max(targets, key=lambda t: t.get_attention_score())
        
        # Allocate most attention to primary target
        allocation = {}
        available_attention = self.context.available_resources
        
        # Adjust available attention based on fatigue
        fatigue_penalty = self.context.fatigue_level * (1.0 - self.fatigue_resistance)
        effective_attention = available_attention * (1.0 - fatigue_penalty)
        
        # Primary target gets majority of attention
        primary_attention = effective_attention * 0.8
        allocation[primary_target.id] = primary_attention
        
        # Distribute remaining attention among other targets
        other_targets = [t for t in targets if t.id != primary_target.id]
        if other_targets:
            remaining_attention = effective_attention * 0.2
            attention_per_other = remaining_attention / len(other_targets)
            
            for target in other_targets:
                allocation[target.id] = attention_per_other
        
        # Update context
        self.context.active_targets = {primary_target.id}
        self.context.cognitive_load = 0.3  # Lower load for sustained focus
        
        # Update state based on vigilance
        if self.context.fatigue_level < self.vigilance_threshold:
            self.state = AttentionState.FOCUSED
        else:
            self.state = AttentionState.DISTRACTED
        
        logger.debug(f"Sustained attention focused on {primary_target.id}")
        return allocation
    
    async def update_attention(self, target_id: str, duration: float) -> bool:
        """Update sustained attention"""
        if target_id not in self.targets:
            return False
        
        target = self.targets[target_id]
        target.update_attention(duration)
        
        # Boost focus for sustained attention
        if target_id == self.context.current_task:
            focus_boost = duration * self.focus_boost_rate
            target.activation_level = min(1.0, target.activation_level + focus_boost)
            self.context.focus_duration += duration
        
        # Update fatigue with resistance
        fatigue_increase = duration * 0.01 * (1.0 - self.fatigue_resistance)
        self.context.fatigue_level = min(1.0, self.context.fatigue_level + fatigue_increase)
        
        # Record attention event
        self.attention_history.append({
            'timestamp': datetime.now(),
            'target_id': target_id,
            'duration': duration,
            'attention_type': 'sustained'
        })
        
        # Update performance metrics
        self.performance_metrics['sustained_focus_time'] += duration
        if target_id == self.context.current_task:
            self.performance_metrics['focus_stability'] += 1
        
        self.context.current_task = target_id
        return True
    
    async def should_switch_attention(self, new_target: AttentionTarget) -> bool:
        """Determine if sustained attention should switch"""
        # Sustained attention resists switching unless new target is much more important
        if not self.context.current_task:
            return True
        
        current_target = self.targets.get(self.context.current_task)
        if not current_target:
            return True
        
        # Calculate resistance to switching
        focus_momentum = min(1.0, self.context.focus_duration / 300)  # 5 minutes to full momentum
        resistance = self.distraction_resistance * focus_momentum
        
        # New target must significantly exceed current target
        current_score = current_target.get_attention_score()
        new_score = new_target.get_attention_score()
        
        threshold = current_score + resistance + self.context.interruption_threshold
        should_switch = new_score > threshold
        
        if should_switch:
            # Reset focus duration on switch
            self.context.focus_duration = 0.0
            self.context.last_switch_time = datetime.now()
            self.state = AttentionState.SWITCHING
        
        return should_switch

class DividedAttention(BaseAttentionMechanism):
    """Divided attention mechanism - manage multiple concurrent tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AttentionType.DIVIDED, config)
        self.max_divisions = config.get('max_divisions', 4)
        self.time_slice_duration = config.get('time_slice_duration', 1.0)  # seconds
        self.load_balancing = config.get('load_balancing', True)
        self.interference_threshold = config.get('interference_threshold', 0.8)
        self.task_queue: List[Tuple[float, str]] = []  # (priority, target_id)
    
    async def allocate_attention(self, targets: List[AttentionTarget]) -> Dict[str, float]:
        """Allocate divided attention across multiple targets"""
        if not targets:
            return {}
        
        # Limit number of concurrent targets
        active_targets = targets[:self.max_divisions]
        
        # Calculate base allocation
        allocation = {}
        available_attention = self.context.available_resources
        
        if self.load_balancing:
            # Equal distribution with priority weighting
            total_priority = sum(t.priority.value for t in active_targets)
            
            for target in active_targets:
                priority_weight = target.priority.value / total_priority
                base_allocation = available_attention / len(active_targets)
                weighted_allocation = base_allocation * (0.5 + 0.5 * priority_weight)
                allocation[target.id] = weighted_allocation
        else:
            # Simple equal distribution
            attention_per_target = available_attention / len(active_targets)
            for target in active_targets:
                allocation[target.id] = attention_per_target
        
        # Adjust for interference
        interference_penalty = self._calculate_interference(active_targets)
        for target_id in allocation:
            allocation[target_id] *= (1.0 - interference_penalty)
        
        # Update context
        self.context.active_targets = {t.id for t in active_targets}
        self.context.cognitive_load = len(active_targets) / self.max_divisions
        
        # Update state
        if len(active_targets) > 1:
            if self.context.cognitive_load > self.interference_threshold:
                self.state = AttentionState.OVERLOADED
            else:
                self.state = AttentionState.DIVIDED
        else:
            self.state = AttentionState.FOCUSED
        
        # Update task queue for time slicing
        self.task_queue = [(target.get_attention_score(), target.id) 
                          for target in active_targets]
        heapq.heapify(self.task_queue)
        
        logger.debug(f"Divided attention across {len(active_targets)} targets")
        return allocation
    
    async def update_attention(self, target_id: str, duration: float) -> bool:
        """Update divided attention with time slicing"""
        if target_id not in self.targets:
            return False
        
        target = self.targets[target_id]
        
        # Implement time slicing
        time_slice = min(duration, self.time_slice_duration)
        target.update_attention(time_slice)
        
        # Update context
        self.context.focus_duration += time_slice
        self._update_fatigue(time_slice)
        
        # Record attention event
        self.attention_history.append({
            'timestamp': datetime.now(),
            'target_id': target_id,
            'duration': time_slice,
            'attention_type': 'divided'
        })
        
        # Update performance metrics
        self.performance_metrics['divided_attention_time'] += time_slice
        self.performance_metrics['task_switches'] += 1
        
        # Rotate to next task in queue if time slice is complete
        if time_slice >= self.time_slice_duration and len(self.task_queue) > 1:
            await self._rotate_attention()
        
        return True
    
    async def should_switch_attention(self, new_target: AttentionTarget) -> bool:
        """Determine if divided attention should include new target"""
        # Add new target if we have capacity
        if len(self.context.active_targets) < self.max_divisions:
            return True
        
        # Replace lowest priority target if new target has higher priority
        if self.context.active_targets:
            min_priority_target = min(
                (self.targets[tid] for tid in self.context.active_targets if tid in self.targets),
                key=lambda t: t.get_attention_score()
            )
            
            return new_target.get_attention_score() > min_priority_target.get_attention_score()
        
        return True
    
    def _calculate_interference(self, targets: List[AttentionTarget]) -> float:
        """Calculate interference between concurrent tasks"""
        if len(targets) <= 1:
            return 0.0
        
        # Simple interference model based on task similarity
        interference = 0.0
        for i, target1 in enumerate(targets):
            for target2 in targets[i+1:]:
                # Calculate similarity based on metadata
                similarity = self._calculate_task_similarity(target1, target2)
                # Higher similarity means more interference
                interference += similarity * 0.1
        
        # Normalize by number of pairs
        num_pairs = len(targets) * (len(targets) - 1) / 2
        return min(1.0, interference / num_pairs) if num_pairs > 0 else 0.0
    
    def _calculate_task_similarity(self, target1: AttentionTarget, target2: AttentionTarget) -> float:
        """Calculate similarity between two tasks"""
        # Simple similarity based on shared metadata keys
        meta1_keys = set(target1.metadata.keys())
        meta2_keys = set(target2.metadata.keys())
        
        if not meta1_keys and not meta2_keys:
            return 0.5  # Default similarity
        
        intersection = len(meta1_keys & meta2_keys)
        union = len(meta1_keys | meta2_keys)
        
        return intersection / union if union > 0 else 0.0
    
    async def _rotate_attention(self):
        """Rotate attention to next task in queue"""
        if len(self.task_queue) > 1:
            # Move current task to end of queue
            current_priority, current_id = heapq.heappop(self.task_queue)
            heapq.heappush(self.task_queue, (current_priority, current_id))
            
            # Update current task
            if self.task_queue:
                _, next_id = self.task_queue[0]
                self.context.current_task = next_id

class ExecutiveAttention(BaseAttentionMechanism):
    """Executive attention mechanism - control and coordinate other attention systems"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AttentionType.EXECUTIVE, config)
        self.control_threshold = config.get('control_threshold', 0.7)
        self.coordination_strength = config.get('coordination_strength', 0.8)
        self.conflict_resolution = config.get('conflict_resolution', 'priority')
        self.attention_systems: Dict[AttentionType, BaseAttentionMechanism] = {}
        self.system_weights: Dict[AttentionType, float] = {
            AttentionType.SELECTIVE: 0.3,
            AttentionType.SUSTAINED: 0.3,
            AttentionType.DIVIDED: 0.4
        }
    
    def add_attention_system(self, system: BaseAttentionMechanism):
        """Add attention system to coordinate"""
        self.attention_systems[system.attention_type] = system
    
    async def allocate_attention(self, targets: List[AttentionTarget]) -> Dict[str, float]:
        """Coordinate attention allocation across systems"""
        if not targets or not self.attention_systems:
            return {}
        
        # Get allocations from each system
        system_allocations = {}
        for att_type, system in self.attention_systems.items():
            allocation = await system.allocate_attention(targets)
            system_allocations[att_type] = allocation
        
        # Resolve conflicts and coordinate
        final_allocation = await self._coordinate_allocations(system_allocations, targets)
        
        # Update executive context
        self.context.active_targets = set(final_allocation.keys())
        self.context.cognitive_load = self._calculate_total_load()
        
        # Determine executive state
        if self.context.cognitive_load > self.control_threshold:
            self.state = AttentionState.OVERLOADED
        elif len(self.context.active_targets) > 1:
            self.state = AttentionState.DIVIDED
        else:
            self.state = AttentionState.FOCUSED
        
        logger.debug(f"Executive attention coordinated {len(final_allocation)} targets")
        return final_allocation
    
    async def update_attention(self, target_id: str, duration: float) -> bool:
        """Update executive attention and coordinate systems"""
        # Update all relevant systems
        updated = False
        for system in self.attention_systems.values():
            if target_id in system.targets:
                system_updated = await system.update_attention(target_id, duration)
                updated = updated or system_updated
        
        if updated and target_id in self.targets:
            target = self.targets[target_id]
            target.update_attention(duration)
            
            # Update executive context
            self.context.focus_duration += duration
            self._update_fatigue(duration)
            
            # Record executive attention event
            self.attention_history.append({
                'timestamp': datetime.now(),
                'target_id': target_id,
                'duration': duration,
                'attention_type': 'executive'
            })
            
            # Update performance metrics
            self.performance_metrics['executive_control_time'] += duration
            self.performance_metrics['coordination_events'] += 1
        
        return updated
    
    async def should_switch_attention(self, new_target: AttentionTarget) -> bool:
        """Executive decision on attention switching"""
        # Consult all systems and make executive decision
        system_decisions = {}
        for att_type, system in self.attention_systems.items():
            decision = await system.should_switch_attention(new_target)
            system_decisions[att_type] = decision
        
        # Make executive decision based on system consensus and weights
        weighted_score = sum(
            self.system_weights.get(att_type, 0.0) * (1.0 if decision else 0.0)
            for att_type, decision in system_decisions.items()
        )
        
        # Executive override based on priority and context
        if new_target.priority == Priority.CRITICAL:
            return True
        
        if self.context.cognitive_load > self.control_threshold:
            # More conservative when overloaded
            return weighted_score > 0.7
        else:
            return weighted_score > 0.5
    
    async def _coordinate_allocations(self, system_allocations: Dict[AttentionType, Dict[str, float]], 
                                    targets: List[AttentionTarget]) -> Dict[str, float]:
        """Coordinate allocations from different attention systems"""
        final_allocation = defaultdict(float)
        
        # Combine allocations based on system weights
        for att_type, allocation in system_allocations.items():
            weight = self.system_weights.get(att_type, 0.0)
            for target_id, attention in allocation.items():
                final_allocation[target_id] += attention * weight
        
        # Resolve conflicts based on strategy
        if self.conflict_resolution == 'priority':
            final_allocation = self._resolve_by_priority(final_allocation, targets)
        elif self.conflict_resolution == 'consensus':
            final_allocation = self._resolve_by_consensus(system_allocations)
        elif self.conflict_resolution == 'weighted_average':
            # Already done above
            pass
        
        # Normalize to available resources
        total_allocation = sum(final_allocation.values())
        if total_allocation > self.context.available_resources:
            scale_factor = self.context.available_resources / total_allocation
            final_allocation = {tid: attention * scale_factor 
                              for tid, attention in final_allocation.items()}
        
        return dict(final_allocation)
    
    def _resolve_by_priority(self, allocation: Dict[str, float], 
                           targets: List[AttentionTarget]) -> Dict[str, float]:
        """Resolve conflicts by giving priority to higher-priority targets"""
        target_priorities = {t.id: t.priority.value for t in targets}
        
        # Sort by priority and allocate in order
        sorted_targets = sorted(allocation.items(), 
                              key=lambda x: target_priorities.get(x[0], 0), 
                              reverse=True)
        
        resolved_allocation = {}
        remaining_resources = self.context.available_resources
        
        for target_id, requested_attention in sorted_targets:
            allocated_attention = min(requested_attention, remaining_resources)
            if allocated_attention > 0:
                resolved_allocation[target_id] = allocated_attention
                remaining_resources -= allocated_attention
        
        return resolved_allocation
    
    def _resolve_by_consensus(self, system_allocations: Dict[AttentionType, Dict[str, float]]) -> Dict[str, float]:
        """Resolve conflicts by finding consensus among systems"""
        consensus_allocation = defaultdict(list)
        
        # Collect all allocations for each target
        for allocation in system_allocations.values():
            for target_id, attention in allocation.items():
                consensus_allocation[target_id].append(attention)
        
        # Use median as consensus
        final_allocation = {}
        for target_id, attentions in consensus_allocation.items():
            attentions.sort()
            n = len(attentions)
            if n % 2 == 0:
                median = (attentions[n//2 - 1] + attentions[n//2]) / 2
            else:
                median = attentions[n//2]
            final_allocation[target_id] = median
        
        return final_allocation
    
    def _calculate_total_load(self) -> float:
        """Calculate total cognitive load across all systems"""
        if not self.attention_systems:
            return 0.0
        
        total_load = sum(system.context.cognitive_load 
                        for system in self.attention_systems.values())
        return min(1.0, total_load / len(self.attention_systems))

class AttentionManager:
    """Main manager for attention mechanisms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize attention mechanisms
        self.selective = SelectiveAttention(config.get('selective', {}))
        self.sustained = SustainedAttention(config.get('sustained', {}))
        self.divided = DividedAttention(config.get('divided', {}))
        self.executive = ExecutiveAttention(config.get('executive', {}))
        
        # Add systems to executive control
        self.executive.add_attention_system(self.selective)
        self.executive.add_attention_system(self.sustained)
        self.executive.add_attention_system(self.divided)
        
        # Current mode
        self.current_mode = AttentionType.EXECUTIVE
        self.targets: Dict[str, AttentionTarget] = {}
        
        # Performance tracking
        self.session_start = datetime.now()
        self.total_attention_time = 0.0
    
    def add_target(self, target_id: str, content: Any, priority: Priority, 
                  metadata: Optional[Dict[str, Any]] = None) -> AttentionTarget:
        """Add attention target"""
        target = AttentionTarget(
            id=target_id,
            content=content,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.targets[target_id] = target
        
        # Add to all attention systems
        self.selective.add_target(target)
        self.sustained.add_target(target)
        self.divided.add_target(target)
        self.executive.add_target(target)
        
        logger.info(f"Added attention target: {target_id}")
        return target
    
    def remove_target(self, target_id: str) -> bool:
        """Remove attention target"""
        if target_id not in self.targets:
            return False
        
        # Remove from all systems
        self.selective.remove_target(target_id)
        self.sustained.remove_target(target_id)
        self.divided.remove_target(target_id)
        self.executive.remove_target(target_id)
        
        del self.targets[target_id]
        
        logger.info(f"Removed attention target: {target_id}")
        return True
    
    async def allocate_attention(self, mode: Optional[AttentionType] = None) -> Dict[str, float]:
        """Allocate attention using specified or current mode"""
        if mode:
            self.current_mode = mode
        
        targets = list(self.targets.values())
        
        if self.current_mode == AttentionType.SELECTIVE:
            return await self.selective.allocate_attention(targets)
        elif self.current_mode == AttentionType.SUSTAINED:
            return await self.sustained.allocate_attention(targets)
        elif self.current_mode == AttentionType.DIVIDED:
            return await self.divided.allocate_attention(targets)
        elif self.current_mode == AttentionType.EXECUTIVE:
            return await self.executive.allocate_attention(targets)
        else:
            return {}
    
    async def update_attention(self, target_id: str, duration: float) -> bool:
        """Update attention for target"""
        if target_id not in self.targets:
            return False
        
        self.total_attention_time += duration
        
        # Update current mode system
        if self.current_mode == AttentionType.SELECTIVE:
            return await self.selective.update_attention(target_id, duration)
        elif self.current_mode == AttentionType.SUSTAINED:
            return await self.sustained.update_attention(target_id, duration)
        elif self.current_mode == AttentionType.DIVIDED:
            return await self.divided.update_attention(target_id, duration)
        elif self.current_mode == AttentionType.EXECUTIVE:
            return await self.executive.update_attention(target_id, duration)
        
        return False
    
    async def should_switch_attention(self, new_target_id: str) -> bool:
        """Determine if attention should switch to new target"""
        if new_target_id not in self.targets:
            return False
        
        new_target = self.targets[new_target_id]
        
        # Use current mode system
        if self.current_mode == AttentionType.SELECTIVE:
            return await self.selective.should_switch_attention(new_target)
        elif self.current_mode == AttentionType.SUSTAINED:
            return await self.sustained.should_switch_attention(new_target)
        elif self.current_mode == AttentionType.DIVIDED:
            return await self.divided.should_switch_attention(new_target)
        elif self.current_mode == AttentionType.EXECUTIVE:
            return await self.executive.should_switch_attention(new_target)
        
        return True
    
    def get_attention_state(self) -> Dict[str, Any]:
        """Get comprehensive attention state"""
        current_system = self._get_current_system()
        
        return {
            'current_mode': self.current_mode.value,
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'total_attention_time': self.total_attention_time,
            'target_count': len(self.targets),
            'current_system_state': current_system.get_attention_state() if current_system else {},
            'all_systems': {
                'selective': self.selective.get_attention_state(),
                'sustained': self.sustained.get_attention_state(),
                'divided': self.divided.get_attention_state(),
                'executive': self.executive.get_attention_state()
            }
        }
    
    def set_mode(self, mode: AttentionType):
        """Set attention mode"""
        self.current_mode = mode
        logger.info(f"Attention mode set to: {mode.value}")
    
    def _get_current_system(self) -> Optional[BaseAttentionMechanism]:
        """Get current attention system"""
        if self.current_mode == AttentionType.SELECTIVE:
            return self.selective
        elif self.current_mode == AttentionType.SUSTAINED:
            return self.sustained
        elif self.current_mode == AttentionType.DIVIDED:
            return self.divided
        elif self.current_mode == AttentionType.EXECUTIVE:
            return self.executive
        return None

# Default configuration
DEFAULT_ATTENTION_CONFIG = {
    'selective': {
        'focus_threshold': 0.6,
        'max_concurrent_targets': 3,
        'inhibition_strength': 0.8
    },
    'sustained': {
        'vigilance_threshold': 0.5,
        'fatigue_resistance': 0.7,
        'focus_boost_rate': 0.05,
        'distraction_resistance': 0.8
    },
    'divided': {
        'max_divisions': 4,
        'time_slice_duration': 1.0,
        'load_balancing': True,
        'interference_threshold': 0.8
    },
    'executive': {
        'control_threshold': 0.7,
        'coordination_strength': 0.8,
        'conflict_resolution': 'priority'
    }
}

def create_attention_manager(config: Optional[Dict[str, Any]] = None) -> AttentionManager:
    """Create attention manager with default or custom config"""
    if config is None:
        config = DEFAULT_ATTENTION_CONFIG
    
    return AttentionManager(config)