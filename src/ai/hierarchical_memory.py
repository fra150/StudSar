"""Hierarchical Memory Module for StudSar V4

Implements multi-level memory structure (short-term, working, long-term)
with automatic consolidation and transfer mechanisms.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import json
import math
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MemoryLevel(Enum):
    """Memory hierarchy levels"""
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"

class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation"""
    FREQUENCY_BASED = "frequency"
    RECENCY_BASED = "recency"
    IMPORTANCE_BASED = "importance"
    HYBRID = "hybrid"

@dataclass
class MemoryTrace:
    """Individual memory trace with temporal and usage information"""
    id: str
    content: str
    embeddings: List[float]
    level: MemoryLevel
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance_score: float = 0.0
    decay_rate: float = 0.1
    consolidation_strength: float = 0.0
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """Update access information"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self._update_importance()
    
    def _update_importance(self):
        """Update importance score based on access patterns"""
        # Recency factor (exponential decay)
        time_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600  # hours
        recency_factor = math.exp(-self.decay_rate * time_since_access)
        
        # Frequency factor (logarithmic)
        frequency_factor = math.log(1 + self.access_count)
        
        # Combined importance
        self.importance_score = recency_factor * frequency_factor
    
    def get_activation_level(self) -> float:
        """Get current activation level (0.0 to 1.0)"""
        self._update_importance()
        return min(1.0, self.importance_score / 10.0)  # Normalize to 0-1

@dataclass
class MemoryCapacity:
    """Memory capacity configuration for each level"""
    max_items: int
    retention_time: timedelta
    consolidation_threshold: float
    decay_rate: float

class BaseMemoryLevel(ABC):
    """Abstract base class for memory levels"""
    
    def __init__(self, capacity: MemoryCapacity, level: MemoryLevel):
        self.capacity = capacity
        self.level = level
        self.traces: Dict[str, MemoryTrace] = {}
        self.access_history: deque = deque(maxlen=1000)
    
    @abstractmethod
    async def add_trace(self, trace: MemoryTrace) -> bool:
        """Add memory trace to this level"""
        pass
    
    @abstractmethod
    async def retrieve_traces(self, query_embeddings: List[float], limit: int = 10) -> List[MemoryTrace]:
        """Retrieve relevant traces"""
        pass
    
    @abstractmethod
    async def consolidate(self) -> List[MemoryTrace]:
        """Consolidate memories and return traces to promote"""
        pass
    
    def get_trace(self, trace_id: str) -> Optional[MemoryTrace]:
        """Get specific trace by ID"""
        trace = self.traces.get(trace_id)
        if trace:
            trace.update_access()
            self.access_history.append((trace_id, datetime.now()))
        return trace
    
    def remove_trace(self, trace_id: str) -> Optional[MemoryTrace]:
        """Remove trace from this level"""
        return self.traces.pop(trace_id, None)
    
    def is_full(self) -> bool:
        """Check if memory level is at capacity"""
        return len(self.traces) >= self.capacity.max_items
    
    def get_size(self) -> int:
        """Get current number of traces"""
        return len(self.traces)
    
    def _calculate_similarity(self, embeddings1: List[float], embeddings2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if not embeddings1 or not embeddings2 or len(embeddings1) != len(embeddings2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(embeddings1, embeddings2))
        norm1 = math.sqrt(sum(a * a for a in embeddings1))
        norm2 = math.sqrt(sum(b * b for b in embeddings2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class ShortTermMemory(BaseMemoryLevel):
    """Short-term memory with rapid decay and limited capacity"""
    
    def __init__(self, capacity: MemoryCapacity):
        super().__init__(capacity, MemoryLevel.SHORT_TERM)
        self.buffer = deque(maxlen=capacity.max_items)
    
    async def add_trace(self, trace: MemoryTrace) -> bool:
        """Add trace to short-term memory"""
        trace.level = MemoryLevel.SHORT_TERM
        trace.decay_rate = self.capacity.decay_rate
        
        # If at capacity, remove oldest trace
        if self.is_full():
            oldest_id = self.buffer[0]
            self.remove_trace(oldest_id)
            self.buffer.popleft()
        
        self.traces[trace.id] = trace
        self.buffer.append(trace.id)
        
        logger.debug(f"Added trace {trace.id} to short-term memory")
        return True
    
    async def retrieve_traces(self, query_embeddings: List[float], limit: int = 10) -> List[MemoryTrace]:
        """Retrieve traces by similarity"""
        if not query_embeddings:
            return list(self.traces.values())[:limit]
        
        # Calculate similarities
        similarities = []
        for trace in self.traces.values():
            similarity = self._calculate_similarity(query_embeddings, trace.embeddings)
            similarities.append((trace, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [trace for trace, sim in similarities[:limit]]
    
    async def consolidate(self) -> List[MemoryTrace]:
        """Consolidate short-term memories"""
        to_promote = []
        to_remove = []
        current_time = datetime.now()
        
        for trace in self.traces.values():
            # Check if trace should be promoted to working memory
            if (trace.access_count >= 2 or 
                trace.get_activation_level() > self.capacity.consolidation_threshold):
                to_promote.append(trace)
                to_remove.append(trace.id)
            
            # Check if trace should decay
            elif (current_time - trace.last_accessed) > self.capacity.retention_time:
                to_remove.append(trace.id)
        
        # Remove traces
        for trace_id in to_remove:
            self.remove_trace(trace_id)
            if trace_id in self.buffer:
                buffer_list = list(self.buffer)
                buffer_list.remove(trace_id)
                self.buffer = deque(buffer_list, maxlen=self.capacity.max_items)
        
        logger.info(f"Short-term consolidation: {len(to_promote)} promoted, {len(to_remove)} removed")
        return to_promote

class WorkingMemory(BaseMemoryLevel):
    """Working memory with active maintenance and manipulation"""
    
    def __init__(self, capacity: MemoryCapacity):
        super().__init__(capacity, MemoryLevel.WORKING)
        self.active_set: Set[str] = set()  # Currently active traces
        self.rehearsal_queue: deque = deque()
    
    async def add_trace(self, trace: MemoryTrace) -> bool:
        """Add trace to working memory"""
        trace.level = MemoryLevel.WORKING
        trace.decay_rate = self.capacity.decay_rate
        
        # If at capacity, remove least important trace
        if self.is_full():
            await self._make_space()
        
        self.traces[trace.id] = trace
        self.active_set.add(trace.id)
        self.rehearsal_queue.append(trace.id)
        
        logger.debug(f"Added trace {trace.id} to working memory")
        return True
    
    async def retrieve_traces(self, query_embeddings: List[float], limit: int = 10) -> List[MemoryTrace]:
        """Retrieve traces with working memory bias"""
        if not query_embeddings:
            # Return active traces first
            active_traces = [self.traces[tid] for tid in self.active_set if tid in self.traces]
            other_traces = [t for t in self.traces.values() if t.id not in self.active_set]
            return (active_traces + other_traces)[:limit]
        
        # Calculate similarities with activation boost
        similarities = []
        for trace in self.traces.values():
            similarity = self._calculate_similarity(query_embeddings, trace.embeddings)
            
            # Boost similarity for active traces
            if trace.id in self.active_set:
                similarity *= 1.5
            
            similarities.append((trace, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [trace for trace, sim in similarities[:limit]]
    
    async def consolidate(self) -> List[MemoryTrace]:
        """Consolidate working memories"""
        to_promote = []
        to_remove = []
        current_time = datetime.now()
        
        # Rehearsal mechanism - maintain active traces
        await self._rehearse_active_traces()
        
        for trace in self.traces.values():
            # Check if trace should be promoted to long-term memory
            if (trace.access_count >= 5 and 
                trace.get_activation_level() > self.capacity.consolidation_threshold):
                to_promote.append(trace)
                to_remove.append(trace.id)
            
            # Check if trace should decay
            elif (current_time - trace.last_accessed) > self.capacity.retention_time:
                to_remove.append(trace.id)
        
        # Remove traces
        for trace_id in to_remove:
            self.remove_trace(trace_id)
            self.active_set.discard(trace_id)
            if trace_id in self.rehearsal_queue:
                queue_list = list(self.rehearsal_queue)
                queue_list.remove(trace_id)
                self.rehearsal_queue = deque(queue_list)
        
        logger.info(f"Working memory consolidation: {len(to_promote)} promoted, {len(to_remove)} removed")
        return to_promote
    
    async def _make_space(self):
        """Remove least important trace to make space"""
        if not self.traces:
            return
        
        # Find least important trace not in active set
        inactive_traces = [(tid, trace) for tid, trace in self.traces.items() 
                          if tid not in self.active_set]
        
        if inactive_traces:
            # Remove least important inactive trace
            least_important = min(inactive_traces, key=lambda x: x[1].get_activation_level())
            self.remove_trace(least_important[0])
        else:
            # Remove least important active trace
            least_important = min(self.traces.items(), key=lambda x: x[1].get_activation_level())
            self.remove_trace(least_important[0])
            self.active_set.discard(least_important[0])
    
    async def _rehearse_active_traces(self):
        """Rehearse active traces to maintain them"""
        # Rotate through rehearsal queue
        if self.rehearsal_queue:
            trace_id = self.rehearsal_queue.popleft()
            if trace_id in self.traces:
                self.traces[trace_id].update_access()
                self.rehearsal_queue.append(trace_id)

class LongTermMemory(BaseMemoryLevel):
    """Long-term memory with permanent storage and slow decay"""
    
    def __init__(self, capacity: MemoryCapacity):
        super().__init__(capacity, MemoryLevel.LONG_TERM)
        self.semantic_clusters: Dict[str, List[str]] = defaultdict(list)
        self.episodic_timeline: List[Tuple[datetime, str]] = []
    
    async def add_trace(self, trace: MemoryTrace) -> bool:
        """Add trace to long-term memory"""
        trace.level = MemoryLevel.LONG_TERM
        trace.decay_rate = self.capacity.decay_rate
        trace.consolidation_strength = 1.0  # Fully consolidated
        
        # If at capacity, remove oldest least important trace
        if self.is_full():
            await self._make_space()
        
        self.traces[trace.id] = trace
        self.episodic_timeline.append((trace.created_at, trace.id))
        self.episodic_timeline.sort(key=lambda x: x[0])  # Keep sorted by time
        
        # Add to semantic clusters
        await self._add_to_semantic_cluster(trace)
        
        logger.debug(f"Added trace {trace.id} to long-term memory")
        return True
    
    async def retrieve_traces(self, query_embeddings: List[float], limit: int = 10) -> List[MemoryTrace]:
        """Retrieve traces using semantic and episodic cues"""
        if not query_embeddings:
            return list(self.traces.values())[:limit]
        
        # Calculate similarities
        similarities = []
        for trace in self.traces.values():
            similarity = self._calculate_similarity(query_embeddings, trace.embeddings)
            
            # Boost similarity based on consolidation strength
            similarity *= trace.consolidation_strength
            
            similarities.append((trace, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [trace for trace, sim in similarities[:limit]]
    
    async def consolidate(self) -> List[MemoryTrace]:
        """Consolidate long-term memories (reorganization)"""
        # Long-term memory doesn't promote to higher levels
        # Instead, it reorganizes and strengthens associations
        
        await self._reorganize_semantic_clusters()
        await self._strengthen_associations()
        
        # Remove only very old, unused traces if at capacity
        to_remove = []
        if self.is_full():
            current_time = datetime.now()
            for trace in self.traces.values():
                if ((current_time - trace.last_accessed) > timedelta(days=365) and 
                    trace.access_count < 2):
                    to_remove.append(trace.id)
        
        for trace_id in to_remove:
            self.remove_trace(trace_id)
            # Remove from episodic timeline
            self.episodic_timeline = [(t, tid) for t, tid in self.episodic_timeline if tid != trace_id]
        
        logger.info(f"Long-term memory consolidation: {len(to_remove)} removed")
        return []  # No promotion from long-term
    
    async def _make_space(self):
        """Remove least important trace to make space"""
        if not self.traces:
            return
        
        # Find oldest, least accessed trace
        oldest_unused = min(self.traces.items(), 
                           key=lambda x: (x[1].access_count, x[1].last_accessed))
        self.remove_trace(oldest_unused[0])
    
    async def _add_to_semantic_cluster(self, trace: MemoryTrace):
        """Add trace to appropriate semantic cluster"""
        # Simple clustering based on content keywords
        keywords = trace.metadata.get('keywords', [])
        for keyword in keywords[:3]:  # Use top 3 keywords
            self.semantic_clusters[keyword].append(trace.id)
    
    async def _reorganize_semantic_clusters(self):
        """Reorganize semantic clusters based on usage patterns"""
        # Remove traces that no longer exist
        for keyword in list(self.semantic_clusters.keys()):
            self.semantic_clusters[keyword] = [
                tid for tid in self.semantic_clusters[keyword] 
                if tid in self.traces
            ]
            # Remove empty clusters
            if not self.semantic_clusters[keyword]:
                del self.semantic_clusters[keyword]
    
    async def _strengthen_associations(self):
        """Strengthen associations between frequently co-accessed traces"""
        # Analyze access patterns to strengthen associations
        access_pairs = defaultdict(int)
        
        # Count co-occurrences in recent access history
        recent_accesses = list(self.access_history)[-100:]  # Last 100 accesses
        for i in range(len(recent_accesses) - 1):
            trace1_id = recent_accesses[i][0]
            trace2_id = recent_accesses[i + 1][0]
            if trace1_id != trace2_id:
                pair = tuple(sorted([trace1_id, trace2_id]))
                access_pairs[pair] += 1
        
        # Strengthen associations for frequently co-accessed traces
        for (trace1_id, trace2_id), count in access_pairs.items():
            if count >= 3:  # Threshold for association
                if trace1_id in self.traces and trace2_id in self.traces:
                    trace1 = self.traces[trace1_id]
                    trace2 = self.traces[trace2_id]
                    
                    if trace2_id not in trace1.associations:
                        trace1.associations.append(trace2_id)
                    if trace1_id not in trace2.associations:
                        trace2.associations.append(trace1_id)

class HierarchicalMemoryManager:
    """Main manager for hierarchical memory system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize memory levels
        self.short_term = ShortTermMemory(MemoryCapacity(
            max_items=config.get('short_term_capacity', 20),
            retention_time=timedelta(minutes=config.get('short_term_retention_minutes', 15)),
            consolidation_threshold=config.get('short_term_threshold', 0.3),
            decay_rate=config.get('short_term_decay', 0.2)
        ))
        
        self.working = WorkingMemory(MemoryCapacity(
            max_items=config.get('working_capacity', 50),
            retention_time=timedelta(hours=config.get('working_retention_hours', 2)),
            consolidation_threshold=config.get('working_threshold', 0.6),
            decay_rate=config.get('working_decay', 0.1)
        ))
        
        self.long_term = LongTermMemory(MemoryCapacity(
            max_items=config.get('long_term_capacity', 10000),
            retention_time=timedelta(days=config.get('long_term_retention_days', 365)),
            consolidation_threshold=config.get('long_term_threshold', 0.8),
            decay_rate=config.get('long_term_decay', 0.01)
        ))
        
        # Consolidation settings
        self.consolidation_interval = config.get('consolidation_interval_minutes', 30)
        self.last_consolidation = datetime.now()
        
        # Start background consolidation
        self._consolidation_task = None
        self.start_consolidation()
    
    async def add_memory(self, content: str, embeddings: List[float], 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add new memory to the hierarchy"""
        # Create memory trace
        trace_id = f"trace_{int(time.time() * 1000000)}"
        trace = MemoryTrace(
            id=trace_id,
            content=content,
            embeddings=embeddings,
            level=MemoryLevel.SHORT_TERM,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            metadata=metadata or {}
        )
        
        # Add to short-term memory
        await self.short_term.add_trace(trace)
        
        logger.info(f"Added memory trace {trace_id} to hierarchy")
        return trace_id
    
    async def retrieve_memories(self, query_embeddings: List[float], 
                               levels: Optional[List[MemoryLevel]] = None,
                               limit: int = 10) -> List[MemoryTrace]:
        """Retrieve memories from specified levels"""
        if levels is None:
            levels = [MemoryLevel.SHORT_TERM, MemoryLevel.WORKING, MemoryLevel.LONG_TERM]
        
        all_traces = []
        
        # Retrieve from each level
        for level in levels:
            if level == MemoryLevel.SHORT_TERM:
                traces = await self.short_term.retrieve_traces(query_embeddings, limit)
            elif level == MemoryLevel.WORKING:
                traces = await self.working.retrieve_traces(query_embeddings, limit)
            elif level == MemoryLevel.LONG_TERM:
                traces = await self.long_term.retrieve_traces(query_embeddings, limit)
            else:
                continue
            
            all_traces.extend(traces)
        
        # Remove duplicates and sort by activation level
        unique_traces = {trace.id: trace for trace in all_traces}
        sorted_traces = sorted(unique_traces.values(), 
                             key=lambda t: t.get_activation_level(), 
                             reverse=True)
        
        return sorted_traces[:limit]
    
    async def get_memory(self, trace_id: str) -> Optional[MemoryTrace]:
        """Get specific memory by ID from any level"""
        # Search in all levels
        trace = self.short_term.get_trace(trace_id)
        if trace:
            return trace
        
        trace = self.working.get_trace(trace_id)
        if trace:
            return trace
        
        trace = self.long_term.get_trace(trace_id)
        if trace:
            return trace
        
        return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory hierarchy"""
        return {
            'short_term': {
                'size': self.short_term.get_size(),
                'capacity': self.short_term.capacity.max_items,
                'utilization': self.short_term.get_size() / self.short_term.capacity.max_items
            },
            'working': {
                'size': self.working.get_size(),
                'capacity': self.working.capacity.max_items,
                'utilization': self.working.get_size() / self.working.capacity.max_items,
                'active_traces': len(self.working.active_set)
            },
            'long_term': {
                'size': self.long_term.get_size(),
                'capacity': self.long_term.capacity.max_items,
                'utilization': self.long_term.get_size() / self.long_term.capacity.max_items,
                'semantic_clusters': len(self.long_term.semantic_clusters)
            },
            'total_memories': (self.short_term.get_size() + 
                             self.working.get_size() + 
                             self.long_term.get_size()),
            'last_consolidation': self.last_consolidation.isoformat()
        }
    
    async def consolidate_memories(self):
        """Perform memory consolidation across all levels"""
        logger.info("Starting memory consolidation")
        
        # Consolidate short-term -> working
        st_promoted = await self.short_term.consolidate()
        for trace in st_promoted:
            await self.working.add_trace(trace)
        
        # Consolidate working -> long-term
        wm_promoted = await self.working.consolidate()
        for trace in wm_promoted:
            await self.long_term.add_trace(trace)
        
        # Consolidate long-term (reorganization)
        await self.long_term.consolidate()
        
        self.last_consolidation = datetime.now()
        
        logger.info(f"Consolidation complete: {len(st_promoted)} ST->WM, {len(wm_promoted)} WM->LT")
    
    def start_consolidation(self):
        """Start background consolidation task"""
        if self._consolidation_task is None or self._consolidation_task.done():
            self._consolidation_task = asyncio.create_task(self._consolidation_loop())
    
    def stop_consolidation(self):
        """Stop background consolidation task"""
        if self._consolidation_task and not self._consolidation_task.done():
            self._consolidation_task.cancel()
    
    async def _consolidation_loop(self):
        """Background consolidation loop"""
        while True:
            try:
                await asyncio.sleep(self.consolidation_interval * 60)  # Convert to seconds
                await self.consolidate_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

# Default configuration
DEFAULT_HIERARCHICAL_CONFIG = {
    'short_term_capacity': 20,
    'short_term_retention_minutes': 15,
    'short_term_threshold': 0.3,
    'short_term_decay': 0.2,
    
    'working_capacity': 50,
    'working_retention_hours': 2,
    'working_threshold': 0.6,
    'working_decay': 0.1,
    
    'long_term_capacity': 10000,
    'long_term_retention_days': 365,
    'long_term_threshold': 0.8,
    'long_term_decay': 0.01,
    
    'consolidation_interval_minutes': 30
}

def create_hierarchical_memory(config: Optional[Dict[str, Any]] = None) -> HierarchicalMemoryManager:
    """Create hierarchical memory manager with default or custom config"""
    if config is None:
        config = DEFAULT_HIERARCHICAL_CONFIG
    
    return HierarchicalMemoryManager(config)