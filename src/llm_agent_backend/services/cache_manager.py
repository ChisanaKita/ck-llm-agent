"""
Advanced Cache Manager for Response Caching and Performance Optimization.

This module provides comprehensive caching capabilities including in-memory caching,
cache invalidation strategies, persistence management, and performance monitoring.
"""

import asyncio
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

from ..models.api import ChatCompletionRequest, ChatCompletionResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    expires_at: Optional[datetime] = None
    size_bytes: int = 0
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.size_bytes == 0:
            self.size_bytes = len(str(self.value))
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "size_bytes": self.size_bytes,
            "tags": list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
            size_bytes=data["size_bytes"],
            tags=set(data.get("tags", []))
        )


class CacheStats:
    """Cache statistics tracking."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.total_size_bytes = 0
        self.start_time = datetime.utcnow()
    
    def record_hit(self) -> None:
        """Record cache hit."""
        self.hits += 1
    
    def record_miss(self) -> None:
        """Record cache miss."""
        self.misses += 1
    
    def record_eviction(self) -> None:
        """Record cache eviction."""
        self.evictions += 1
    
    def record_expiration(self) -> None:
        """Record cache expiration."""
        self.expirations += 1
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_uptime(self) -> timedelta:
        """Get cache uptime."""
        return datetime.utcnow() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.get_hit_rate(),
            "total_size_bytes": self.total_size_bytes,
            "uptime_seconds": self.get_uptime().total_seconds()
        }


class ResponseCacheManager:
    """
    Advanced response cache manager with multiple invalidation strategies.
    
    Provides in-memory caching with TTL, LRU, and other eviction policies,
    plus optional persistence for cache warming and recovery.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None,
        max_memory_mb: int = 100,
        cleanup_interval: int = 300
    ):
        """
        Initialize the cache manager.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
            strategy: Cache eviction strategy
            enable_persistence: Enable cache persistence
            persistence_path: Path for cache persistence
            max_memory_mb: Maximum memory usage in MB
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_persistence = enable_persistence
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU
        self._frequency_counter: Dict[str, int] = {}  # For LFU
        
        # Statistics
        self.stats = CacheStats()
        
        # Persistence
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(
            f"ResponseCacheManager initialized: max_size={max_size}, "
            f"ttl={default_ttl}s, strategy={strategy.value}"
        )
    
    async def initialize(self) -> None:
        """Initialize the cache manager."""
        # Load persisted cache if enabled
        if self.enable_persistence and self.persistence_path:
            await self._load_cache()
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("ResponseCacheManager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the cache manager."""
        self._shutdown_event.set()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Persist cache if enabled
        if self.enable_persistence:
            await self._save_cache()
        
        logger.info("ResponseCacheManager shutdown complete")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)
        
        if entry is None:
            self.stats.record_miss()
            return None
        
        if entry.is_expired():
            await self._remove_entry(key)
            self.stats.record_miss()
            self.stats.record_expiration()
            return None
        
        # Update access metadata
        entry.touch()
        self._update_access_order(key)
        
        self.stats.record_hit()
        return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            tags: Optional tags for cache invalidation
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            expires_at=expires_at,
            tags=tags or set()
        )
        
        # Check if we need to evict entries
        await self._ensure_capacity()
        
        # Store entry
        self._cache[key] = entry
        self._update_access_order(key)
        self.stats.total_size_bytes += entry.size_bytes
        
        logger.debug(f"Cached entry: key={key}, ttl={ttl}s, size={entry.size_bytes} bytes")
    
    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted, False if not found
        """
        if key in self._cache:
            await self._remove_entry(key)
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        self._frequency_counter.clear()
        self.stats.total_size_bytes = 0
        
        logger.info("Cache cleared")
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """
        Invalidate cache entries by tags.
        
        Args:
            tags: Tags to invalidate
            
        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []
        
        for key, entry in self._cache.items():
            if entry.tags.intersection(tags):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            await self._remove_entry(key)
        
        logger.info(f"Invalidated {len(keys_to_remove)} entries by tags: {tags}")
        return len(keys_to_remove)
    
    async def get_chat_response(self, request: ChatCompletionRequest) -> Optional[ChatCompletionResponse]:
        """
        Get cached chat response.
        
        Args:
            request: Chat completion request
            
        Returns:
            Cached response or None
        """
        cache_key = self._generate_chat_cache_key(request)
        return await self.get(cache_key)
    
    async def cache_chat_response(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache chat response.
        
        Args:
            request: Chat completion request
            response: Chat completion response
            ttl: Time-to-live in seconds
        """
        cache_key = self._generate_chat_cache_key(request)
        tags = {
            "chat_response",
            f"model:{request.model}",
            f"temperature:{request.temperature}"
        }
        
        await self.set(cache_key, response, ttl=ttl, tags=tags)
    
    def _generate_chat_cache_key(self, request: ChatCompletionRequest) -> str:
        """Generate cache key for chat request."""
        # Create deterministic key from request parameters
        key_data = {
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": getattr(request, 'max_tokens', None),
            "messages": [
                {"role": msg.role, "content": msg.content[:200]}  # Truncate for key
                for msg in request.messages[-3:]  # Use last 3 messages
            ]
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    async def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed capacity limits."""
        # Check entry count limit
        while len(self._cache) >= self.max_size:
            await self._evict_entry()
        
        # Check memory limit
        while self.stats.total_size_bytes > self.max_memory_bytes:
            await self._evict_entry()
    
    async def _evict_entry(self) -> None:
        """Evict an entry based on the configured strategy."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            key_to_evict = self._access_order[0] if self._access_order else next(iter(self._cache))
        elif self.strategy == CacheStrategy.LFU:
            key_to_evict = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.strategy == CacheStrategy.FIFO:
            key_to_evict = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        else:  # Default to LRU
            key_to_evict = self._access_order[0] if self._access_order else next(iter(self._cache))
        
        await self._remove_entry(key_to_evict)
        self.stats.record_eviction()
    
    async def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and update metadata."""
        entry = self._cache.pop(key, None)
        if entry:
            self.stats.total_size_bytes -= entry.size_bytes
            
            if key in self._access_order:
                self._access_order.remove(key)
            
            if key in self._frequency_counter:
                del self._frequency_counter[key]
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU strategy."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # Update frequency counter for LFU
        self._frequency_counter[key] = self._frequency_counter.get(key, 0) + 1
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_expired_entries()
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.cleanup_interval
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_expired_entries(self) -> None:
        """Clean up expired cache entries."""
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_entry(key)
            self.stats.record_expiration()
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    async def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self.persistence_path:
            return
        
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                "entries": [entry.to_dict() for entry in self._cache.values()],
                "stats": self.stats.to_dict(),
                "saved_at": datetime.utcnow().isoformat()
            }
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cache saved to {self.persistence_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    async def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        try:
            with open(self.persistence_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restore entries
            for entry_data in cache_data.get("entries", []):
                entry = CacheEntry.from_dict(entry_data)
                
                # Skip expired entries
                if entry.is_expired():
                    continue
                
                self._cache[entry.key] = entry
                self._update_access_order(entry.key)
                self.stats.total_size_bytes += entry.size_bytes
            
            logger.info(f"Cache loaded from {self.persistence_path}: {len(self._cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats_dict = self.stats.to_dict()
        stats_dict.update({
            "total_entries": len(self._cache),
            "max_size": self.max_size,
            "strategy": self.strategy.value,
            "memory_usage_mb": self.stats.total_size_bytes / (1024 * 1024),
            "memory_limit_mb": self.max_memory_bytes / (1024 * 1024)
        })
        return stats_dict
    
    async def warm_cache(self, entries: List[Tuple[str, Any, Optional[int]]]) -> int:
        """
        Warm cache with predefined entries.
        
        Args:
            entries: List of (key, value, ttl) tuples
            
        Returns:
            Number of entries added
        """
        added_count = 0
        
        for key, value, ttl in entries:
            try:
                await self.set(key, value, ttl=ttl)
                added_count += 1
            except Exception as e:
                logger.error(f"Failed to warm cache entry {key}: {e}")
        
        logger.info(f"Cache warmed with {added_count} entries")
        return added_count