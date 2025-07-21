"""Helper classes for backend testing without mocks."""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List


class TestableDateTime:
    """A testable datetime class that can be controlled for testing."""
    
    def __init__(self, base_time: float = None):
        self._base_time = base_time or datetime.now().timestamp()
        self._current_offset = 0
        
    def now(self):
        """Return current test time."""
        return datetime.fromtimestamp(self._base_time + self._current_offset)
        
    def advance(self, seconds: float):
        """Advance the test time by given seconds."""
        self._current_offset += seconds
        
    def set_time(self, timestamp: float):
        """Set the current test time to a specific timestamp."""
        self._base_time = timestamp
        self._current_offset = 0


class TestableRedisClient:
    """A testable Redis client for testing without real Redis."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._sorted_sets: Dict[str, List[tuple]] = {}
        self._connected = True
        self.call_history = []
        
    async def hset(self, key: str, field: str, value: str):
        """Set hash field."""
        self.call_history.append(('hset', key, field, value))
        if key not in self._data:
            self._data[key] = {}
        self._data[key][field] = value
        
    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field."""
        self.call_history.append(('hget', key, field))
        if key in self._data and isinstance(self._data[key], dict):
            return self._data[key].get(field)
        return None
        
    async def zadd(self, key: str, mapping: Dict[str, float]):
        """Add to sorted set."""
        self.call_history.append(('zadd', key, mapping))
        if key not in self._sorted_sets:
            self._sorted_sets[key] = []
        for member, score in mapping.items():
            self._sorted_sets[key].append((score, member))
        self._sorted_sets[key].sort(key=lambda x: x[0])
        
    async def zrange(self, key: str, start: int, stop: int, withscores: bool = False):
        """Get range from sorted set."""
        self.call_history.append(('zrange', key, start, stop, withscores))
        if key not in self._sorted_sets:
            return []
        items = self._sorted_sets[key][start:stop+1 if stop >= 0 else None]
        if withscores:
            return [(member, score) for score, member in items]
        return [member for score, member in items]
        
    async def delete(self, *keys):
        """Delete keys."""
        self.call_history.append(('delete', *keys))
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                deleted += 1
            if key in self._sorted_sets:
                del self._sorted_sets[key]
                deleted += 1
        return deleted
        
    async def ping(self):
        """Check connection."""
        self.call_history.append(('ping',))
        if not self._connected:
            raise ConnectionError("Redis connection failed")
        return True
        
    def set_connected(self, connected: bool):
        """Set connection state for testing."""
        self._connected = connected


class TestableAsyncpgPool:
    """A testable asyncpg pool for testing without real PostgreSQL."""
    
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._connected = True
        self.call_history = []
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def fetchrow(self, query: str, *args):
        """Fetch single row."""
        self.call_history.append(('fetchrow', query, args))
        
        if "SELECT * FROM" in query and "WHERE checkpoint_id = $1" in query:
            checkpoint_id = args[0]
            for table_data in self._data.values():
                if checkpoint_id in table_data:
                    return table_data[checkpoint_id]
        return None
        
    async def fetch(self, query: str, *args):
        """Fetch multiple rows."""
        self.call_history.append(('fetch', query, args))
        
        if "SELECT * FROM" in query and "ORDER BY timestamp DESC" in query:
            results = []
            for table_data in self._data.values():
                results.extend(table_data.values())
            results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Apply LIMIT if present
            if "LIMIT $" in query:
                limit_idx = query.count("$") - 1
                limit = args[limit_idx] if limit_idx < len(args) else None
                if limit:
                    results = results[:limit]
            
            return results
        return []
        
    async def execute(self, query: str, *args):
        """Execute query."""
        self.call_history.append(('execute', query, args))
        
        if "INSERT INTO" in query:
            # Parse table name
            table_name = query.split("INSERT INTO")[1].split("(")[0].strip()
            if table_name not in self._data:
                self._data[table_name] = {}
                
            # Assume standard checkpoint insert
            checkpoint_id, execution_id, state, metadata, timestamp = args
            self._data[table_name][checkpoint_id] = {
                'checkpoint_id': checkpoint_id,
                'execution_id': execution_id,
                'state': state,
                'metadata': metadata,
                'timestamp': timestamp
            }
            
        elif "DELETE FROM" in query and "WHERE checkpoint_id = $1" in query:
            checkpoint_id = args[0]
            for table_data in self._data.values():
                if checkpoint_id in table_data:
                    del table_data[checkpoint_id]
                    return "DELETE 1"
            return "DELETE 0"
            
        elif "CREATE TABLE IF NOT EXISTS" in query:
            # Table creation - just track it
            pass
            
        return "OK"
        
    def set_connected(self, connected: bool):
        """Set connection state for testing."""
        self._connected = connected


class TestableAsyncpgConnection:
    """A testable asyncpg connection."""
    
    def __init__(self):
        self.pool = TestableAsyncpgPool()
        
    async def create_pool(self, *args, **kwargs):
        """Create connection pool."""
        return self.pool