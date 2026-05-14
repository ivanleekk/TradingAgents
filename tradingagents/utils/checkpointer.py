import os
import sqlite3
import pickle
import logging
from typing import Any, Optional, Iterator, Dict, List, Tuple
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

class TradingGraphCheckpointer(BaseCheckpointSaver):
    """
    A robust SQLite-based checkpointer for LangGraph.
    Stores checkpoints and writes in a relational database for atomicity and durability.
    """

    def __init__(
        self, 
        checkpoint_dir: str = "data_dir/checkpoints", 
        serde: Optional[SerializerProtocol] = None
    ):
        # Use JsonPlusSerializer as default, consistent with LangGraph defaults
        super().__init__(serde=serde or JsonPlusSerializer())
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.db_path = os.path.join(self.checkpoint_dir, "checkpoints.db")
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    parent_checkpoint_id TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    task_id TEXT,
                    idx INTEGER,
                    channel TEXT,
                    value BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
                """
            )
            conn.commit()

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Retrieve a checkpoint tuple for the given configuration."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        with sqlite3.connect(self.db_path) as conn:
            if checkpoint_id:
                query = (
                    "SELECT checkpoint_id, checkpoint, metadata, parent_checkpoint_id "
                    "FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
                )
                params = (thread_id, checkpoint_ns, checkpoint_id)
            else:
                # Get the latest checkpoint if no ID is provided
                query = (
                    "SELECT checkpoint_id, checkpoint, metadata, parent_checkpoint_id "
                    "FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? "
                    "ORDER BY checkpoint_id DESC LIMIT 1"
                )
                params = (thread_id, checkpoint_ns)

            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            if not row:
                return None

            res_id, checkpoint_blob, metadata_blob, parent_id = row
            
            # Load writes for this checkpoint
            writes_cursor = conn.execute(
                "SELECT task_id, channel, value FROM writes "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_ns, res_id)
            )
            pending_writes = [
                (task_id, channel, self.serde.loads(value_blob))
                for task_id, channel, value_blob in writes_cursor
            ]

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": res_id,
                    }
                },
                checkpoint=self.serde.loads(checkpoint_blob),
                metadata=self.serde.loads(metadata_blob),
                parent_config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                } if parent_id else None,
                pending_writes=pending_writes
            )

    def list(self, config: dict, *, filter: Optional[dict] = None, before: Optional[dict] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the criteria."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        query = "SELECT checkpoint_id, checkpoint, metadata, parent_checkpoint_id FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?"
        params = [thread_id, checkpoint_ns]
        
        if before:
            query += " AND checkpoint_id < ?"
            params.append(before["configurable"]["checkpoint_id"])
            
        query += " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor:
                res_id, checkpoint_blob, metadata_blob, parent_id = row
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": res_id,
                        }
                    },
                    checkpoint=self.serde.loads(checkpoint_blob),
                    metadata=self.serde.loads(metadata_blob),
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_id,
                        }
                    } if parent_id else None
                )

    def put(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_releases: Any) -> dict:
        """Store a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    self.serde.dumps(checkpoint),
                    self.serde.dumps(metadata),
                    parent_id
                )
            )
            conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(self, config: dict, writes: List[tuple[str, Any]], task_id: str) -> None:
        """Store writes for a specific task."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with sqlite3.connect(self.db_path) as conn:
            for idx, (channel, value) in enumerate(writes):
                conn.execute(
                    "INSERT OR REPLACE INTO writes VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx,
                        channel,
                        self.serde.dumps(value)
                    )
                )
            conn.commit()
