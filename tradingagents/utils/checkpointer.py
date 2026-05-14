import os
import json
import time
import pickle
import shutil
import logging
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Optional, Dict
from langgraph.checkpoint.memory import InMemorySaver
import threading

logger = logging.getLogger(__name__)

class FilePersistentDict(defaultdict):
    """A defaultdict that persists to a file using pickle."""
    def __init__(self, filename: str, *args, **kwargs):
        self.filename = filename
        super().__init__(*args, **kwargs)
        if os.path.exists(self.filename):
            self.load()

    _lock = threading.Lock()

    def sync(self) -> None:
        """Write dict to disk using a thread-safe direct write."""
        from pathlib import Path
        file_path = Path(self.filename)
        
        # Ensure the directory exists
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {file_path.parent}: {e}")
            return

        # Prepare the data
        data = {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in self.items()}

        # Thread-safe write
        with self._lock:
            try:
                with open(self.filename, "wb") as f:
                    pickle.dump(data, f)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except:
                        pass
            except Exception as e:
                print(f"CRITICAL SYNC ERROR: {e}")
                logger.error(f"Failed to sync to {self.filename}: {e}")

    def load(self) -> None:
        """Load dict from disk with robustness for pickle fragility."""
        if not os.path.exists(self.filename):
            return
        try:
            with open(self.filename, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.update(data)
                else:
                    logger.error(f"Data in {self.filename} is not a dict: {type(data)}")
        except (EOFError, pickle.UnpicklingError, AttributeError, ModuleNotFoundError) as e:
            logger.error(f"CRITICAL: Failed to load checkpointer from {self.filename}: {e}")
            logger.error("State might be corrupted. Attempting to preserve what's left...")
            # If load fails, we keep the empty defaultdict or the partially loaded state.
            # In a production environment, we might want to backup the corrupted file for analysis.
            backup_name = self.filename + f".corrupt.{int(time.time())}"
            shutil.copy2(self.filename, backup_name)
            logger.error(f"Corrupted state backed up to {backup_name}")

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Optional: sync on every write for high durability, or manually sync
        # self.sync() 

class TradingGraphCheckpointer(InMemorySaver):
    """
    A persistent checkpointer for LangGraph that saves to disk.
    Wraps InMemorySaver but uses FilePersistentDict for storage.
    """
    def __init__(self, checkpoint_dir: str = "data_dir/checkpoints", serde=None):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Use absolute paths to prevent Errno 2 issues in different environments
        storage_file = os.path.join(self.checkpoint_dir, "storage.pkl")
        writes_file = os.path.join(self.checkpoint_dir, "writes.pkl")
        blobs_file = os.path.join(self.checkpoint_dir, "blobs.pkl")

        # We need to be careful with the defaultdict factory lambdas as they aren't picklable
        # So we use a custom load/save logic in FilePersistentDict
        
        self.storage_dict = FilePersistentDict(storage_file, lambda: defaultdict(dict))
        self.writes_dict = FilePersistentDict(writes_file, dict)
        self.blobs_dict = FilePersistentDict(blobs_file, tuple)
        
        super().__init__(serde=serde)
        
        # Override the memory storage with our persistent ones
        self.storage = self.storage_dict
        self.writes = self.writes_dict
        self.blobs = self.blobs_dict

    def put(self, *args, **kwargs):
        res = super().put(*args, **kwargs)
        self.storage_dict.sync()
        self.blobs_dict.sync()
        return res

    def put_writes(self, *args, **kwargs):
        res = super().put_writes(*args, **kwargs)
        self.writes_dict.sync()
        return res
