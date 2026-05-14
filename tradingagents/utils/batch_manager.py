import os
import json
import time
from typing import List, Dict, Any, Optional, Union, ClassVar
import contextvars
from openai import OpenAI
from pathlib import Path
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

class BatchCaptureException(Exception):
    """Exception raised when an LLM prompt is captured for batch processing."""
    pass

class OpenAIBatchManager:
    """Manages OpenAI Batch API lifecycle."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), 
                             base_url=base_url or "https://api.openai.com/v1")
        self.requests = []
        self.results = {} # custom_id -> content

    def add_request(self, custom_id: str, model: str, messages: List[Dict[str, str]], **kwargs):
        """Add a request to the current batch."""
        # Convert tools to OpenAI format if present
        if "tools" in kwargs:
            kwargs["tools"] = [convert_to_openai_tool(t) for t in kwargs["tools"]]
            
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                **kwargs
            }
        }
        self.requests.append(request)
        
        # Incremental saving to disk
        if hasattr(self, 'current_batch_file') and self.current_batch_file:
            os.makedirs(os.path.dirname(self.current_batch_file), exist_ok=True)
            with open(self.current_batch_file, "a") as f:
                f.write(json.dumps(request) + "\n")

    def submit_batch(self, batch_file_path: str = "batches/batch_requests.jsonl") -> str:
        """Submit the collected requests as a batch job."""
        if not self.requests:
            raise ValueError("No requests to submit.")

        # Write to JSONL
        with open(batch_file_path, "w") as f:
            for req in self.requests:
                f.write(json.dumps(req) + "\n")

        # Upload file
        batch_input_file = self.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )

        # Create batch
        batch_job = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"TradingAgents Batch Run {time.strftime('%Y-%m-%d %H:%M:%S')}"
            }
        )
        
        # Clear requests for next use
        self.requests = []
        
        return batch_job.id

    def wait_for_batch(self, batch_id: str, poll_interval: int = 60) -> Dict[str, Any]:
        """Poll until the batch job is completed."""
        print(f"Waiting for batch {batch_id} to complete...")
        while True:
            batch_job = self.client.batches.retrieve(batch_id)
            status = batch_job.status
            print(f"Batch {batch_id} status: {status}")
            
            if status == "completed":
                return self._get_results(batch_job.output_file_id)
            if status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch job {batch_id} ended with status: {status}")
            
            time.sleep(poll_interval)

    def _get_results(self, file_id: str) -> Dict[str, Any]:
        """Download and parse results from the output file."""
        file_response = self.client.files.content(file_id)
        results = {}
        for line in file_response.text.splitlines():
            res = json.loads(line)
            custom_id = res["custom_id"]
            if res["response"]["status_code"] == 200:
                results[custom_id] = res["response"]["body"]["choices"][0]["message"]["content"]
            else:
                results[custom_id] = f"ERROR: {res['response']['body']}"
        return results

    def get_batch_status(self, batch_id: str) -> str:
        """Get the current status of a batch job."""
        return self.client.batches.retrieve(batch_id).status

class CaptureLLM(BaseChatModel):
    """Mock LLM that captures prompts for the Batch API."""
    
    # Context variable to hold the custom ID for the current thread/task
    # Use ClassVar so Pydantic doesn't try to deepcopy it
    _context_custom_id: ClassVar[contextvars.ContextVar] = contextvars.ContextVar("current_custom_id", default="")
    
    batch_manager: OpenAIBatchManager = Field(exclude=True)
    model: str = "gpt-5.4"
    
    @property
    def current_custom_id(self) -> str:
        return self._context_custom_id.get()
    
    @current_custom_id.setter
    def current_custom_id(self, value: str):
        self._context_custom_id.set(value)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Extract prompt in OpenAI format
        openai_messages = []
        full_text = ""
        for m in messages:
            role = "user"
            if m.type == "human": role = "user"
            elif m.type == "ai": role = "assistant"
            elif m.type == "system": role = "system"
            openai_messages.append({"role": role, "content": m.content})
            full_text += f"{role}: {m.content}\n"
            
        # Create a unique ID for this specific call to prevent overwrites
        import hashlib
        prompt_hash = hashlib.md5(full_text.encode()).hexdigest()[:8]
        unique_custom_id = f"{self.current_custom_id}_{prompt_hash}"

        # Check if we already have a result injected
        if unique_custom_id in self.batch_manager.results:
            content = self.batch_manager.results[unique_custom_id]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
            
        # Otherwise capture and return placeholder
        self.batch_manager.add_request(
            custom_id=unique_custom_id,
            model=self.model,
            messages=openai_messages,
            **kwargs
        )
        
        raise BatchCaptureException(f"Prompt captured for {unique_custom_id}")

    @property
    def _llm_type(self) -> str:
        return "capture-llm"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Bind tools to the model. Required for agents that use tools."""
        return self.bind(tools=tools, **kwargs)
