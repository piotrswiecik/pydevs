import os
from typing import Optional, Union, Any
from langfuse import Langfuse
from langfuse.client import StatefulTraceClient, StatefulClient, StatefulSpanClient, PromptClient

class LangfuseService:
    def __init__(self, host: Optional[str] = None, secret_key: Optional[str] = None, public_key: Optional[str] = None):
        if host is None:
            host = os.environ.get("LANGFUSE_HOST")
            if host is None:
                raise ValueError("Host must be provided through the constructor or the LANGFUSE_HOST environment variable.")
        
        if secret_key is None:
            secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
            if secret_key is None:
                raise ValueError("Secret_key must be provided through the constructor or the LANGFUSE_SECRET_KEY environment variable.")
        
        if public_key is None:
            public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
            if public_key is None:
                raise ValueError("Public_key must be provided through the constructor or the LANGFUSE_PUBLIC_KEY environment variable.")
        
        self._host = host
        self._secret_key = secret_key
        self._public_key = public_key

        self._client = Langfuse(host=self._host, secret_key=self._secret_key, public_key=self._public_key)

    def create_trace(self, id, name, session_id, user_id) -> StatefulTraceClient:
        return self._client.trace(id=id, name=name, session_id=session_id, user_id=user_id)

    def create_span(self, trace: Union[StatefulTraceClient, StatefulClient], name: str, input: Optional[Any]) -> StatefulSpanClient:
        return trace.span(name=name, input=input)
        
    def create_generation(self, trace: Union[StatefulTraceClient, StatefulClient], name: str, input: Optional[Any], prompt: Optional[PromptClient] = None, config_kwargs: Optional[dict] = None) -> StatefulSpanClient:
        return trace.generation(name=name, input=input, prompt=prompt, **(config_kwargs or {}))
    
    def finalize_span(self, span: StatefulSpanClient, name: str, input: Optional[Any], output: Optional[Any]):
        span.update(name=name, input=input, output=output)
        span.end()

    def finalize_trace(self, trace: StatefulTraceClient, input: Optional[Any], output: Optional[Any]):
        trace.update(input=input, output=output)
        self._client.flush()