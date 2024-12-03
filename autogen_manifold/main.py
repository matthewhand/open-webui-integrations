"""
DO NOT CHANGE THE FOLLOWING:
old requirements: autogen-agentchat==0.4.0.dev8, asyncio, dataclasses, pydantic, fastapi, aiohttp, requests
requirements: ag2,flaml[automl]

Code doesn't work for you? USE DOCKER!
"""
import asyncio
import json
import logging
import html
import random
import uuid
from dataclasses import dataclass
from typing import (
    Dict,
    Any,
    Callable,
    Awaitable,
    Optional,
    Sequence,
    Union,
    AsyncGenerator,
    List,
)

from pydantic import BaseModel, Field

# Necessary imports for AutoGen AgentChat functionality
from autogen.agentchat.contrib.captainagent import CaptainAgent
from autogen import UserProxyAgent
from autogen_core.components.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    RequestUsage,
)

# Import Open WebUI's generate_chat_completions
from open_webui.main import generate_chat_completions

# Mocking StreamingResponse for demonstration purposes
# Replace this with the actual import if available
from typing import AsyncIterator


class StreamingResponse:
    def __init__(self, body_iterator: AsyncIterator[Union[bytes, str]]):
        self.body_iterator = body_iterator


# Define the User class with required attributes
@dataclass
class User:
    id: str
    username: str
    name: str
    role: str
    email: str


# Instantiate the mock_user with all necessary attributes
mock_user = User(
    id="autogen_manifold",
    username="autogen_manifold",
    name="Autogen Manifold",
    role="admin",
    email="root@host.docker.internal",
)

# Define available functions for function calling through valves
available_functions = []  # This will be populated dynamically from valves

# Function Implementations Mapping
function_implementations: Dict[str, Callable[[Dict[str, Any]], str]] = {}


# Example Function Implementations
def get_current_time(arguments: Dict[str, Any]) -> str:
    """
    Returns the current UTC time.
    """
    logging.debug("Function 'get_current_time' started with arguments: %s", arguments)
    import datetime

    current_time = datetime.datetime.utcnow().isoformat() + "Z"
    response = f"The current UTC time is {current_time}."
    logging.debug("Function 'get_current_time' returning: %s", response)
    return response


def echo(arguments: Dict[str, Any]) -> str:
    """
    Echoes the input message.
    """
    logging.debug("Function 'echo' started with arguments: %s", arguments)
    message = arguments.get("message", "")
    response = f"Echo: {message}"
    logging.debug("Function 'echo' returning: %s", response)
    return response


# Register Example Functions
function_implementations["get_current_time"] = get_current_time
function_implementations["echo"] = echo


class OpenWebUIChatCompletionClient(ChatCompletionClient):
    """
    A custom chat completion client that interfaces with Open WebUI's generate_chat_completions.
    """

    def __init__(self, model: str, functions: Optional[list] = None, **kwargs):
        """
        Initialize the client with the specified model and any additional parameters.

        Args:
            model (str): The model name to use for generating completions.
            functions (Optional[list]): A list of function definitions.
            **kwargs: Additional keyword arguments for configuration.
        """
        logging.debug(
            "Initializing OpenWebUIChatCompletionClient with model: %s, functions: %s, kwargs: %s",
            model,
            functions,
            kwargs,
        )
        self.model = model
        self.functions = functions or []
        self.config = kwargs
        # Function calling is always enabled
        self._model_capabilities = ModelCapabilities(
            vision=False, function_calling=True, json_output=True
        )
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        # Logging setup
        self.log = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)
        self.log.setLevel(
            logging.DEBUG if kwargs.get("debug_valve", True) else logging.INFO
        )

        self.log_debug("OpenWebUIChatCompletionClient initialized.")

    def log_debug(self, message: str):
        """Log debug messages."""
        self.log.debug(message)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return the model's simulated capabilities."""
        self.log_debug(f"Accessing capabilities: {self._model_capabilities}")
        return self._model_capabilities

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Any] = [],  # Adjust tool typing if necessary
        json_output: Optional[bool] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[Any] = None,  # Adjust typing based on actual token
    ) -> CreateResult:
        """
        Generate a chat completion using Open WebUI's generate_chat_completions.

        Args:
            messages (Sequence[LLMMessage]): The input messages.
            tools (Sequence[Any], optional): Tools to assist in completion. Defaults to [].
            json_output (Optional[bool], optional): Whether to output in JSON format. Defaults to None.
            extra_create_args (Dict[str, Any], optional): Additional arguments. Defaults to {}.
            cancellation_token (Optional[Any], optional): Token to cancel the operation. Defaults to None.

        Returns:
            CreateResult: The result of the completion.
        """
        self.log_debug(
            "Function 'create' started with messages: %s, tools: %s, json_output: %s, extra_create_args: %s, cancellation_token: %s",
            messages,
            tools,
            json_output,
            extra_create_args,
            cancellation_token,
        )

        prompt = self._combine_messages(messages)
        self.log_debug(f"Creating completion with prompt: {prompt}")

        # Ensure no empty 'system' messages are included
        filtered_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
            if hasattr(msg, "role") and msg.content and msg.content.strip() != ""
        ]

        # Check if a non-empty 'system' message already exists
        has_non_empty_system = any(
            msg["role"] == "system" and msg["content"].strip() != ""
            for msg in filtered_messages
        )
        self.log_debug(f"Has non-empty 'system' message: {has_non_empty_system}")

        payload = {
            "model": self.model,
            "messages": filtered_messages,
            "functions": self.functions,  # Include functions in the payload
            "stream": False,  # Set to False for simplicity
            **extra_create_args,
        }

        self.log_debug(f"Payload sent to generate_chat_completions: {payload}")

        try:
            response = await generate_chat_completions(
                form_data=payload, user=mock_user, bypass_filter=True
            )
            self.log_debug(f"Received response: {response}")
            self.log_debug(f"Response type: {type(response)}")

            # Handle response based on its type
            if isinstance(response, dict):
                message = response.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "No response generated.")
                function_call = message.get("function_call", None)
            elif hasattr(response, "content"):
                content = response.content
                function_call = None  # Assuming TextMessage doesn't support function_call
            else:
                content = "No response generated."
                function_call = None

            self.log_debug(f"Extracted content: {content}")
            self.log_debug(f"Function call: {function_call}")

            # Check if function_call is present in the response
            if function_call:
                self.log_debug(f"Function call detected: {function_call}")
                # Execute the function
                mock_function_response = self._execute_function(function_call)
                self.log_debug(f"Function executed: {mock_function_response}")

                # Only append if function response content is not empty
                if mock_function_response["response"].strip():
                    function_message = {
                        "role": "function",
                        "name": mock_function_response["name"],
                        "content": mock_function_response["response"],
                    }

                    # Append the function response as a separate message
                    updated_messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in filtered_messages
                    ] + [function_message]

                    self.log_debug(
                        f"Updated messages with function response: {function_message}"
                    )

                    # Make another call to generate_chat_completions with updated messages
                    response = await generate_chat_completions(
                        form_data={
                            "model": self.model,
                            "messages": updated_messages,
                            "functions": self.functions,
                            "stream": False,
                            **extra_create_args,
                        },
                        user=mock_user,
                        bypass_filter=True,
                    )
                    self.log_debug(f"Received response after function execution: {response}")

                    # Extract content from the new response
                    if isinstance(response, dict):
                        message = response.get("choices", [{}])[0].get("message", {})
                        content = message.get("content", "No response generated.")
                    elif hasattr(response, "content"):
                        content = response.content
                    else:
                        content = "No response generated."

            usage = RequestUsage(
                prompt_tokens=0, completion_tokens=len(content.split())
            )
            self._actual_usage = usage
            self._total_usage.prompt_tokens += usage.prompt_tokens
            self._total_usage.completion_tokens += usage.completion_tokens

            self.log_debug(f"Usage updated: {usage}")
            self.log_debug(f"Total usage: {self._total_usage}")

            self.log_debug("Function 'create' completed successfully.")
            return CreateResult(
                finish_reason="stop", content=content, usage=usage, cached=False
            )

        except Exception as e:
            self.log_debug(f"Error generating completion: {e}")
            return CreateResult(
                finish_reason="error",
                content=f"Error generating completion: {str(e)}",
                usage=self._actual_usage,
                cached=False,
            )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Any] = [],  # Adjust tool typing if necessary
        json_output: Optional[bool] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[Any] = None,  # Adjust typing based on actual token
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completions using Open WebUI's generate_chat_completions.

        Args:
            messages (Sequence[LLMMessage]): The input messages.
            tools (Sequence[Any], optional): Tools to assist in completion. Defaults to [].
            json_output (Optional[bool], optional): Whether to output in JSON format. Defaults to None.
            extra_create_args (Dict[str, Any], optional): Additional arguments. Defaults to {}.
            cancellation_token (Optional[Any], optional): Token to cancel the operation. Defaults to None.

        Yields:
            AsyncGenerator[str, None]: The generated tokens.
        """
        self.log_debug(
            "Function 'create_stream' started with messages: %s, tools: %s, json_output: %s, extra_create_args: %s, cancellation_token: %s",
            messages,
            tools,
            json_output,
            extra_create_args,
            cancellation_token,
        )

        prompt = self._combine_messages(messages)
        self.log_debug(f"Creating streaming completion with prompt: {prompt}")

        # Ensure no empty 'system' messages are included
        filtered_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
            if hasattr(msg, "role") and msg.content and msg.content.strip() != ""
        ]

        # Check if a non-empty 'system' message already exists
        has_non_empty_system = any(
            msg["role"] == "system" and msg["content"].strip() != ""
            for msg in filtered_messages
        )
        self.log_debug(f"Has non-empty 'system' message: {has_non_empty_system}")

        payload = {
            "model": self.model,
            "messages": filtered_messages,
            "functions": self.functions,  # Include functions in the payload
            "stream": True,  # Enable streaming
            **extra_create_args,
        }

        self.log_debug(
            f"Payload sent to generate_chat_completions for streaming: {payload}"
        )

        try:
            response = await generate_chat_completions(
                form_data=payload, user=mock_user, bypass_filter=True
            )
            self.log_debug(f"Received streaming response: {response}")
            self.log_debug(f"Response type: {type(response)}")

            if isinstance(response, StreamingResponse):
                async for chunk in response.body_iterator:
                    self.log_debug(f"Received chunk: {chunk}")
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode("utf-8")
                        self.log_debug(f"Decoded chunk: {chunk_str}")
                    else:
                        chunk_str = chunk
                        self.log_debug(f"Chunk is already a string: {chunk_str}")

                    # Parse the chunk to identify if it's a function_call
                    try:
                        data = json.loads(chunk_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            message = data["choices"][0].get("message", {})
                            function_call = message.get("function_call", None)
                            role = message.get("role", "unknown")

                            self.log_debug(f"Parsed message: role={role}, function_call={function_call}")

                            if role == "system" and not message.get("content", "").strip():
                                self.log_debug("Skipping empty 'system' message.")
                                continue  # Skip empty 'system' messages

                            if function_call:
                                self.log_debug(
                                    f"Function call detected in stream: {function_call}"
                                )
                                # Execute the function
                                mock_function_response = self._execute_function(
                                    function_call
                                )
                                self.log_debug(
                                    f"Function executed in stream: {mock_function_response}"
                                )

                                # Only emit if function response content is not empty
                                if mock_function_response["response"].strip():
                                    # Create a function response message
                                    function_message = {
                                        "role": "function",
                                        "name": mock_function_response["name"],
                                        "content": mock_function_response["response"],
                                    }

                                    # Yield the function response as a separate event
                                    function_response_str = json.dumps(function_message)
                                    self.log_debug(
                                        f"Emitting function response: {function_response_str}"
                                    )
                                    yield f"data: {function_response_str}\n\n"

                                    # Continue the conversation with the function response
                                    updated_messages = [
                                        {"role": msg["role"], "content": msg["content"]}
                                        for msg in filtered_messages
                                    ] + [function_message]
                                    # Update messages for continued conversation
                                    filtered_messages = updated_messages

                                    # Optionally, send updated messages to continue the stream
                                    # Here, we'll assume the framework handles the continuation

                                    continue  # Skip yielding the original chunk

                    except json.JSONDecodeError:
                        self.log_debug("JSON decode error for chunk: %s", chunk_str)
                        pass  # Not a JSON chunk, yield as is

                    yield chunk_str
            else:
                if isinstance(response, dict):
                    message = response.get("choices", [{}])[0].get("message", {})
                    role = message.get("role", "Unknown")
                    content = message.get("content", "No response generated.")

                    self.log_debug(f"Parsed message: role={role}, content={content}")

                    if role == "system" and not content.strip():
                        self.log_debug("Skipping empty 'system' message.")
                    else:
                        self.log_debug(f"Final content from non-streaming response: {content}")
                        yield content

            self.log_debug("Function 'create_stream' completed successfully.")

        except Exception as e:
            self.log_debug(f"Error generating streaming completion: {e}")
            yield f"Error generating streaming completion: {str(e)}"

    def _execute_function(self, function_call: dict) -> dict:
        """
        Execute the function call based on the function name and arguments.

        Args:
            function_call (dict): The function call details.

        Returns:
            dict: The function execution result.
        """
        self.log_debug(f"Function '_execute_function' started with function_call: {function_call}")
        function_name = function_call.get("name", "unknown_function")
        arguments = function_call.get("arguments", {})

        self.log_debug(
            f"Executing function '{function_name}' with arguments: {arguments}"
        )

        # Retrieve the function implementation from the mapping
        function = function_implementations.get(function_name, None)

        if function:
            try:
                response = function(arguments)
                self.log_debug(f"Function '{function_name}' executed successfully with response: {response}")
            except Exception as e:
                response = f"Error executing function '{function_name}': {str(e)}"
                self.log_debug(response)
        else:
            response = f"Function '{function_name}' is not defined."
            self.log_debug(response)

        self.log_debug("Function '_execute_function' completed.")
        return {"name": function_name, "response": response}

    def _combine_messages(self, messages: Sequence[LLMMessage]) -> str:
        """
        Combine LLMMessage instances into a single string prompt.

        Args:
            messages (Sequence[LLMMessage]): The input messages.

        Returns:
            str: The combined prompt string.
        """
        self.log_debug("Function '_combine_messages' started with messages: %s", messages)
        combined = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in messages
            if hasattr(msg, "role") and msg.content and msg.content.strip() != ""
        )
        self.log_debug(f"Combined messages into prompt: {combined}")
        self.log_debug("Function '_combine_messages' completed.")
        return combined

    def reset(self):
        """Reset usage statistics."""
        self.log_debug("Function 'reset' started.")
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.log_debug("Usage statistics reset.")
        self.log_debug("Function 'reset' completed.")


class Pipe:
    """
    The Pipe class manages the AG2 CaptainAgent workflow with OpenAI-compatible endpoint,
    collapsible UI, and a configurable CaptainAgent valve.
    """

    class Valves(BaseModel):
        """
        Configuration valves for the pipeline.
        """
        default_model_id: str = Field(
            default="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0",
            description="Default model ID for OpenWebUIChatCompletionClient.",
        )
        enable_captain_agent: bool = Field(
            default=True, description="Enable CaptainAgent functionality."
        )
        function_definitions: list = Field(
            default=[
                {
                    "name": "get_current_time",
                    "description": "Returns the current UTC time.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
                {
                    "name": "echo",
                    "description": "Echoes the input message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to echo back.",
                            },
                        },
                        "required": ["message"],
                    },
                },
                # Add more function definitions here
            ],
            description="List of function definitions.",
        )
        function_implementations: Dict[str, Callable[[Dict[str, Any]], str]] = Field(
            default_factory=lambda: {
                "get_current_time": get_current_time,
                "echo": echo,
                # Add more function implementations here
            },
            description="Mapping of function names to their implementations.",
        )
        debug_valve: bool = Field(default=True, description="Enable debug logging.")

    def __init__(self, endpoint: Optional[str] = None):
        self.log_debug("Initializing Pipe class.")
        self.valves = self.Valves()
        if endpoint:
            self.valves.default_model_id = endpoint
            self.log_debug(f"Overriding default_model_id with endpoint: {endpoint}")

        # Initialize global function mappings
        global available_functions, function_implementations
        available_functions = self.valves.function_definitions
        function_implementations = self.valves.function_implementations

        # Initialize logging
        self.log = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)
        self.log.setLevel(logging.DEBUG if self.valves.debug_valve else logging.INFO)

        self.log_debug("Pipe initialized with the following valves:")
        self.log_debug(f"{self.valves}")

        # Initialize agents if the valve is enabled
        self.captain_agent: Optional[CaptainAgent] = None
        self.user_proxy_agent: Optional[UserProxyAgent] = None
        if self.valves.enable_captain_agent:
            self._initialize_agents()

        # State variables
        self.request_id: Optional[uuid.UUID] = None
        self.completed_contributors: set = set()
        self.interrupted_contributors: set = set()
        self.model_outputs: Dict[str, Any] = {}
        self.final_status_emitted: bool = False

    def log_debug(self, message: str):
        """Log debug messages."""
        if self.valves.debug_valve:
            self.log.debug(message)

    def _initialize_agents(self):
        """
        Initializes the CaptainAgent and UserProxyAgent with OpenWebUIChatCompletionClient.
        """
        self.log_debug("[PIPE] Initializing CaptainAgent and UserProxyAgent.")

        # Define the custom model client configuration
        model_config = {
            "model": self.valves.default_model_id,  # Use default_model_id from valves
            "functions": available_functions,
            "debug_valve": self.valves.debug_valve,
        }

        # Initialize the OpenWebUIChatCompletionClient
        self.custom_model_client = OpenWebUIChatCompletionClient(
            model=model_config["model"],
            functions=model_config["functions"],
            debug_valve=model_config["debug_valve"],
        )

        # Define LLM configuration for CaptainAgent
        llm_config = {
            "temperature": 0,  # Adjust as needed
            "client": self.custom_model_client,
        }

        # Initialize CaptainAgent with the custom model client
        self.captain_agent = CaptainAgent(
            name="captain_agent",
            llm_config=llm_config,
            code_execution_config={"use_docker": False, "work_dir": "groupchat"},
        )

        # Initialize UserProxyAgent
        self.user_proxy_agent = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
        )

        self.log_debug("CaptainAgent and UserProxyAgent initialized successfully.")

    async def pipe(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the AG2 CaptainAgent workflow and generates collapsible UI for agent-to-agent communication.

        Args:
            body (Dict[str, Any]): The incoming request payload.
            __event_emitter__: The event emitter for sending messages and statuses.

        Returns:
            Dict[str, Any]: The task results or error information.
        """
        self.log_debug(
            "Function 'pipe' started with body: %s, __event_emitter__: %s", body, __event_emitter__
        )
        if not self.valves.enable_captain_agent:
            error_message = "[PIPE] CaptainAgent is disabled via valve."
            self.log.error(error_message)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "Error",
                            "description": error_message,
                            "done": True,
                        },
                    }
                )
            self.log_debug("Function 'pipe' completed with errors.")
            return {"status": "error", "message": error_message}

        # Reset state for the new request
        self.reset_state()
        self.request_id = uuid.uuid4()
        self.log_debug(f"[PIPE] Starting new request with ID: {self.request_id}")

        try:
            # Step 1: Validate incoming messages
            user_messages = body.get("messages", [])
            self.log_debug(f"[PIPE] Received user messages: {user_messages}")
            if not user_messages:
                raise ValueError("[PIPE] No messages provided in the request body.")

            # Extract the latest user message
            latest_message = user_messages[-1].get("content", "").strip()
            self.log_debug(f"[PIPE] Latest user message: '{latest_message}'")
            if not latest_message:
                raise ValueError("[PIPE] No valid user message content provided.")

            # Step 2: Summarize history if available
            history_summary = "No prior conversation history."
            if len(user_messages) > 1:
                history_summary = self.summarize_history(user_messages[:-1])
            self.log_debug(f"[PIPE] History summary: '{history_summary}'")

            # Construct the task message
            task_message = (
                f"Summary of conversation history: '{history_summary}'. "
                f"User's instruction: '{latest_message}'."
            )
            self.log_debug(f"[PIPE] Constructed task message: {task_message}")

            # Emit task initiation status
            if __event_emitter__:
                self.log_debug("Emitting task initiation status.")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "Info",
                            "description": "Task initiated.",
                            "done": False,
                        },
                    }
                )

            # Step 3: Execute the task using CaptainAgent via UserProxyAgent
            self.log.debug("[PIPE] Starting conversation via UserProxyAgent.")
            result = await self.user_proxy_agent.initiate_chat_async(
                self.captain_agent,
                message=task_message
            )
            self.log.debug(f"[PIPE] Conversation result: {result}")

            # Step 4: Process and emit agent-to-agent communication
            if __event_emitter__:
                for idx, message in enumerate(result):
                    # Determine the sender based on the index (assuming alternating)
                    sender = "CaptainAgent" if idx % 2 == 0 else "UserProxyAgent"
                    collapsible_content = self._generate_collapsible_ui(
                        sender=sender,
                        message=message
                    )
                    self.log.debug(f"[PIPE] Emitting collapsible content: {collapsible_content}")
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": collapsible_content},
                        }
                    )

            # Emit task completion status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "Info",
                            "description": "Task completed successfully.",
                            "done": True,
                        },
                    }
                )
                self.final_status_emitted = True
                self.log_debug("Emitting task completion status.")

            # Return the final result
            self.log_debug("Function 'pipe' completed successfully.")
            return {"status": "success", "data": result}

        except Exception as e:
            error_message = str(e)
            self.log.error(f"[PIPE] Error in pipeline: {e}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "Error",
                            "description": error_message,
                            "done": True,
                        },
                    }
                )
            self.log_debug("Function 'pipe' completed with errors.")
            return {"status": "error", "message": error_message}

    def summarize_history(self, messages: List[Dict[str, Any]]) -> str:
        """
        Summarizes the message history for context.

        Args:
            messages (List[Dict[str, Any]]): List of messages.

        Returns:
            str: Summary of the message history.
        """
        self.log_debug("[PIPE] Summarizing message history.")
        summary = " | ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        self.log_debug(f"[PIPE] History summary: {summary}")
        return summary

    def reset_state(self):
        """Reset state variables for a new task pipeline."""
        self.log_debug("Function 'reset_state' started.")
        self.request_id = None
        self.completed_contributors.clear()
        self.interrupted_contributors.clear()
        self.model_outputs.clear()
        self.final_status_emitted = False
        self.log_debug("[PIPE] State reset for new request.")
        self.log_debug("Function 'reset_state' completed.")

    def _generate_collapsible_ui(self, sender: str, message: str) -> str:
        """
        Generates collapsible HTML UI for agent-to-agent communication.

        Args:
            sender (str): The sender of the message.
            message (str): The content of the message.

        Returns:
            str: Collapsible HTML content.
        """
        self.log_debug("Function '_generate_collapsible_ui' started with sender: %s, message: %s", sender, message)
        summary = f"{sender} says: {html.escape(message[:50])}..." if len(message) > 50 else f"{sender} says: {html.escape(message)}"
        collapsible_content = f"""
        <details>
            <summary>{summary}</summary>
            <p>{html.escape(message)}</p>
        </details>
        """
        self.log_debug(f"Generated collapsible UI: {collapsible_content.strip()}")
        self.log_debug("Function '_generate_collapsible_ui' completed.")
        return collapsible_content.strip()

    # Mock event emitter for testing
    @staticmethod
    async def mock_event_emitter(event: Dict[str, Any]):
        print(f"Emitted Event: {json.dumps(event, indent=2)}")

    # Example Runner (Optional)
    # Uncomment to test the Pipe class independently.

    # async def run_pipe_example():
    #     pipe = Pipe()
    #     body = {
    #         "messages": [
    #             {"role": "user", "content": "Explain how AI can improve education."}
    #         ]
    #     }
    #     print(pipe.valves.default_model_id)  # Should print the default model ID
    #     await pipe.pipe(body=body, __event_emitter__=Pipe.mock_event_emitter)

    # def execute_run_pipe():
    #     logging.debug("Function 'execute_run_pipe' started.")
    #     try:
    #         asyncio.run(run_pipe_example())
    #         logging.debug("Asyncio run completed successfully.")
    #     except RuntimeError as e:
    #         if "asyncio.run() cannot be called from a running event loop" in str(e):
    #             loop = asyncio.get_event_loop()
    #             if loop and loop.is_running():
    #                 logging.debug("Event loop already running. Creating a task for 'run_pipe_example'.")
    #                 task = loop.create_task(run_pipe_example())
    #             else:
    #                 logging.debug("Event loop not running. Re-raising the exception.")
    #                 raise
    #         else:
    #             logging.debug("RuntimeError encountered: %s", e)
    #             raise
    #     logging.debug("Function 'execute_run_pipe' completed.")

    # # Uncomment the following line to execute the pipe when this script is run directly
    # # execute_run_pipe()
