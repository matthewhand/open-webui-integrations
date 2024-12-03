"""
DO NOT CHANGE THE FOLLOWING:
requirements: ag2, flaml[automl]

Code doesn't work for you? USE DOCKER!
"""
import asyncio
import json
import logging
import html
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

# --------------------- Global Function Definitions ---------------------

# Define available functions for function calling
available_functions = [
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
    # Add more function definitions here as needed
]

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

# --------------------- End of Global Function Definitions ---------------------


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

        # **Hard strip all 'system' messages entirely**
        filtered_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
            if hasattr(msg, "role")
            and msg.content
            and msg.content.strip() != ""
            and msg.role.lower().strip() != "system"  # Exclude 'system' messages
        ]

        self.log_debug(f"Filtered messages (excluding 'system'): {filtered_messages}")

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
                        function_call = message.get("function_call", None)
                        self.log_debug(f"Post-function call function_call: {function_call}")
                    elif hasattr(response, "content"):
                        content = response.content
                        function_call = None
                        self.log_debug("No function_call in post-function response.")
                    else:
                        content = "No response generated."
                        function_call = None
                        self.log_debug("Unknown response format after function execution.")

                    # Prevent further function calls to avoid infinite loops
                    if function_call:
                        self.log_debug("Additional function_call detected post-execution. Halting to prevent loop.")
                        content += "\n[Note]: Multiple function calls detected. Halting to prevent infinite loop."

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
        except:
            pass

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

        # **Hard strip all 'system' messages entirely**
        filtered_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
            if hasattr(msg, "role")
            and msg.content
            and msg.content.strip() != ""
            and msg.role.lower().strip() != "system"  # Exclude 'system' messages
        ]

        self.log_debug(f"Filtered messages (excluding 'system'): {filtered_messages}")

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

                            if role.lower().strip() == "system" and not message.get("content", "").strip():
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
                                    yield f"Agent Output: {function_response_str}\n"

                                    # Continue the conversation with the function response
                                    updated_messages = [
                                        {"role": msg["role"], "content": msg["content"]}
                                        for msg in filtered_messages
                                    ] + [function_message]
                                    # Update messages for continued conversation
                                    filtered_messages = updated_messages

                                    # Prevent further function calls to avoid infinite loops
                                    self.log_debug("Preventing further function calls to avoid infinite loop.")
                                    break  # Exit the streaming loop

                    except json.JSONDecodeError:
                        self.log_debug("JSON decode error for chunk: %s", chunk_str)
                        pass  # Not a JSON chunk, yield as is

                    yield f"Agent Output: {chunk_str}\n"
            else:
                if isinstance(response, dict):
                    message = response.get("choices", [{}])[0].get("message", {})
                    role = message.get("role", "Unknown")
                    content = message.get("content", "No response generated.")

                    self.log_debug(f"Parsed message: role={role}, content={content}")

                    if role.lower().strip() == "system" and not content.strip():
                        self.log_debug("Skipping empty 'system' message.")
                    else:
                        self.log_debug(f"Final content from non-streaming response: {content}")
                        yield f"Agent Output: {content}\n"

            self.log_debug("Function 'create_stream' completed successfully.")

        except Exception as e:
            self.log_debug(f"Error generating streaming completion: {e}")
            yield f"Agent Output: Error generating streaming completion: {str(e)}\n"

    def oai_messages_to_anthropic_messages(params: Dict[str, Any]) -> list[dict[str, Any]]:
        """Convert messages from OAI format to Anthropic format.
        We correct for any specific role orders and types, etc.
        """

        # Track whether we have tools passed in. If not,  tool use / result messages should be converted to text messages.
        # Anthropic requires a tools parameter with the tools listed, if there are other messages with tool use or tool results.
        # This can occur when we don't need tool calling, such as for group chat speaker selection.
        has_tools = "tools" in params

        # Convert messages to Anthropic compliant format
        processed_messages = []

        # Used to interweave user messages to ensure user/assistant alternating
        user_continue_message = {"content": "Please continue.", "role": "user"}
        assistant_continue_message = {"content": "Please continue.", "role": "assistant"}

        tool_use_messages = 0
        tool_result_messages = 0
        last_tool_use_index = -1
        last_tool_result_index = -1
        for message in params["messages"]:
            if message["role"] == "system":
                params["system"] = params.get("system", "") + ("\n" if "system" in params else "") + message["content"]
            else:
                # New messages will be added here, manage role alternations
                expected_role = "user" if len(processed_messages) % 2 == 0 else "assistant"

                if "tool_calls" in message:
                    # Map the tool call options to Anthropic's ToolUseBlock
                    tool_uses = []
                    tool_names = []
                    for tool_call in message["tool_calls"]:
                        tool_uses.append(
                            ToolUseBlock(
                                type="tool_use",
                                id=tool_call["id"],
                                name=tool_call["function"]["name"],
                                input=json.loads(tool_call["function"]["arguments"]),
                            )
                        )
                        if has_tools:
                            tool_use_messages += 1
                        tool_names.append(tool_call["function"]["name"])

                    if expected_role == "user":
                        # Insert an extra user message as we will append an assistant message
                        processed_messages.append(user_continue_message)

                    if has_tools:
                        processed_messages.append({"role": "assistant", "content": tool_uses})
                        last_tool_use_index = len(processed_messages) - 1
                    else:
                        # Not using tools, so put in a plain text message
                        processed_messages.append(
                            {
                                "role": "assistant",
                                "content": f"Some internal function(s) that could be used: [{', '.join(tool_names)}]",
                            }
                        )
                elif "tool_call_id" in message:
                    if has_tools:
                        # Map the tool usage call to tool_result for Anthropic
                        tool_result = {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message["content"],
                        }

                        # If the previous message also had a tool_result, add it to that
                        # Otherwise append a new message
                        if last_tool_result_index == len(processed_messages) - 1:
                            processed_messages[-1]["content"].append(tool_result)
                        else:
                            if expected_role == "assistant":
                                # Insert an extra assistant message as we will append a user message
                                processed_messages.append(assistant_continue_message)

                            processed_messages.append({"role": "user", "content": [tool_result]})
                            last_tool_result_index = len(processed_messages) - 1

                        tool_result_messages += 1
                    else:
                        # Not using tools, so put in a plain text message
                        processed_messages.append(
                            {"role": "user", "content": f"Running the function returned: {message['content']}"}
                        )
                elif message["content"] == "":
                    # Ignoring empty messages
                    pass
                else:
                    if expected_role != message["role"]:
                        # Inserting the alternating continue message
                        processed_messages.append(
                            user_continue_message if expected_role == "user" else assistant_continue_message
                        )

                    processed_messages.append(message)

        # We'll replace the last tool_use if there's no tool_result (occurs if we finish the conversation before running the function)
        if has_tools and tool_use_messages != tool_result_messages:
            processed_messages[last_tool_use_index] = assistant_continue_message

        # name is not a valid field on messages
        for message in processed_messages:
            if "name" in message:
                message.pop("name", None)

        # Note: When using reflection_with_llm we may end up with an "assistant" message as the last message and that may cause a blank response
        # So, if the last role is not user, add a 'user' continue message at the end
        if processed_messages and processed_messages[-1]["role"] != "user":
            processed_messages.append(user_continue_message)

        return processed_messages


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
        debug_valve: bool = Field(default=True, description="Enable debug logging.")

    def __init__(self, endpoint: Optional[str] = None):
        self.log_debug("Initializing Pipe class.")
        self.valves = self.Valves()
        if endpoint:
            self.valves.default_model_id = endpoint
            self.log_debug(f"Overriding default_model_id with endpoint: {endpoint}")

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
            "model": self.valves.default_model_id,        # Use default_model_id from valves
            "functions": available_functions,              # Use global available_functions
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
                    self.log_debug(f"[PIPE] Emitting collapsible content: {collapsible_content}")
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
            str: Collapsible HTML content prefixed with 'Agent Output: '.
        """
        self.log_debug("Function '_generate_collapsible_ui' started with sender: %s, message: %s", sender, message)
        summary = f"{sender} says: {html.escape(message[:50])}..." if len(message) > 50 else f"{sender} says: {html.escape(message)}"
        collapsible_content = f"""
        <details>
            <summary>{summary}</summary>
            Agent Output: {html.escape(message)}
        </details>
        """
        self.log_debug(f"Generated collapsible UI: {collapsible_content.strip()}")
        self.log_debug("Function '_generate_collapsible_ui' completed.")
        return collapsible_content.strip()

    # --------------------- End of Pipe Class ---------------------

# --------------------- OpenWebUIChatCompletionClient ---------------------

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

        # **Check if messages are LLMMessage instances or dicts**
        if all(isinstance(msg, LLMMessage) for msg in messages):
            self.log_debug("All messages are instances of LLMMessage.")
            message_roles = [msg.role.lower().strip() for msg in messages]
            message_contents = [msg.content.strip() for msg in messages]
        elif all(isinstance(msg, dict) for msg in messages):
            self.log_debug("All messages are dictionaries.")
            message_roles = [msg.get("role", "").lower().strip() for msg in messages]
            message_contents = [msg.get("content", "").strip() for msg in messages]
        else:
            # Mixed types or unexpected types
            self.log.error("Messages are of mixed or unexpected types.")
            error_message = "Messages are of mixed or unexpected types."
            return CreateResult(
                finish_reason="error",
                content=error_message,
                usage=self._actual_usage,
                cached=False,
            )

        # **Guard: Ensure all roles and contents are non-empty**
        for idx, (role, content) in enumerate(zip(message_roles, message_contents)):
            if not role:
                error_message = f"[OpenWebUIChatCompletionClient] Message at index {idx} has an empty 'role'."
                self.log.error(error_message)
                return CreateResult(
                    finish_reason="error",
                    content=error_message,
                    usage=self._actual_usage,
                    cached=False,
                )
            if not content:
                error_message = f"[OpenWebUIChatCompletionClient] Message at index {idx} has empty 'content'."
                self.log.error(error_message)
                return CreateResult(
                    finish_reason="error",
                    content=error_message,
                    usage=self._actual_usage,
                    cached=False,
                )

        # **Count the number of 'system' messages**
        system_message_count = sum(1 for role in message_roles if role == "system")
        self.log_debug(f"Number of 'system' messages: {system_message_count}")

        # **If two or more 'system' messages exist, return immediately**
        if system_message_count >= 2:
            error_message = (
                f"[OpenWebUIChatCompletionClient] Detected {system_message_count} 'system' messages. "
                "Returning immediately to prevent infinite loop."
            )
            self.log.error(error_message)
            return CreateResult(
                finish_reason="error",
                content=error_message,
                usage=self._actual_usage,
                cached=False,
            )

        # **Hard strip all 'system' messages entirely**
        if all(isinstance(msg, LLMMessage) for msg in messages):
            filtered_messages = [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in messages
                if msg.role.lower().strip() != "system" and msg.content.strip() != ""
            ]
        else:  # messages are dicts
            filtered_messages = [
                {
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                }
                for msg in messages
                if msg.get("role", "").lower().strip() != "system" and msg.get("content", "").strip() != ""
            ]

        self.log_debug(f"Filtered messages (excluding 'system'): {filtered_messages}")

        # **Guard: Check if filtered_messages is not empty**
        if not filtered_messages:
            warning_message = "[OpenWebUIChatCompletionClient] No messages left after filtering out 'system' messages."
            self.log.warning(warning_message)
            return CreateResult(
                finish_reason="error",
                content=warning_message,
                usage=self._actual_usage,
                cached=False,
            )

        # **Serialize the filtered messages to a JSON string for regex checking**
        try:
            messages_str = json.dumps(filtered_messages)
            self.log_debug(f"Serialized filtered messages: {messages_str}")
        except Exception as e:
            error_message = f"[OpenWebUIChatCompletionClient] Failed to serialize messages: {str(e)}"
            self.log.error(error_message)
            return CreateResult(
                finish_reason="error",
                content=error_message,
                usage=self._actual_usage,
                cached=False,
            )

        # **Use regex to check if 'system' appears more than once in the serialized messages**
        system_role_matches = re.findall(r'"role"\s*:\s*"system"', messages_str, re.IGNORECASE)
        system_role_count_regex = len(system_role_matches)
        self.log_debug(f"Regex detected 'system' roles: {system_role_count_regex}")

        if system_role_count_regex >= 1:
            # This should not happen as we've already filtered out 'system' messages
            error_message = (
                f"[OpenWebUIChatCompletionClient] Regex detected {system_role_count_regex} 'system' roles in messages after filtering. "
                "Returning immediately to prevent infinite loop."
            )
            self.log.error(error_message)
            return CreateResult(
                finish_reason="error",
                content=error_message,
                usage=self._actual_usage,
                cached=False,
            )

        # **Combine messages into a prompt (optional, based on your implementation)**
        if all(isinstance(msg, LLMMessage) for msg in messages):
            prompt = self._combine_messages(filtered_messages)
        else:
            # If messages are dicts, convert them to LLMMessage instances for combining
            try:
                converted_messages = [LLMMessage(**msg) for msg in filtered_messages]
                prompt = self._combine_messages(converted_messages)
            except Exception as e:
                error_message = f"[OpenWebUIChatCompletionClient] Failed to convert messages to LLMMessage: {str(e)}"
                self.log.error(error_message)
                return CreateResult(
                    finish_reason="error",
                    content=error_message,
                    usage=self._actual_usage,
                    cached=False,
                )
        self.log_debug(f"Creating completion with prompt: {prompt}")

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

            # **Guard: Ensure extracted content is not empty**
            if not content.strip():
                error_message = "[OpenWebUIChatCompletionClient] Extracted content is empty."
                self.log.error(error_message)
                return CreateResult(
                    finish_reason="error",
                    content=error_message,
                    usage=self._actual_usage,
                    cached=False,
                )

            # Check if function_call is present in the response
            if function_call:
                self.log_debug(f"Function call detected: {function_call}")
                # Execute the function
                mock_function_response = self._execute_function(function_call)
                self.log_debug(f"Function executed: {mock_function_response}")

                # **Guard: Ensure function response is not empty**
                if not mock_function_response["response"].strip():
                    error_message = f"[OpenWebUIChatCompletionClient] Function '{mock_function_response['name']}' returned empty response."
                    self.log.error(error_message)
                    return CreateResult(
                        finish_reason="error",
                        content=error_message,
                        usage=self._actual_usage,
                        cached=False,
                    )

                function_message = {
                    "role": "function",
                    "name": mock_function_response["name"],
                    "content": mock_function_response["response"],
                }

                # Append the function response as a separate message
                updated_messages = filtered_messages + [function_message]

                self.log_debug(
                    f"Updated messages with function response: {function_message}"
                )

                # **Guard: Ensure no 'system' messages are introduced**
                system_in_updated = any(
                    msg.get("role", "").lower().strip() == "system" for msg in updated_messages
                )
                if system_in_updated:
                    error_message = (
                        "[OpenWebUIChatCompletionClient] 'system' message detected after function execution. "
                        "Halting to prevent infinite loop."
                    )
                    self.log.error(error_message)
                    return CreateResult(
                        finish_reason="error",
                        content=error_message,
                        usage=self._actual_usage,
                        cached=False,
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
                    function_call = message.get("function_call", None)
                    self.log_debug(f"Post-function call function_call: {function_call}")
                elif hasattr(response, "content"):
                    content = response.content
                    function_call = None
                    self.log_debug("No function_call in post-function response.")
                else:
                    content = "No response generated."
                    function_call = None
                    self.log_debug("Unknown response format after function execution.")

                # **Guard: Ensure extracted content is not empty after function execution**
                if not content.strip():
                    error_message = "[OpenWebUIChatCompletionClient] Extracted content after function execution is empty."
                    self.log.error(error_message)
                    return CreateResult(
                        finish_reason="error",
                        content=error_message,
                        usage=self._actual_usage,
                        cached=False,
                    )

                # Prevent further function calls to avoid infinite loops
                if function_call:
                    self.log_debug("Additional function_call detected post-execution. Halting to prevent loop.")
                    content += "\n[Note]: Multiple function calls detected. Halting to prevent infinite loop."

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
            self.log_debug(f"Error generating streaming completion: {e}")
            yield f"Agent Output: Error generating streaming completion: {str(e)}\n"

    def _execute_function(self, function_call: dict) -> dict:
        """
        Execute the function call based on the function name and arguments.

        Args:
            function_call (dict): The function call details.

        Returns:
            dict: The function execution result.
        """
        logging.debug(f"Function '_execute_function' started with function_call: {function_call}")
        function_name = function_call.get("name", "unknown_function")
        arguments = function_call.get("arguments", {})

        logging.debug(
            f"Executing function '{function_name}' with arguments: {arguments}"
        )

        # Retrieve the function implementation from the mapping
        function = function_implementations.get(function_name, None)

        if function:
            try:
                response = function(arguments)
                logging.debug(f"Function '{function_name}' executed successfully with response: {response}")
            except Exception as e:
                response = f"Error executing function '{function_name}': {str(e)}"
                logging.debug(response)
        else:
            response = f"Function '{function_name}' is not defined."
            logging.debug(response)

        logging.debug("Function '_execute_function' completed.")
        return {"name": function_name, "response": response}

    def _combine_messages(self, messages: Sequence[LLMMessage]) -> str:
        """
        Combine LLMMessage instances into a single string prompt.

        Args:
            messages (Sequence[LLMMessage]): The input messages.

        Returns:
            str: The combined prompt string.
        """
        logging.debug("Function '_combine_messages' started with messages: %s", messages)
        combined = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in messages
            if hasattr(msg, "role") and msg.content and msg.content.strip() != ""
        )
        logging.debug(f"Combined messages into prompt: {combined}")
        logging.debug("Function '_combine_messages' completed.")
        return combined

    def _generate_collapsible_ui(self, sender: str, message: str) -> str:
        """
        Generates collapsible HTML UI for agent-to-agent communication.

        Args:
            sender (str): The sender of the message.
            message (str): The content of the message.

        Returns:
            str: Collapsible HTML content prefixed with 'Agent Output: '.
        """
        self.log_debug("Function '_generate_collapsible_ui' started with sender: %s, message: %s", sender, message)
        summary = f"{sender} says: {html.escape(message[:50])}..." if len(message) > 50 else f"{sender} says: {html.escape(message)}"
        collapsible_content = f"""
        <details>
            <summary>{summary}</summary>
            Agent Output: {html.escape(message)}
        </details>
        """
        self.log_debug(f"Generated collapsible UI: {collapsible_content.strip()}")
        self.log_debug("Function '_generate_collapsible_ui' completed.")
        return collapsible_content.strip()
