"""
DO NOT CHANGE THE FOLLOWING:
requirements: autogen-agentchat==0.4.0.dev8, asyncio, dataclasses, pydantic, fastapi, aiohttp, requests

Code doesn't work for you? USE DOCKER!
"""

import asyncio
import json
import logging
import html
import random
import re
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, Sequence, List

from pydantic import BaseModel, Field, validator, ValidationError

# Necessary imports for AutoGen AgentChat functionality
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.task import TextMentionTermination
from autogen_core.components.models import (
    ChatCompletionClient,
    CreateResult,
    ModelCapabilities,
    RequestUsage,
)

# Import Open WebUI's generate_chat_completions and utility functions
from open_webui.main import generate_chat_completions
from open_webui.utils.misc import pop_system_message


# Define the User class with required attributes
@dataclass
class User:
    id: str
    username: str
    name: str
    role: str
    email: str


# Instantiate the mock_user with all necessary attributes, including 'name'
mock_user = User(
    id="autogen_manifold",
    username="autogen_manifold",
    name="Autogen Manifold",
    role="admin",
    email="root@host.docker.internal",
)

# Define available functions for function calling through valves
available_functions: List[Dict[str, Any]] = []  # This will be populated dynamically from valves

# Function Implementations Mapping
function_implementations: Dict[str, Callable[[Dict[str, Any]], str]] = {}


# Example Function Implementations
def get_current_time(arguments: Dict[str, Any]) -> str:
    """
    Returns the current UTC time.
    """
    import datetime

    current_time = datetime.datetime.utcnow().isoformat() + "Z"
    return f"The current UTC time is {current_time}."


def echo(arguments: Dict[str, Any]) -> str:
    """
    Echoes the input message.
    """
    message = arguments.get("message", "")
    return f"Echo: {html.escape(message)}"


# Register Example Functions
function_implementations["get_current_time"] = get_current_time
function_implementations["echo"] = echo


# Define the Custom LLMMessage Class for Autogen Interactions
class LLMMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

    def __repr__(self):
        return f"LLMMessage(role='{self.role}', content='{self.content}')"


class OpenWebUIChatCompletionClient(ChatCompletionClient):
    """
    A custom chat completion client that interfaces with Open WebUI's generate_chat_completions.
    """

    def __init__(
        self,
        model: str,
        functions: Optional[List[Dict[str, Any]]] = None,
        system_message: str = "",
        debug_valve: bool = True,
        **kwargs,
    ):
        """
        Initialize the client with the specified model and any additional parameters.

        Args:
            model (str): The model name to use for generating completions.
            functions (Optional[List[Dict[str, Any]]]): A list of function definitions.
            system_message (str): The system message specific to the agent.
            debug_valve (bool): Enable or disable debug logging.
            **kwargs: Additional keyword arguments for configuration.
        """
        self.model = model
        self.functions = functions or []
        self.system_message = system_message
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
        self.log.setLevel(logging.DEBUG if debug_valve else logging.INFO)

        self.log_debug("OpenWebUIChatCompletionClient initialized.")
        self.log_debug(f"Model: {self.model}")
        self.log_debug(f"Functions: {self.functions}")
        self.log_debug(f"System Message: {self.system_message}")
        self.log_debug(f"Debug Valve: {debug_valve}")

    def log_debug(self, message: str):
        """Log debug messages."""
        self.log.debug(message)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return the model's simulated capabilities."""
        self.log_debug(f"Capabilities type: {type(self._model_capabilities)}")
        return self._model_capabilities

    def verify_messages(self, messages: Sequence[Dict[str, Any]]) -> bool:
        """
        Verify that the message payload contains exactly one system message.

        Args:
            messages (Sequence[Dict[str, Any]]): The input messages.

        Returns:
            bool: True if verification passes, False otherwise.
        """
        self.log_debug("Verifying messages for correct system message count.")
        system_messages = [msg for msg in messages if msg.get("role", "").lower() == "system"]
        self.log_debug(f"Total messages received for verification: {len(messages)}")
        self.log_debug(f"Number of system messages detected: {len(system_messages)}")
        for i, msg in enumerate(messages):
            self.log_debug(f"Message {i+1}: Role='{msg.get('role')}', Content='{msg.get('content')}'")

        if len(system_messages) != 1:
            self.log_debug(
                f"Verification failed: Expected 1 system message, found {len(system_messages)}."
            )
            return False

        self.log_debug("Verification passed: Exactly one system message found.")
        return True

    def _combine_messages(self, messages: Sequence[Dict[str, Any]]) -> str:
        """
        Combine message dictionaries into a single string prompt.

        Args:
            messages (Sequence[Dict[str, Any]]): The input messages.

        Returns:
            str: The combined prompt string.
        """
        combined = "\n".join(
            f"{msg.get('role')}: {msg.get('content')}"
            for msg in messages
            if msg.get('role') and msg.get('content', '').strip()
        )
        self.log_debug(f"Combined messages into prompt: {combined}")
        return combined

    def _execute_function(self, function_call: Dict[str, Any]) -> Dict[str, str]:
        """
        Execute the function call based on the function name and arguments.

        Args:
            function_call (Dict[str, Any]): The function call details.

        Returns:
            Dict[str, str]: The function execution result.
        """
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

        return {"name": function_name, "response": response}

    async def create(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Sequence[Any] = [],  # Adjust tool typing if necessary
        json_output: Optional[bool] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[Any] = None,  # Adjust typing based on actual token
    ) -> CreateResult:
        """
        Generate a chat completion using Open WebUI's generate_chat_completions.

        Args:
            messages (Sequence[Dict[str, Any]]): The input messages.
            tools (Sequence[Any], optional): Tools to assist in completion. Defaults to [].
            json_output (Optional[bool], optional): Whether to output in JSON format. Defaults to None.
            extra_create_args (Dict[str, Any], optional): Additional arguments. Defaults to {}.
            cancellation_token (Optional[Any], optional): Token to cancel the operation. Defaults to None.

        Returns:
            CreateResult: The result of the completion.
        """
        self.log_debug("Entering 'create' method.")

        # Filter out existing system messages except the agent-specific one
        self.log_debug("Filtering out system messages from the input.")
        filtered_messages = [msg for msg in messages if msg.get("role", "").lower() != "system"]
        self.log_debug(f"Filtered messages (excluding system messages): {filtered_messages}")

        # Check if the first message is the system message with the same content
        if not (
            filtered_messages
            and filtered_messages[0].get("role", "").lower() == "system"
            and filtered_messages[0].get("content", "").strip() == self.system_message.strip()
        ):
            if self.system_message.strip():
                system_message = {"role": "system", "content": self.system_message.strip()}
                # Ensure system message is first
                filtered_messages.insert(0, system_message)
                self.log_debug("System message added to the beginning of messages.")
            else:
                self.log_debug("No system message provided; skipping insertion.")

        self.log_debug(f"Messages after system message insertion: {filtered_messages}")

        # Verify messages before proceeding
        if not self.verify_messages(filtered_messages):
            error_content = "Payload verification failed: Incorrect number of system messages."
            self.log_debug(error_content)
            return CreateResult(
                finish_reason="error",
                content=error_content,
                usage=self._actual_usage,
                cached=False,
            )

        # Combine messages into prompt (for logging purposes)
        prompt = self._combine_messages(filtered_messages)
        self.log_debug(f"Creating completion with prompt: {prompt}")

        # Prepare the payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                }
                for msg in filtered_messages
                if msg.get("role") and msg.get("content", '').strip()
            ],
            "functions": self.functions,  # Include functions in the payload
            "stream": False,  # Set to False since we're focusing on non-streaming generation
            **extra_create_args,
        }

        self.log_debug(f"Payload prepared for generate_chat_completions:\n{json.dumps(payload, indent=2)}")

        try:
            # Send the request to generate_chat_completions
            self.log_debug("Sending request to generate_chat_completions.")
            response = await generate_chat_completions(
                form_data=payload, user=mock_user, bypass_filter=True
            )
            self.log_debug(f"Received response from generate_chat_completions: {response}")

            # Handle response based on its type
            if isinstance(response, dict):
                message = response.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "No response generated.").strip()
                function_call = message.get("function_call", None)
                self.log_debug(f"Parsed response content: {content}")
                self.log_debug(f"Function call in response: {function_call}")
            elif hasattr(response, "content"):
                content = response.content.strip()
                function_call = None  # Assuming TextMessage doesn't support function_call
                self.log_debug(f"Parsed response content from TextMessage: {content}")
            else:
                content = "No response generated."
                function_call = None
                self.log_debug("Unknown response format received.")

            # Check if function_call is present in the response
            if function_call:
                self.log_debug(f"Function call detected: {function_call}")
                # Execute the function
                function_response = self._execute_function(function_call)
                self.log_debug(f"Function execution result: {function_response}")

                # Append the function response as a separate message
                function_message = {"role": "function", "content": function_response["response"]}
                self.log_debug(f"Function message created: {function_message}")

                # Update the payload to include the function response
                updated_messages = list(filtered_messages) + [function_message]
                self.log_debug(f"Updated messages after appending function response: {updated_messages}")

                # Verify updated messages
                if not self.verify_messages(updated_messages):
                    error_content = "Payload verification failed after function execution: Incorrect number of system messages."
                    self.log_debug(error_content)
                    return CreateResult(
                        finish_reason="error",
                        content=error_content,
                        usage=self._actual_usage,
                        cached=False,
                    )

                # Convert updated messages to LLMMessage for autogen processing
                llm_messages = convert_dict_to_llm(updated_messages)
                self.log_debug(f"Converted updated messages to LLMMessage: {llm_messages}")

                # Here you would interact with Autogen using llm_messages
                # For example purposes, we'll assume a synchronous interaction
                # Replace the following with actual Autogen interaction as needed

                # Simulate Autogen processing (Placeholder)
                # In real implementation, you might pass llm_messages to Autogen's agents
                # and receive responses back as LLMMessage objects

                # For demonstration, let's just convert back to dicts
                # and assume no further processing
                user_messages = convert_llm_to_dict(llm_messages)
                self.log_debug(f"Converted back to dictionaries after Autogen processing: {user_messages}")

                # Update usage statistics
                usage = RequestUsage(
                    prompt_tokens=0, completion_tokens=len(content.split())
                )
                self._actual_usage = usage
                self._total_usage.prompt_tokens += usage.prompt_tokens
                self._total_usage.completion_tokens += usage.completion_tokens

                self.log_debug(f"Usage statistics updated: {usage}")
                self.log_debug(f"Total usage so far: {self._total_usage}")

                return CreateResult(
                    finish_reason="stop", content=content, usage=usage, cached=False
                )

            # Update usage statistics
            usage = RequestUsage(
                prompt_tokens=0, completion_tokens=len(content.split())
            )
            self._actual_usage = usage
            self._total_usage.prompt_tokens += usage.prompt_tokens
            self._total_usage.completion_tokens += usage.completion_tokens

            self.log_debug(f"Usage statistics updated: {usage}")
            self.log_debug(f"Total usage so far: {self._total_usage}")

            return CreateResult(
                finish_reason="stop", content=content, usage=usage, cached=False
            )
        except Exception as e:
            self.log_debug(f"Error during initial request generation: {e}")
            content = f"Error during initial request generation: {str(e)}"
            return CreateResult(
                finish_reason="error",
                content=content,
                usage=self._actual_usage,
                cached=False,
            )
        finally:
            self.log_debug("Completed processing of generate_chat_completions request.")

    async def cleanup(self):
        """Perform cleanup tasks at the end of the workflow."""
        self.log_debug("Performing cleanup tasks.")
        # Reset usage statistics for all agents
        for agent in self.agents:
            if hasattr(agent, "model_client") and hasattr(agent.model_client, "reset"):
                agent.model_client.reset()
                self.log_debug(f"Reset usage statistics for agent: {agent.name}")
        self.log_debug("Cleanup completed.")

    # Mock event emitter to print output
    @staticmethod
    async def mock_event_emitter(event: Dict[str, Any]):
        print(f"Emitted Event: {json.dumps(event, indent=2)}")


def convert_dict_to_llm(messages: List[Dict[str, Any]]) -> List[LLMMessage]:
    """
    Convert a list of message dictionaries to LLMMessage instances.

    Args:
        messages (List[Dict[str, Any]]): List of message dictionaries.

    Returns:
        List[LLMMessage]: List of LLMMessage instances.
    """
    llm_messages = []
    for msg in messages:
        try:
            llm_msg = LLMMessage(**msg)
            llm_messages.append(llm_msg)
            logging.debug(f"Converted dict to LLMMessage: {llm_msg}")
        except ValidationError as ve:
            logging.debug(f"Validation error while converting dict to LLMMessage: {ve}")
    return llm_messages


def convert_llm_to_dict(messages: List[LLMMessage]) -> List[Dict[str, Any]]:
    """
    Convert a list of LLMMessage instances to message dictionaries.

    Args:
        messages (List[LLMMessage]): List of LLMMessage instances.

    Returns:
        List[Dict[str, Any]]: List of message dictionaries.
    """
    dict_messages = []
    for msg in messages:
        dict_msg = msg.dict(exclude_none=True)
        dict_messages.append(dict_msg)
        logging.debug(f"Converted LLMMessage to dict: {dict_msg}")
    return dict_messages


class MessageModel(BaseModel):
    role: str = Field(..., description="Role of the message sender.")
    content: str = Field(..., description="Content of the message.")


class Pipe:
    """The Pipe class manages agentic workflows with debugging, custom chat completions, and interaction limits."""

    class Valves(BaseModel):
        default_model_id: str = Field(
            default="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0",
            description="Default model ID for OpenWebUIChatCompletionClient.",
        )
        enable_llm_summaries: bool = Field(
            default=True, description="Enable LLM-based progress summaries."
        )
        use_collapsible: bool = Field(
            default=True,
            description="Enable collapsible UI for displaying model outputs.",
        )
        debug_valve: bool = Field(default=True, description="Enable debug logging.")
        max_interactions: int = Field(
            default=10,
            description="Maximum number of back-and-forth interactions between agents.",
        )
        # Valve for function definitions
        function_definitions: List[Dict[str, Any]] = Field(
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
        # Valve for function implementations
        function_implementations: Dict[str, Callable[[Dict[str, Any]], str]] = Field(
            default_factory=lambda: {
                "get_current_time": get_current_time,
                "echo": echo,
                # Add more function implementations here
            },
            description="Mapping of function names to their implementations.",
        )
        # Valve for default agent names
        default_agent_names: List[str] = Field(
            default=[
                "Neoblitz",
                "Byteforge",
                "Vexstrike",
                "Synthraid",
                "Pixelfang",
                "Mechstalk",
                "Tronslash",
                "Automstorm",
                "Matrixbane",
                "Robhawke",
            ],
            description="List of default agent names.",
        )
        # Valve for random agent name assignment
        assign_random_names: bool = Field(
            default=True,
            description="Enable random assignment of agent names from the default list.",
        )
        # Valve for pipeline types (only RoundRobin enabled)
        enable_round_robin: bool = Field(
            default=True,
            description="Enable RoundRobin pipeline.",
        )
        enable_swarm: bool = Field(
            default=False,
            description="Enable Swarm pipeline.",
        )
        # Valves for agent-specific system messages
        agent_system_messages: Dict[str, str] = Field(
            default={
                "Neoblitz": "You are Neoblitz, a dynamic AI agent focused on innovative AI discussions.",
                "Byteforge": "You are Byteforge, an AI agent specializing in AI research and development.",
                "Vexstrike": "You are Vexstrike, an AI agent dedicated to exploring advanced AI concepts.",
                "Synthraid": "You are Synthraid, an AI agent with expertise in synthetic intelligence applications.",
                "Pixelfang": "You are Pixelfang, an AI agent passionate about pixel-based AI solutions.",
                "Mechstalk": "You are Mechstalk, an AI agent skilled in mechanical AI interactions.",
                "Tronslash": "You are Tronslash, an AI agent proficient in Tron-based AI systems.",
                "Automstorm": "You are Automstorm, an AI agent focused on automation and AI synergy.",
                "Matrixbane": "You are Matrixbane, an AI agent specializing in matrix-based AI operations.",
                "Robhawke": "You are Robhawke, an AI agent with a keen interest in robotic AI integrations.",
                # Add more agent-specific system messages here
            },
            description="Mapping of agent names to their specific system messages.",
        )
        # Valve for treating 'User_Proxy' as an agent
        treat_user_as_agent: bool = Field(
            default=True,
            description="Treat 'User_Proxy' as an agent with a specific system message.",
        )
        user_agent_name: str = Field(
            default="User_Proxy",  # Changed from "User Proxy" to "User_Proxy"
            description="Name assigned to the user agent.",
        )
        user_agent_system_message: str = Field(
            default="You are User_Proxy, representing the user interacting with the AI agents.",  # Updated to match the new name
            description="System message for the user agent.",
        )

        @validator('user_agent_name')
        def validate_user_agent_name(cls, v):
            """
            Validate that the user_agent_name is a valid Python identifier.
            """
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', v):
                raise ValueError("The agent name must be a valid Python identifier.")
            return v

    def __init__(self):
        self.valves = self.Valves()
        self.start_time = None

        # Initialize interaction counter
        self.interaction_count = 0

        # Initialize functions from valves
        global available_functions, function_implementations
        available_functions = self.valves.function_definitions
        function_implementations = self.valves.function_implementations

        # Initialize agent names
        self.available_agent_names = self.valves.default_agent_names.copy()
        self.assigned_agent_names = []  # To keep track of assigned names

        # Initialize list to store agent references for cleanup
        self.agents: List[AssistantAgent] = []

        # Logging setup
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

    def log_debug(self, message: str):
        """Log debug messages."""
        if self.valves.debug_valve:
            self.log.debug(message)

    def _assign_agent_name(self) -> str:
        """
        Assigns a name to an agent based on the assign_random_names valve.

        Returns:
            str: The assigned agent name.
        """
        if self.valves.assign_random_names and self.available_agent_names:
            name = random.choice(self.available_agent_names)
            self.available_agent_names.remove(name)  # Ensure uniqueness
            self.assigned_agent_names.append(name)
            self.log_debug(f"Randomly assigned agent name: {name}")
            return name
        else:
            # Default naming convention if not assigning randomly
            agent_number = len(self.assigned_agent_names) + 1
            name = f"Agent_{agent_number}"
            self.assigned_agent_names.append(name)
            self.log_debug(f"Assigned default agent name: {name}")
            return name

    def validate_messages(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validates and converts a sequence of messages into message dictionaries.

        Args:
            messages (Sequence[Dict[str, Any]]): The input messages as dictionaries.

        Returns:
            List[Dict[str, Any]]: A list of valid message dictionaries.
        """
        validated_messages = []

        for idx, message in enumerate(messages):
            self.log_debug(f"Validating message at index {idx}: {message}")
            if not isinstance(message, dict):
                self.log_debug(f"Message at index {idx} is not a dictionary: {message}")
                continue  # Skip invalid messages

            role = message.get("role")
            content = message.get("content")

            self.log_debug(f"Extracted role: {role}, content: {content} from message at index {idx}")

            if not isinstance(role, str) or not isinstance(content, str):
                self.log_debug(f"Invalid message format at index {idx}: {message}")
                continue  # Skip invalid messages

            # Append the valid message dictionary
            validated_message = {"role": role, "content": content}
            validated_messages.append(validated_message)
            self.log_debug(f"Validated message {idx}: {validated_message}")

        self.log_debug(f"Total validated messages: {len(validated_messages)}")
        return validated_messages

    async def emit_message(self, role: str, content: str, __event_emitter__):
        """Helper method to emit messages."""
        self.log_debug(f"Emitting message from role: {role} with content: {content}")
        # Only wrap in collapsible if not 'User_Proxy' role and if collapsible is enabled
        if role != self.valves.user_agent_name and self.valves.use_collapsible:
            collapsible_content = f"""
<details>
    <summary>{html.escape(role)} says:</summary>
    <p>{html.escape(content)}</p>
</details>
""".strip()
        else:
            collapsible_content = (
                content.strip()
            )  # Direct content for 'User_Proxy' role or if collapsible is disabled

        self.log_debug(f"Formatted content to emit: {collapsible_content}")

        if __event_emitter__:
            self.log_debug("Calling __event_emitter__ with message.")
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": collapsible_content},
                }
            )
            self.log_debug("Message emitted successfully.")

    async def pipe(self, body: Optional[Dict[str, Any]] = None, __event_emitter__=None):
        """Runs a multi-agent chat workflow and emits interactions as collapsible UI elements."""
        try:
            self.start_time = asyncio.get_event_loop().time()
            self.log_debug("Pipe started.")

            # Emit initiating status
            if __event_emitter__:
                self.log_debug("Emitting initiation status.")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "Info",
                            "description": "Initiating agentic workflow.",
                            "done": False,
                        },
                    }
                )

            # Use the default model ID from valves
            default_model_id = self.valves.default_model_id
            self.log_debug(f"Using default model ID: {default_model_id}")

            # Assign names to agents
            agent_a_name = self._assign_agent_name()
            agent_b_name = self._assign_agent_name()

            # Retrieve system messages for agents
            agent_a_system_message = self.valves.agent_system_messages.get(agent_a_name, "")
            agent_b_system_message = self.valves.agent_system_messages.get(agent_b_name, "")

            self.log_debug(f"Agent A name: {agent_a_name}, system message: {agent_a_system_message}")
            self.log_debug(f"Agent B name: {agent_b_name}, system message: {agent_b_system_message}")

            # Handle 'User_Proxy' as an agent if valve is enabled
            if self.valves.treat_user_as_agent:
                user_agent_name = self.valves.user_agent_name
                user_agent_system_message = self.valves.user_agent_system_message

                self.log_debug(f"Initializing user agent with name: {user_agent_name}")
                user_agent = AssistantAgent(
                    name=user_agent_name,
                    model_client=OpenWebUIChatCompletionClient(
                        model=default_model_id,
                        functions=available_functions,
                        system_message=user_agent_system_message,
                        debug_valve=self.valves.debug_valve,
                    ),
                    tools=[],
                )
                self.log_debug(f"{user_agent_name} initialized with system message: {user_agent_system_message}")
                self.agents.append(user_agent)
            else:
                user_agent = None
                self.log_debug("User agent is not treated as an agent.")

            # Create two agents
            self.log_debug(f"Initializing agent A: {agent_a_name}")
            agent_a = AssistantAgent(
                name=agent_a_name,
                model_client=OpenWebUIChatCompletionClient(
                    model=default_model_id,
                    functions=available_functions,
                    system_message=agent_a_system_message,
                    debug_valve=self.valves.debug_valve,
                ),
                tools=[],
            )
            self.log_debug(f"{agent_a_name} initialized with system message: {agent_a_system_message}")
            self.agents.append(agent_a)

            self.log_debug(f"Initializing agent B: {agent_b_name}")
            agent_b = AssistantAgent(
                name=agent_b_name,
                model_client=OpenWebUIChatCompletionClient(
                    model=default_model_id,
                    functions=available_functions,
                    system_message=agent_b_system_message,
                    debug_valve=self.valves.debug_valve,
                ),
                tools=[],
            )
            self.log_debug(f"{agent_b_name} initialized with system message: {agent_b_system_message}")
            self.agents.append(agent_b)

            # Define termination condition
            termination = TextMentionTermination("STOP")
            self.log_debug("Termination condition set to 'STOP'.")

            # Configure the team
            team_agents = [agent_a, agent_b]
            if user_agent:
                team_agents.append(user_agent)
                self.log_debug("User agent included in the team_agents.")
            else:
                self.log_debug("User agent not included in the team_agents.")

            self.log_debug("Initializing RoundRobinGroupChat with team agents.")
            team = RoundRobinGroupChat(team_agents, termination_condition=termination)
            self.log_debug("Using RoundRobinGroupChat pipeline.")

            # Process incoming messages
            if body and 'messages' in body:
                raw_messages = body['messages']
                self.log_debug(f"Received raw messages: {raw_messages}")

                # Validate the message payload
                self.log_debug("Validating raw messages.")
                validated_messages = self.validate_messages(raw_messages)
                self.log_debug(f"Validated messages: {validated_messages}")

                if not validated_messages:
                    error_message = "No valid messages found in the payload."
                    self.log_debug(error_message)
                    if __event_emitter__:
                        self.log_debug("Emitting error status due to no valid messages.")
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
                    return  # Early exit

                # Call pop_system_message with validated messages (dictionaries)
                self.log_debug("Calling pop_system_message with validated messages.")
                system_message, user_messages = pop_system_message(validated_messages)
                self.log_debug(f"System message extracted: {system_message}")
                self.log_debug(f"User messages extracted: {user_messages}")

                # No need to convert system_message and user_messages to LLMMessage here
                # They remain as dictionaries

                # Ensure system_message is present
                if system_message and isinstance(system_message, dict):
                    self.log_debug("System message is a dictionary.")
                elif system_message is not None:
                    self.log_debug("System message is not a dictionary. Skipping.")
                    system_message = None  # No system message
                else:
                    self.log_debug("No system message found.")

                # Ensure user_messages are dictionaries
                if user_messages:
                    self.log_debug("Ensuring user messages are dictionaries.")
                    # Already dictionaries, no action needed
                else:
                    self.log_debug("No user messages found.")

                # Get message history excluding the last user message
                if len(user_messages) > 1:
                    message_history = user_messages[:-1]
                    self.log_debug(f"Message history: {message_history}")
                else:
                    message_history = []
                    self.log_debug("No message history to summarize.")

                # Generate a summary if required
                if self.valves.enable_llm_summaries and message_history:
                    self.log_debug("Generating summary of message history.")
                    summary_content = await self._generate_summary(message_history)
                    self.log_debug(f"Summary content generated: {summary_content}")
                else:
                    summary_content = "No previous conversation to summarize."
                    self.log_debug(f"Summary content: {summary_content}")

                # Define the task prompt
                if user_messages:
                    last_user_message = user_messages[-1].get("content", "").strip()
                    self.log_debug(f"Last user message extracted: {last_user_message}")
                else:
                    last_user_message = ""
                    self.log_debug("No user messages to extract last message from.")

                task = (
                    f"Summary of previous conversations: '{summary_content}'. "
                    f"User's instruction: '{last_user_message}'. "
                    f"{agent_a_name}, based on the user's instruction, ask {agent_b_name} a related question about AI. "
                    f"{agent_b_name}, respond and ask a follow-up question. "
                    f"{self.valves.user_agent_name}, provide feedback or ask additional questions as needed. "
                    "Continue the conversation until you decide to stop."
                )
                self.log_debug(f"Task defined: {task}")

            else:
                task = (
                    f"{agent_a_name}, ask {agent_b_name} a question about AI. "
                    f"{agent_b_name}, respond and ask a follow-up question. "
                    f"{self.valves.user_agent_name}, provide feedback or ask additional questions as needed. "
                    "Continue the conversation until you decide to stop."
                )
                self.log_debug(f"No messages provided. Default task: {task}")

            # Start the team task
            self.log_debug("Starting team run_stream with defined task.")
            stream = team.run_stream(task=task)
            self.log_debug("Team stream started.")

            async for message in stream:
                self.log_debug(f"Received message from stream: {message}")
                await self._process_stream_message(message, __event_emitter__)

            # Final cleanup
            self.log_debug("Completed processing all messages from stream.")
            await self._finalize_workflow(__event_emitter__)

        except Exception as e:
            error_message = str(e)
            self.log_debug(f"Error occurred: {error_message}")
            if __event_emitter__:
                self.log_debug("Emitting error status due to exception.")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "Error",
                            "description": f"An error occurred: {error_message}",
                            "done": True,
                        },
                    }
                )
            await self.cleanup()

    async def _generate_summary(self, message_history: List[Dict[str, Any]]) -> str:
        """Generates a summary of the message history."""
        self.log_debug("Entering _generate_summary method.")
        self.log_debug(f"Message history for summary: {message_history}")
        # Placeholder implementation
        # In real implementation, call generate_chat_completions or similar
        # For demonstration, return a static summary
        summary_content = "This is a summary of the previous messages."
        self.log_debug(f"Generated summary content: {summary_content}")
        return summary_content

    async def _process_stream_message(self, message, __event_emitter__):
        """Processes a message from the team stream."""
        self.log_debug(f"Processing message from stream: {message}")

        # Increment interaction counter
        self.interaction_count += 1
        self.log_debug(f"Interaction count incremented to: {self.interaction_count}")

        # Check if the interaction count has reached the maximum limit
        if self.interaction_count >= self.valves.max_interactions:
            self.log_debug("Max interactions reached. Terminating workflow.")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "Info",
                            "description": f"Agentic workflow terminated after reaching the maximum of {self.valves.max_interactions} interactions.",
                            "done": True,
                        },
                    }
                )
                self.log_debug("Termination status emitted.")
            # Stop processing further messages
            return

        # Handle TaskResult objects
        if hasattr(message, "messages") and isinstance(message.messages, list):
            self.log_debug("Message is a TaskResult with messages.")
            # Handle TaskResult object
            for sub_message in message.messages:
                role = getattr(sub_message, "source", "Unknown")
                content = getattr(sub_message, "content", "").strip()
                self.log_debug(
                    f"Processing sub-message: Type={type(sub_message)}, Role='{role}', Content='{content}'"
                )
                if content:
                    await self.emit_message(role, content, __event_emitter__)
            return  # Skip to the next message

        # Handle dictionary messages
        if isinstance(message, dict):
            self.log_debug("Message is a dictionary.")
            role = message.get("role", "Unknown")
            content = message.get("content", "").strip()
            self.log_debug(f"Processed dict message: Role='{role}', Content='{content}'")
        elif hasattr(message, "source") and hasattr(message, "content"):
            self.log_debug("Message is an object with source and content attributes.")
            # Handle TextMessage object
            role = message.source
            content = message.content.strip()
            self.log_debug(f"Processed TextMessage: Role='{role}', Content='{content}'")
        else:
            self.log_debug("Received an unexpected message format.")
            return  # Skip unexpected message formats

        if not content:
            self.log_debug("Empty content detected. Skipping emission.")
            return  # Skip empty messages

        self.log_debug(
            f"Processing message from role: '{role}' with content: '{content}'"
        )
        await self.emit_message(role, content, __event_emitter__)

    async def _finalize_workflow(self, __event_emitter__):
        """Performs finalization tasks after the workflow completes."""
        self.log_debug("Finalizing workflow.")
        # Calculate elapsed time
        elapsed_time = asyncio.get_event_loop().time() - self.start_time
        # Format elapsed time into minutes and seconds
        minutes, seconds = divmod(int(elapsed_time), 60)
        if minutes > 0:
            time_formatted = f"{minutes} minute(s) and {seconds} second(s)"
        else:
            time_formatted = f"{seconds} second(s)"
        final_status = f"Agentic workflow completed in {time_formatted}."
        self.log_debug(f"Final status: {final_status}")

        if __event_emitter__:
            self.log_debug("Emitting final status.")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "level": "Info",
                        "description": final_status,
                        "done": True,
                    },
                }
            )
            self.log_debug("Final status emitted.")

        # Perform cleanup task
        await self.cleanup()

    async def cleanup(self):
        """Perform cleanup tasks at the end of the workflow."""
        self.log_debug("Performing cleanup tasks.")
        # Reset usage statistics for all agents
        for agent in self.agents:
            if hasattr(agent, "model_client") and hasattr(agent.model_client, "reset"):
                agent.model_client.reset()
                self.log_debug(f"Reset usage statistics for agent: {agent.name}")
        self.log_debug("Cleanup completed.")

    # Mock event emitter to print output
    @staticmethod
    async def mock_event_emitter(event: Dict[str, Any]):
        print(f"Emitted Event: {json.dumps(event, indent=2)}")


# Helper Functions for Conversions

def convert_dict_to_llm(messages: List[Dict[str, Any]]) -> List[LLMMessage]:
    """
    Convert a list of message dictionaries to LLMMessage instances.

    Args:
        messages (List[Dict[str, Any]]): List of message dictionaries.

    Returns:
        List[LLMMessage]: List of LLMMessage instances.
    """
    llm_messages = []
    for msg in messages:
        try:
            llm_msg = LLMMessage(**msg)
            llm_messages.append(llm_msg)
            logging.debug(f"Converted dict to LLMMessage: {llm_msg}")
        except ValidationError as ve:
            logging.debug(f"Validation error while converting dict to LLMMessage: {ve}")
    return llm_messages


def convert_llm_to_dict(messages: List[LLMMessage]) -> List[Dict[str, Any]]:
    """
    Convert a list of LLMMessage instances to message dictionaries.

    Args:
        messages (List[LLMMessage]): List of LLMMessage instances.

    Returns:
        List[Dict[str, Any]]: List of message dictionaries.
    """
    dict_messages = []
    for msg in messages:
        dict_msg = msg.dict(exclude_none=True)
        dict_messages.append(dict_msg)
        logging.debug(f"Converted LLMMessage to dict: {dict_msg}")
    return dict_messages


# Run the pipe with the multi-agent workflow
async def run_pipe():
    pipe = Pipe()
    # Example body with user messages
    body = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you! How can I assist you today?"},
            {"role": "user", "content": "Can you explain how AI can improve healthcare?"}
        ]
    }
    await pipe.pipe(body=body, __event_emitter__=Pipe.mock_event_emitter)


# Check if running in an existing event loop and handle accordingly
try:
    asyncio.run(run_pipe())
except RuntimeError as e:
    if "asyncio.run() cannot be called from a running event loop" in str(e):
        loop = asyncio.get_event_loop()
        if loop and loop.is_running():
            task = loop.create_task(run_pipe())
    else:
        raise
