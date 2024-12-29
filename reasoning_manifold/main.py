"""
title: Open-WebUI Reasoning Manifold
version: 0.5.0

- [x] Updated to work on OWUI 0.4.x
- [ ] Updated to work on OWUI 0.5.x
 - [x] Wrapper functions to support new signatures
 - [ ] Fix ERROR [asyncio] Unclosed client session (OWUI bug?)
 - [ ] Fix collapsible in follow up queries.
- [x] OpenAI streaming
- [x] Ollama streaming
- [x] Thought expansion
- [x] LLM summary generation
- [x] Advanced parsing logic
- [x] Optimised emission timing
- [x] Valve to append system prompt to every request
- [ ] Support summary models that have connection prefix eg 'ollama.' (OWUI bug?)
"""

import os
import json
import time
import asyncio
import importlib
import re
from datetime import datetime
from typing import List, Union, Optional, Callable, Awaitable, Dict, Any, Tuple
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import StreamingResponse

from open_webui.config import TASK_MODEL, TASK_MODEL_EXTERNAL

import logging

THOUGHT_SUMMARY_PLACEHOLDER = "your-task-model-id-goes-here"
SYSTEM_MESSAGE_DEFAULT = "Respond using <Thought> and <Output> tags."

# Initialize logging
logger = logging.getLogger("PipeWrapper")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all debug messages


async def _get_version_info() -> Dict[str, str]:
    """
    Fetch the OWUI version information.
    """
    try:
        module = importlib.import_module("open_webui.main")
        get_app_version = getattr(module, "get_app_version", None)
        if not get_app_version:
            logger.debug("get_app_version not found in open_webui.main")
            return {"version": "0.0.0"}
        version_info = await get_app_version()  # e.g., {"version": "0.4.8c"}
        logger.debug(f"Fetched OWUI version: {version_info.get('version', '0.0.0')}")
        return version_info
    except Exception as e:
        logger.error(f"Error fetching version info: {e}")
        return {"version": "0.0.0"}


def is_version_0_5_or_greater(version_info: Dict[str, str]) -> bool:
    """
    Determine if the OWUI version is 0.5 or greater.

    Args:
        version_info (Dict[str, str]): The version information dictionary.

    Returns:
        bool: True if version is 0.5 or greater, False otherwise.
    """
    version_str = version_info.get("version", "0.0.0")
    match = re.match(r"(\d+)\.(\d+)", version_str)
    if not match:
        logger.debug(
            f"Version parsing failed for version string: '{version_str}'. Assuming version is below 0.5."
        )
        return False
    major, minor = map(int, match.groups())
    logger.debug(f"Parsed OWUI version: major={major}, minor={minor}")
    return (major, minor) >= (0, 5)

async def _pop_system_message(
    body_messages: List[dict],
) -> Tuple[Optional[dict], List[dict]]:
    """
    Wrapper that calls the correct pop_system_message function based on OWUI version.
    """
    version_info = await _get_version_info()
    # Assuming 'pop_system_message' hasn't moved, but if it has, handle accordingly
    try:
        module = importlib.import_module("open_webui.utils.misc")
        pop_system_message = getattr(module, "pop_system_message", None)
        if pop_system_message:
            logger.debug("Using 'pop_system_message' from 'open_webui.utils.misc'.")
            return pop_system_message(body_messages)
        else:
            raise ImportError(
                "Unable to import 'pop_system_message' from 'open_webui.utils.misc'"
            )
    except ImportError as e:
        logger.error(e)
        raise


async def _add_or_update_system_message(message: str, messages: List[dict]) -> None:
    """
    Wrapper that calls the correct add_or_update_system_message function based on OWUI version.
    """
    version_info = await _get_version_info()
    # Assuming 'add_or_update_system_message' hasn't moved, but if it has, handle accordingly
    try:
        module = importlib.import_module("open_webui.utils.misc")
        add_or_update_system_message = getattr(
            module, "add_or_update_system_message", None
        )
        if add_or_update_system_message:
            logger.debug(
                "Using 'add_or_update_system_message' from 'open_webui.utils.misc'."
            )
            add_or_update_system_message(message, messages)
        else:
            raise ImportError(
                "Unable to import 'add_or_update_system_message' from 'open_webui.utils.misc'"
            )
    except ImportError as e:
        logger.error(e)
        raise


class Pipe:
    """
    The Pipe class handles streaming responses from OpenAI or Ollama models,
    processes internal thought tags, manages output parsing, generates summaries,
    and ensures clean and user-friendly message emission without exposing internal tags.
    """

    def _get_complete_user(self) -> dict:
        """
        Add default values for missing user fields and return a complete dictionary.
        Ensures datetime fields are Unix timestamps (int).
        """
        now = int(datetime.now().timestamp())  # Current Unix timestamp
        return {
            **self.user,
            "profile_image_url": self.user.get("profile_image_url", ""),
            "last_active_at": self._convert_to_timestamp(self.user.get("last_active_at", now)),
            "updated_at": self._convert_to_timestamp(self.user.get("updated_at", now)),
            "created_at": self._convert_to_timestamp(self.user.get("created_at", now)),
        }

    @staticmethod
    def _convert_to_timestamp(value):
        """
        Convert a value to Unix timestamp (int).
        If the value is already an int, return as-is.
        If it's a string (ISO format), parse and convert.
        """
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            try:
                # Parse ISO 8601 string and convert to timestamp
                dt = datetime.fromisoformat(value)
                return int(dt.timestamp())
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {value}")
        else:
            raise TypeError(f"Unsupported type for timestamp conversion: {type(value)}")

    async def _chat_completion(self, form_data: dict) -> dict:
        version_info = await _get_version_info()
        if is_version_0_5_or_greater(version_info):
            try:
                from open_webui.models.users import UserModel
                module = importlib.import_module("open_webui.main")
                chat_completion = getattr(module, "chat_completion", None)
                if chat_completion:
                    logger.debug("Using 'chat_completion' for OWUI 0.5.x.")
                    user_model = UserModel(**self._get_complete_user())
                    return await chat_completion(request=self.request, form_data=form_data, user=user_model)
                else:
                    raise ImportError("Function 'chat_completion' not found in Open-WebUI main module")
            except ImportError as e:
                logger.error(f"ImportError: {e}")
                raise
        else:
            # OWUI 0.4.x fallback
            try:
                module = importlib.import_module("open_webui.main")
                generate_chat_completions = getattr(module, "generate_chat_completions", None)
                if generate_chat_completions:
                    logger.debug("Using 'generate_chat_completions' for OWUI 0.4.x.")
                    return await generate_chat_completions(form_data=form_data, user=self.user, bypass_filter=True)
                else:
                    raise ImportError("Unable to import 'generate_chat_completions' from 'open_webui.main'")
            except ImportError as e:
                logger.error(f"ImportError: {e}")
                raise

    async def _non_stream_completion(self, form_data: dict) -> Optional[str]:
        version_info = await _get_version_info()
        try:
            module = importlib.import_module("open_webui.main")
            if is_version_0_5_or_greater(version_info):
                from open_webui.models.users import UserModel
                chat_completion = getattr(module, "chat_completion", None)
                if chat_completion:
                    logger.debug("Using 'chat_completion' for OWUI 0.5.x (non-streaming).")
                    user_model = UserModel(**self._get_complete_user())
                    response = await chat_completion(request=self.request, form_data=form_data, user=user_model)
            else:
                generate_chat_completions = getattr(module, "generate_chat_completions", None)
                if generate_chat_completions:
                    logger.debug("Using 'generate_chat_completions' for OWUI 0.4.x (non-streaming).")
                    response = await generate_chat_completions(form_data=form_data, user=self.user, bypass_filter=True)

            # Extract summary
            if response and "choices" in response:
                content = response["choices"][0].get("message", {}).get("content", "").strip()
                if content:
                    logger.debug(f"Generated summary: {content}")
                    return content
            logger.debug("Response structure invalid or no summary generated.")
            return None
        except Exception as e:
            logger.error(f"Error in _non_stream_completion: {e}")
            return None

    class Valves(BaseModel):
        """
        Configuration for the Open-WebUI Reasoning Manifold.
        """

        # Model Configuration (Renamed from model_ids to reasoning_model_ids)
        reasoning_model_ids: str = Field(
            default="marco-o1:latest",
            description="Comma-separated list of model IDs (not names) to be used by the manifold.",
        )
        manifold_prefix: str = Field(
            default="reason/",
            description="Prefix used for model names.",
        )
        streaming_enabled: bool = Field(
            default=True, description="Enable or disable streaming for responses."
        )

        # Thought Handling Configuration
        thought_tag: str = Field(
            default="Thought", description="The XML tag for internal thoughts."
        )
        output_tag: str = Field(
            default="Output", description="The XML tag for final output."
        )
        # Status Update Configuration
        use_collapsible: bool = Field(
            default=False,
            description="Collapsible UI to reveal generated thoughts.",
        )
        enable_llm_summaries: bool = Field(
            default=False,
            description="Enable LLM-generated summaries for updates.",
        )
        thought_summary_model_id: str = Field(
            default=THOUGHT_SUMMARY_PLACEHOLDER,
            description=(
                "Model ID used to generate thought summaries. "
                "Not the name of the model."
            ),
        )
        thought_summary_interval_words: int = Field(
            default=20,
            description="Number of words after which to generate a thought summary.",
        )
        thought_summary_timeout: float = Field(
            default=15.0,
            description="Maximum time in seconds to wait for a summary before proceeding.",
        )
        dynamic_status_prompt: str = Field(
            default="Summarize the current thought process in exactly 4 words.",
            description="Prompt for generating LLM summaries.",
        )
        dynamic_status_system_prompt: str = Field(
            default="You are a helpful assistant summarizing thoughts concisely, only respond with the summary.",
            description="System prompt for dynamic status generation.",
        )
        emit_interval: float = Field(
            default=3.0,
            description="Interval in seconds between status updates.",
        )

        # System Message Override Configuration
        system_message_override_enabled: bool = Field(
            default=True,
            description=(
                "Enable system message override with a custom message. "
                "If set to False, the original system message is retained."
            ),
        )
        system_message_override: str = Field(
            default=SYSTEM_MESSAGE_DEFAULT,
            description=(
                "The system message to use if system_message_override_enabled is True."
            ),
        )

        append_system_prompt: bool = Field(
            default=False,
            description=(
                "If True, appends the system message to every user request, enclosed in parentheses."
            ),
        )

        # Debugging
        debug_valve: bool = Field(default=False, description="Enable debug logging.")

    def reset_state(self):
        """Reset state variables for reuse in new requests."""
        # Safely reset essential state variables
        self.thought_buffer = ""  # Accumulate thoughts for collapsible sections
        self.output_buffer = ""  # Accumulate emitted output tokens
        self.word_count_since_last_summary = 0  # Word count tracker for summaries
        self.start_time = time.time()  # Reset start time for duration tracking
        self.tracked_tokens = []  # Track tokens for dynamic summaries
        self.stop_emitter.clear()  # Ensure emitter can be stopped cleanly
        self.final_status_emitted = False  # Reset final status emission flag
        self.summary_in_progress = False  # Summary generation flag
        self.mode = "buffer_parsing"  # Default operational mode

        # **Reset last_emit_time to allow immediate emission**
        self.last_emit_time = 0

        # Log non-reset fields for safety verification
        self.log_debug(f"[RESET_STATE] Buffer retained: {self.buffer}")
        self.log_debug(f"[RESET_STATE] Messages retained: {self.messages}")
        self.log_debug(
            f"[RESET_STATE] Output buffer tokens retained: {self.output_buffer_tokens}"
        )

    def __init__(self):
        """
        Initialize the Pipe with default valves and necessary state variables.
        """
        self.type = "manifold"
        self.id = "reason"
        self.valves = self.Valves()
        self.name = self.valves.manifold_prefix
        self.stop_emitter = asyncio.Event()

        # State Variables
        self.thought_buffer = ""
        self.output_buffer = ""
        self.word_count_since_last_summary = 0
        self.start_time = None
        self.buffer = ""  # Initialize the buffer
        self.tracked_tokens = []  # Track tokens for dynamic summaries
        self.last_summary_time = time.time()  # Last time a summary was emitted
        self.determined = False
        self.inside_thought = False
        self.inside_output = False
        self.tags_detected = False  # Initialize tags_detected
        self.summary_in_progress = False  # Flag to indicate summary generation
        self.llm_tasks = []  # List to track LLM tasks
        self.final_status_emitted = False  # Flag to indicate if final status is emitted

        self.messages = []  # Store the messages list
        self.thought_tags = []  # List of thought tags per model
        self.output_tags = []  # List of output tags per model

        # Initialize current tags
        self.current_thought_tag = (
            self.valves.thought_tag
        )  # Default to valve's thought_tag
        self.current_output_tag = (
            self.valves.output_tag
        )  # Default to valve's output_tag

        # Setup Logging
        self.log = logger  # Use the initialized logger
        # The logger is already set up with handlers and level in the wrapper functions

        # Parse tags based on reasoning_model_ids
        self.parse_tags()

        # Operational Mode
        self.mode = "buffer_parsing"  # Modes: buffer_parsing, thought_parsing, output_parsing, standard_streaming

        # Buffer for Output tag detection
        self.output_buffer_tokens = (
            []
        )  # Buffer to hold last few tokens for Output tag detection
        self.output_buffer_limit = 5  # Number of tokens to buffer for tag detection

        # **Initialize last_emit_time to 0**
        self.last_emit_time = 0

        self.reset_state()

    def log_debug(self, message: str):
        """Log debug messages if debugging is enabled."""
        if self.valves.debug_valve:
            self.log.debug(message)

    def parse_tags(self):
        """
        Parse the thought_tag and output_tag fields into lists matching the reasoning_model_ids.
        Supports both singular tags (applied to all models) and comma-separated lists.
        """
        reasoning_model_ids = [
            m.strip() for m in self.valves.reasoning_model_ids.split(",") if m.strip()
        ]
        model_count = len(reasoning_model_ids)
        self.log_debug(f"Parsing tags for {model_count} models.")

        # Parse thought_tags
        thought_tags = [tag.strip() for tag in self.valves.thought_tag.split(",")]
        if len(thought_tags) == 1:
            self.thought_tags = thought_tags * model_count
        elif len(thought_tags) == model_count:
            self.thought_tags = thought_tags
        else:
            self.log_debug(
                f"[TAG_ERROR] Number of thought_tags ({len(thought_tags)}) does not match number of reasoning_model_ids ({model_count}). "
                f"Defaulting all thought_tags to '{thought_tags[0]}'"
            )
            self.thought_tags = [thought_tags[0]] * model_count

        self.log_debug(f"Parsed thought_tags: {self.thought_tags}")

        # Parse output_tags
        output_tags = [tag.strip() for tag in self.valves.output_tag.split(",")]
        if len(output_tags) == 1:
            self.output_tags = output_tags * model_count
        elif len(output_tags) == model_count:
            self.output_tags = output_tags
        else:
            self.log_debug(
                f"[TAG_ERROR] Number of output_tags ({len(output_tags)}) does not match number of reasoning_model_ids ({model_count}). "
                f"Defaulting all output_tags to '{output_tags[0]}'"
            )
            self.output_tags = [output_tags[0]] * model_count

        self.log_debug(f"Parsed output_tags: {self.output_tags}")

    def get_models(self) -> List[dict]:
        """List of models derived from the valve configuration."""
        reasoning_model_ids = [
            model_id.strip()
            for model_id in self.valves.reasoning_model_ids.split(",")
            if model_id.strip()
        ]
        return [{"id": model_id, "name": model_id} for model_id in reasoning_model_ids]

    def pipes(self) -> List[dict]:
        """Return the list of model pipes."""
        return self.get_models()

    def get_summary_model_id(self) -> str:
        """
        Determine the model ID to use for generating summaries.
        Returns the `thought_summary_model_id` if specified and not default; otherwise, returns an empty string.
        """
        if (
            self.valves.thought_summary_model_id
            and self.valves.thought_summary_model_id.lower()
            != THOUGHT_SUMMARY_PLACEHOLDER.lower()
        ):
            self.log_debug(
                f"[SUMMARY_CHECK] Using model ID for summary: {self.valves.thought_summary_model_id}"
            )
            return self.valves.thought_summary_model_id
        else:
            self.log_debug(f"[SUMMARY_CHECK] Thought summary model id not configured.")
            return ""

    # Strip out <details> tags from message content
    def clean_message_content(self, content: str) -> str:
        """
        Remove <details> HTML tags and their contents from the message content.
        """
        details_pattern = r"<details>\s*<summary>.*?</summary>\s*.*?</details>"
        return re.sub(details_pattern, "", content, flags=re.DOTALL).strip()

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __request__: Optional[Request] = None,
    ) -> Union[str, StreamingResponse]:
        """
        Main handler for processing requests.

        Steps:
        1. Reset state variables for each new request.
        2. Extract and process the model ID.
        3. Prepare the payload for the model.
        4. Handle streaming or non-streaming responses based on configuration.
        5. Manage errors gracefully.

        Args:
            body (dict): The incoming request payload.
            __user__ (Optional[dict]): The user information.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            __request__ (Optional[Request]): The request object.

        Returns:
            Union[str, StreamingResponse]: The response to be sent back.
        """
        # Reset state variables for each new request
        self.reset_state()

        self.user = __user__
        self.request = __request__

        self.client_session = aiohttp.ClientSession() # Workaround for OWUI 0.5.x session management

        try:
            # Extract model ID and clean prefix by stripping up to the first dot
            model_id = body.get("model", "")
            self.log_debug(f"Original model_id: {model_id}")

            if "." in model_id:
                model_id = model_id.split(".", 1)[1]
                self.log_debug(f"Stripped model_id: {model_id}")
            else:
                self.log_debug(
                    f"model_id does not contain a dot and remains unchanged: {model_id}"
                )

            # Determine the summary model ID dynamically, considering valve overrides
            summary_model_id = self.get_summary_model_id()
            self.log_debug(f"Resolved summary_model_id: {summary_model_id}")

            # Handle system messages if present (assumes system messages are separated)
            body_messages = body.get("messages", [])
            system_message, messages = await _pop_system_message(body_messages)

            # Append system prompt to user messages if the valve is enabled
            if self.valves.append_system_prompt:
                system_prompt = (
                    self.valves.system_message_override
                    if self.valves.system_message_override_enabled
                    else (system_message.get("content") if system_message else "")
                )
                for message in messages:
                    if message["role"] == "user":
                        original_content = message.get("content", "")
                        message["content"] = f"{original_content} ({system_prompt})"
                        self.log_debug(
                            f"[APPEND_SYSTEM_PROMPT] Modified user message: {message['content']}"
                        )

            # Handle system message override or pass-through
            if self.valves.system_message_override_enabled:
                # Append the overridden system message to the beginning of the list
                await _add_or_update_system_message(
                    self.valves.system_message_override, messages
                )
                self.log_debug(
                    f"Overriding system message with system_message_override: {self.valves.system_message_override}"
                )
            elif system_message:
                # Append the provided system message to the beginning of the list
                system_message_content = system_message.get("content", "")
                await _add_or_update_system_message(system_message_content, messages)
                self.log_debug(
                    f"Using provided system message: {system_message_content}"
                )

            # Log the total number of messages after handling system message
            self.log_debug(
                f"Total number of messages after processing system message: {len(messages)}"
            )

            self.messages = messages  # Store messages for later modification

            # Preprocess messages to clean content only for assistant role
            processed_messages = [
                {
                    "role": message["role"],
                    "content": (
                        self.clean_message_content(message.get("content", ""))
                        if message["role"] == "assistant"
                        else message.get("content", "")
                    ),
                }
                for message in messages
            ]

            # Prepare payload using 'messages' everywhere
            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": 500,  # Adjust as needed
                "temperature": 0.7,  # Adjust as needed
                "stream": self.valves.streaming_enabled,  # Ensure streaming is enabled based on configuration
            }

            self.log_debug(f"Payload prepared for model '{model_id}': {payload}")

            if self.valves.streaming_enabled:
                self.log_debug("Streaming is enabled. Handling stream response.")
                response = await self.stream_response(
                    payload, __event_emitter__
                )
            else:
                self.log_debug("Streaming is disabled. Handling non-stream response.")
                summary = await self.non_stream_response(
                    payload, __event_emitter__
                )
                response = summary  # Since non_stream_response now returns the summary directly

            self.log_debug("Response handling completed.")
            return response

        except json.JSONDecodeError as e:
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__, "Error", f"JSON Decode Error: {str(e)}", True
                )
            self.log_debug(f"JSON Decode Error in pipe method: {e}")
            return f"JSON Decode Error: {e}"
        except Exception as e:
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__, "Error", f"Error: {str(e)}", True
                )
            self.log_debug(f"Error in pipe method: {e}")
            return f"Error: {e}"
        finally:
            await self.client_session.close() # Workaround for OWUI 0.5.x session management

    async def stream_response(
        self, payload: dict, __event_emitter__
    ) -> StreamingResponse:
        """
        Handle streaming responses using the appropriate chat completion wrapper.

        Args:
            payload (dict): The payload to send to the model.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.

        Returns:
            StreamingResponse: The streaming response to be sent back.
        """

        async def response_generator():
            self.log_debug(
                "[STREAM_RESPONSE] Entered response_generator for stream_response."
            )

            try:
                # Ensure streaming is enabled in the payload
                payload["stream"] = True
                self.log_debug(
                    f"[STREAM_RESPONSE] Payload for generate_chat_completions: {payload}"
                )

                # Call the appropriate chat completion wrapper
                stream_response = await self._chat_completion(
                    form_data=payload
                )
                self.log_debug(
                    "[STREAM_RESPONSE] _chat_completion called successfully."
                )

                # Handle the streaming response
                # Assuming _chat_completion returns a StreamingResponse
                if isinstance(stream_response, StreamingResponse):
                    self.log_debug(
                        "[STREAM_RESPONSE] Received StreamingResponse from _chat_completion."
                    )

                    # Iterate over the body_iterator of the StreamingResponse
                    async for chunk in stream_response.body_iterator:
                        try:
                            # Process the chunk based on its type (bytes or str)
                            if isinstance(chunk, bytes):
                                chunk_str = chunk.decode("utf-8").rstrip(
                                    "\n"
                                )  # Remove trailing newline
                            elif isinstance(chunk, str):
                                chunk_str = chunk.rstrip(
                                    "\n"
                                )  # Remove trailing newline
                            else:
                                self.log_debug(
                                    f"[STREAM_RESPONSE] Unexpected chunk type: {type(chunk)}"
                                )
                                continue

                            self.log_debug(
                                f"[STREAM_RESPONSE] Raw chunk processed: {chunk_str}"
                            )

                            # Skip empty chunks
                            if not chunk_str:
                                self.log_debug(
                                    "[STREAM_RESPONSE] Empty chunk received. Skipping."
                                )
                                continue

                            # SSE format starts with 'data: '
                            if chunk_str.startswith("data:"):
                                data_str = chunk_str[5:].lstrip()
                                self.log_debug(
                                    f"[STREAM_RESPONSE] Extracted data_str: {data_str}"
                                )

                                # Handle the "[DONE]" marker
                                if data_str == "[DONE]":
                                    self.log_debug(
                                        "[STREAM_RESPONSE] Received [DONE]. Finalizing stream."
                                    )

                                    # Finalize any remaining buffer content
                                    if self.buffer:
                                        if not self.tags_detected:
                                            # No tags detected, emit the remaining buffer in standard_streaming mode
                                            self.log_debug(
                                                "[STREAM_FINALIZE] No tags detected. Dumping remaining buffer."
                                            )
                                            await self.emit_output(
                                                __event_emitter__, self.buffer.strip()
                                            )
                                            self.buffer = ""
                                        else:
                                            # Tags detected, process remaining buffer
                                            self.log_debug(
                                                f"[STREAM_FINALIZE] Tags detected. Finalizing remaining buffer: {self.buffer}"
                                            )
                                            await self.process_streaming_data(
                                                "",  # No new data
                                                __event_emitter__,
                                                is_final_chunk=True,
                                            )

                                    # Emit any remaining thought content
                                    if (
                                        self.thought_buffer
                                        and self.mode == "thought_parsing"
                                    ):
                                        self.log_debug(
                                            f"[STREAM_RESPONSE] Emitting remaining thought content: {self.thought_buffer}"
                                        )
                                        if self.valves.use_collapsible:
                                            await self.emit_collapsible(
                                                __event_emitter__
                                            )
                                        self.thought_buffer = ""

                                    # Finalize Output Parsing Mode if active
                                    if self.mode == "output_parsing":
                                        await self.finalize_output(__event_emitter__)

                                    break

                                # Parse the JSON content
                                try:
                                    chunk_data = json.loads(data_str)
                                    self.log_debug(
                                        f"[STREAM_RESPONSE] Parsed chunk data: {chunk_data}"
                                    )
                                except json.JSONDecodeError as parse_error:
                                    self.log_debug(
                                        f"[STREAM_RESPONSE] Failed to parse chunk: {chunk_str}. Error: {parse_error}"
                                    )
                                    continue

                                # Extract content from the chunk
                                data = (
                                    chunk_data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                self.log_debug(
                                    f"[STREAM_RESPONSE] Extracted data from chunk: {data}"
                                )

                                # Process the data if it exists
                                if data:
                                    # Track tokens for dynamic summaries
                                    self.tracked_tokens.append(data)
                                    self.log_debug(
                                        f"[STREAM_RESPONSE] Appended data to tracked_tokens: {data}"
                                    )

                                    # Increment token count based on word count
                                    word_count = len(data.split())
                                    self.word_count_since_last_summary += word_count
                                    self.log_debug(
                                        f"[TOKEN_COUNT] Tokens since last summary: {self.word_count_since_last_summary}"
                                    )

                                    # Detect tags and set tags_detected flag
                                    await self.process_streaming_data(
                                        data, __event_emitter__
                                    )

                                    # **Trigger summary if token count exceeds interval**
                                    if (
                                        self.word_count_since_last_summary
                                        >= self.valves.thought_summary_interval_words
                                    ):
                                        # Check if a summary is already in progress
                                        if not self.summary_in_progress:
                                            self.log_debug(
                                                f"[TOKEN_SUMMARY] Reached {self.valves.thought_summary_interval_words} tokens. Generating summary."
                                            )
                                            # Create a task for summary generation and track it
                                            summary_task = asyncio.create_task(
                                                self.trigger_summary(__event_emitter__)
                                            )
                                            self.llm_tasks.append(summary_task)
                                        else:
                                            self.log_debug(
                                                "[TOKEN_SUMMARY] Summary already in progress. Resetting word count."
                                            )
                                            # Reset the counter if a summary is already being generated
                                            self.word_count_since_last_summary = 0

                            else:
                                # Handle unexpected data format
                                self.log_debug(
                                    f"[STREAM_RESPONSE] Unexpected chunk format: {chunk_str}"
                                )

                        except Exception as e:
                            # Handle errors in processing chunks
                            self.log.debug(
                                f"[STREAM_RESPONSE] Error processing chunk: {chunk_str}. Error: {e}"
                            )
                            continue

                else:
                    self.log_debug(
                        "[STREAM_RESPONSE] Expected a StreamingResponse but got something else."
                    )

                # Yield nothing as we are managing emissions via event emitter
                return

            except Exception as e:
                # Handle errors in the response generator
                self.log.debug(f"[STREAM_RESPONSE] Error in response_generator: {e}")
                yield f'data: {{"error": "{str(e)}"}}\n\n'

            self.log_debug("[STREAM_RESPONSE] Exiting response_generator.")

        self.log_debug(
            "[STREAM_RESPONSE] Returning StreamingResponse from stream_response."
        )
        return StreamingResponse(response_generator(), media_type="text/event-stream")

    async def trigger_summary(self, __event_emitter__) -> None:
        """
        Trigger summary generation based on tracked tokens.
        Ensures that only one summary is generated at a time to prevent race conditions.
        Applies a timeout to prevent indefinite waiting.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        if self.summary_in_progress:
            self.log_debug(
                "[TRIGGER_SUMMARY] Summary already in progress. Skipping new summary trigger."
            )
            return

        if self.final_status_emitted:
            self.log_debug(
                "[SUMMARY] Final status already emitted. Skipping summary generation."
            )
            return

        if not self.valves.enable_llm_summaries:
            self.log_debug("[TRIGGER_SUMMARY] LLM summaries are disabled.")
            return

        if not self.tags_detected:
            self.log_debug("[TRIGGER_SUMMARY] No tags detected.")
            return

        # Set the flag to indicate a summary is in progress
        self.summary_in_progress = True
        try:
            # Use asyncio.wait_for to apply a timeout to the summary generation
            self.log_debug(
                f"[TRIGGER_SUMMARY] Starting summary generation with timeout of {self.valves.thought_summary_timeout} seconds."
            )
            summary = await asyncio.wait_for(
                self.generate_summary(), timeout=self.valves.thought_summary_timeout
            )

            if summary:
                # Ensure we are in thought_parsing mode
                if self.mode != "thought_parsing":
                    self.log_debug(
                        f"[TRIGGER_SUMMARY] Not in thought_parsing mode (current mode: {self.mode}). Skipping summary generation."
                    )
                    return
                # Emit the summary as a status update
                await self.emit_status(__event_emitter__, "Info", summary, False)
                # Reset token count and tracked tokens after summary
                self.tracked_tokens = []
                self.word_count_since_last_summary = 0
                self.log_debug(
                    "[TRIGGER_SUMMARY] Cleared tracked_tokens and reset word_count_since_last_summary after summary."
                )

            else:
                self.log_debug("[TRIGGER_SUMMARY] No summary generated.")

        except asyncio.TimeoutError:
            self.log_debug(
                f"[TRIGGER_SUMMARY] Summary generation timed out after {self.valves.thought_summary_timeout} seconds."
            )
            # Do not emit a timeout status to the user
            # Just log the timeout and proceed
        except Exception as e:
            self.log_debug(f"[TRIGGER_SUMMARY] Error during summary generation: {e}")
        finally:
            # Reset the flag
            self.summary_in_progress = False

    async def generate_summary(self) -> Optional[str]:
        """
        Generate a summary of the accumulated thoughts using an LLM.

        Returns:
            Optional[str]: The generated summary if successful; otherwise, None.
        """
        if not self.tracked_tokens:
            self.log_debug(
                "[GENERATE_SUMMARY] No thoughts collected. Skipping summary generation."
            )
            return None

        summary_model_id = self.get_summary_model_id()
        if not summary_model_id:
            self.log_debug("[GENERATE_SUMMARY] Summary model ID not configured.")
            return None

        # Prepare payload using 'messages' everywhere
        payload = {
            "model": summary_model_id,
            "messages": [
                {
                    "role": "system",
                    "content": self.valves.dynamic_status_system_prompt,
                },
                {
                    "role": "user",
                    "content": f"{self.valves.dynamic_status_prompt} {' '.join(self.tracked_tokens)}",
                },
            ],
            "max_tokens": 20,
            "temperature": 0.5,
        }

        self.log_debug(f"[GENERATE_SUMMARY] Generating summary with payload: {payload}")
        
        try:
            # Using the non-streaming chat completion wrapper which returns the summary directly
            result = await self._non_stream_completion(
                form_data=payload
            )
            if result:
                self.log_debug(f"[SUMMARY_GENERATED] Generated summary: {result}")
                return result
            else:
                self.log_debug("[GENERATE_SUMMARY] _non_stream_completion returned None.")
                return None
        except Exception as e:
            self.log_debug(f"[GENERATE_SUMMARY] Error generating summary: {e}")
            return None


    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: Optional[str] = None,
        done: bool = False,
        initial: bool = False,
    ):
        """
        Emit status updates with a formatted message for initial, follow-up, and final updates.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            level (str): The severity level of the status (e.g., "Info", "Error").
            message (Optional[str]): The status message to emit.
            done (bool): Indicates if this is the final status update.
            initial (bool): Indicates if this is the initial status update.
        """
        if __event_emitter__:
            current_time = time.time()
            elapsed_since_last_emit = current_time - self.last_emit_time

            # Determine if we should emit the status
            should_emit = False
            if done:
                should_emit = True  # Always emit final status
            elif initial:
                should_emit = True  # Always emit initial status
            elif elapsed_since_last_emit >= self.valves.emit_interval:
                should_emit = True  # Emit if emit_interval has passed

            if not should_emit:
                self.log_debug(
                    f"[EMIT_STATUS] Skipping emission. Elapsed time since last emit: {elapsed_since_last_emit:.2f}s < emit_interval: {self.valves.emit_interval}s."
                )
                return  # Skip emitting the status

            # Calculate total elapsed time from start_time for final status
            elapsed_time = current_time - self.start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            if minutes > 0:
                time_suffix = f"{minutes}m {seconds}s"
            else:
                time_suffix = f"{seconds}s"

            # Determine the appropriate status message
            if initial:
                formatted_message = "Thinking"
            elif done:
                formatted_message = f"Thought for {time_suffix}"
            else:
                formatted_message = message if message else "Processing..."

            # Prepare the status event
            event = {
                "type": "status",
                "data": {
                    "description": formatted_message,
                    "done": done,
                },
            }
            self.log_debug(f"[EMIT_STATUS] Emitting status: {formatted_message}")

            try:
                await __event_emitter__(event)
                self.log_debug("[EMIT_STATUS] Status emitted successfully.")
                # Update last_emit_time after successful emission
                self.last_emit_time = current_time
            except Exception as e:
                self.log_debug(f"[EMIT_STATUS] Error emitting status: {e}")

    async def emit_collapsible(self, __event_emitter__) -> None:
        """
        Update the UI with a collapsible section containing the accumulated thoughts.
        Ensures the collapsible has content from the start.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        # Accumulate all thoughts, preserving internal whitespace
        thoughts = self.thought_buffer.strip()
        if not thoughts:
            thoughts = ""

        # Create the collapsible enclosure with the accumulated thoughts
        enclosure = f"""
<details>
<summary>Click to expand thoughts</summary>
{thoughts}
</details>
""".strip()

        self.log_debug(
            f"[UPDATE_MESSAGES] Preparing to update message with: {enclosure}"
        )

        self.log_debug("[UPDATE_MESSAGES] Emitting new collapsible thought message.")
        message_event = {
            "type": "message",
            "data": {"content": enclosure},
        }
        try:
            await __event_emitter__(message_event)
            self.log_debug(
                "[UPDATE_MESSAGES] Collapsible thought message emitted successfully."
            )
        except Exception as e:
            self.log_debug(
                f"[UPDATE_MESSAGES] Error emitting collapsible thought message: {e}"
            )

    async def emit_output(self, __event_emitter__, content: str) -> None:
        """
        Emit <Output> content as a message to the UI, ensuring that <Output> and </Output> tags are omitted.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            content (str): The content to emit.
        """
        if __event_emitter__ and content:
            if not self.current_output_tag:
                self.current_output_tag = (
                    self.output_tags[0] if self.output_tags else "Output"
                )

            # Clean content by removing <Output> and </Output> tags
            # Use regex to remove tags without affecting surrounding whitespace
            tag_pattern = re.compile(
                rf"</?{re.escape(self.current_output_tag)}>", re.IGNORECASE
            )
            content_cleaned = tag_pattern.sub("", content)

            # Preserve internal whitespace and ensure no leading/trailing newlines
            content_cleaned = content_cleaned  # .strip("\n")

            # Log the before-and-after of the content cleaning
            self.log_debug(f"[EMIT_OUTPUT] Raw content: {content}")
            self.log_debug(f"[EMIT_OUTPUT] Cleaned content: {content_cleaned}")

            # Emit the cleaned content
            message_event = {
                "type": "message",
                "data": {"content": content_cleaned},
            }
            self.log_debug(
                f"[EMIT_OUTPUT] Emitting cleaned <{self.current_output_tag}> message: {content_cleaned}"
            )
            try:
                await __event_emitter__(message_event)
                self.log_debug("[EMIT_OUTPUT] Output message emitted successfully.")
            except Exception as e:
                self.log_debug(f"[EMIT_OUTPUT] Error emitting output message: {e}")

    async def emit_final_status(self, __event_emitter__):
        """
        Emit the final status message indicating the duration of the 'Thinking' phase.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        self.log_debug("[FINAL_STATUS] Preparing to emit the final status.")

        # Wait for any in-progress summary to complete
        if self.summary_in_progress:
            self.log_debug("[FINAL_STATUS] Waiting for ongoing summary generation.")
            while self.summary_in_progress:
                await asyncio.sleep(0.1)
            self.log_debug("[FINAL_STATUS] Ongoing summary generation completed.")

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time

        # Attempt to generate a final summary if summaries are enabled
        if self.valves.enable_llm_summaries and self.tags_detected:
            self.log_debug("[FINAL_STATUS] Generating final summary.")
            try:
                final_status = await asyncio.wait_for(
                    self.generate_summary(), timeout=self.valves.thought_summary_timeout
                )
                if not final_status:
                    self.log_debug(
                        "[FINAL_STATUS] No summary generated, using default."
                    )
                    final_status = f"Thought for {int(elapsed_time)}s"
            except asyncio.TimeoutError:
                self.log_debug("[FINAL_STATUS] Summary generation timed out.")
                final_status = f"Thought for {int(elapsed_time)}s"
        else:
            final_status = f"Thought for {int(elapsed_time)}s"

        # Explicitly clean any residual tags in the output buffer
        if self.output_buffer.strip():
            tag_pattern = re.compile(
                rf"</?{re.escape(self.current_output_tag)}>", re.IGNORECASE
            )
            cleaned_output = tag_pattern.sub("", self.output_buffer).strip("\n").strip()
            self.log_debug(
                f"[FINAL_STATUS] Cleaning residual tags from output: {cleaned_output}"
            )
            self.output_buffer = cleaned_output

        # Emit the final status message
        self.final_status_emitted = True
        self.log_debug(f"[FINAL_STATUS] Emitting final status: {final_status}")
        await self.emit_status(__event_emitter__, "Info", final_status, True)

    async def finalize_output(self, __event_emitter__):
        """
        Finalize output parsing mode by emitting any remaining buffered tokens and cleaning up.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        # Emit any remaining tokens in the buffer
        if self.output_buffer_tokens:
            remaining_output = "".join(self.output_buffer_tokens).strip("\n").strip()
            # Remove any residual <Output> tags if present
            tag_pattern = re.compile(
                rf"</?{re.escape(self.current_output_tag)}>", re.IGNORECASE
            )
            remaining_output_cleaned = tag_pattern.sub("", remaining_output).strip()
            await self.emit_output(__event_emitter__, remaining_output_cleaned)
            self.log_debug(
                f"[FINALIZE_OUTPUT] Emitting remaining output: {remaining_output_cleaned}"
            )
            self.output_buffer_tokens = []

        # Switch back to standard streaming
        self.mode = "standard_streaming"
        self.log_debug(
            "[FINALIZE_OUTPUT] Switching back to standard_streaming mode after finalizing output."
        )

    async def non_stream_response(
        self, payload: dict, __event_emitter__
    ) -> Optional[str]:
        """
        Handle non-streaming responses using the appropriate chat completion wrapper and return the summary directly.

        Args:
            payload (dict): The payload to send to the model.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            __request__ (Optional[Request]): The request object.

        Returns:
            Optional[str]: The summary generated by the model.
        """
        self.log_debug("[NON_STREAM_RESPONSE] Entered non_stream_response function.")
        try:
            if self.valves.debug_valve:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Calling _non_stream_completion for non-streaming with payload: {payload}"
                )
            summary = await self._non_stream_completion(
                form_data=payload
            )
            self.log_debug(
                f"[NON_STREAM_RESPONSE] _non_stream_completion returned summary: {summary}"
            )

            if summary:
                # Emit the summary as a message
                response_event = {
                    "type": "message",
                    "data": {"content": summary},
                }

                if __event_emitter__:
                    if self.tags_detected:
                        await self.emit_status(
                            __event_emitter__, "Info", "Thinking", initial=True
                        )
                    await __event_emitter__(response_event)
                    self.log_debug(
                        "[NON_STREAM_RESPONSE] Non-streaming summary message emitted successfully."
                    )

                # Emit the final status or modify message content based on valve
                await self.emit_final_status(__event_emitter__)

                if self.valves.debug_valve:
                    self.log_debug(
                        f"[NON_STREAM_RESPONSE] Emitted summary event: {response_event}"
                    )

                return summary
            else:
                self.log_debug(
                    "[NON_STREAM_RESPONSE] No summary returned from _non_stream_completion."
                )
                return None
        except Exception as e:
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__, "Error", f"Error: {str(e)}", True
                )
            if self.valves.debug_valve:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Error in non-stream response handling: {e}"
                )
            return None

    async def process_streaming_data(
        self, data: str, __event_emitter__, is_final_chunk: bool = False
    ) -> None:
        """
        Process incoming streaming data based on the current operational mode.
        Handles <Thought> and <Output> tags, manages buffers, and triggers appropriate actions.

        Operational Modes:
        - buffer_parsing: Detects the presence of <Thought> and <Output> tags.
        - thought_parsing: Accumulates thoughts between <Thought> and </Thought> tags.
        - output_parsing: Emits output while omitting <Output> tags.
        - standard_streaming: Emits tokens directly to the user (for when model chooses not to use thought/output tags).

        Args:
            data (str): The incoming data chunk to process.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            is_final_chunk (bool): Indicates if this is the final chunk in the stream.
        """
        self.log_debug(
            f"[PROCESS_DATA] Processing streaming data: {data}, Final Chunk: {is_final_chunk}"
        )

        # Buffer incoming data without stripping whitespace
        self.buffer += data  # Do not strip any whitespace from actual content
        self.log_debug(f"[PROCESS_DATA] Buffer updated: {self.buffer}")

        while self.buffer:
            if self.mode == "buffer_parsing":
                self.log_debug(f"[BUFFER_PARSING] Processing buffer: {self.buffer}")
                thought_tag_start = self.buffer.find(f"<{self.valves.thought_tag}>")
                output_tag_start = self.buffer.find(f"<{self.valves.output_tag}>")

                if output_tag_start != -1:
                    # Found <Output> tag
                    self.mode = "output_parsing"
                    self.tags_detected = True
                    self.log_debug(
                        "[BUFFER_PARSING] Detected <Output> tag. Switching to output_parsing mode."
                    )
                    # Remove the <Output> tag from buffer
                    self.buffer = self.buffer[
                        output_tag_start + len(f"<{self.valves.output_tag}>") :
                    ]
                    continue
                elif thought_tag_start != -1:
                    # Found <Thought> tag
                    self.mode = "thought_parsing"
                    self.tags_detected = True
                    self.log_debug(
                        "[BUFFER_PARSING] Detected <Thought> tag. Switching to thought_parsing mode."
                    )
                    # Emit 'Thinking' status
                    await self.emit_status(
                        __event_emitter__, level="Info", initial=True
                    )
                    # Remove the <Thought> tag from buffer
                    self.buffer = self.buffer[
                        thought_tag_start + len(f"<{self.valves.thought_tag}>") :
                    ]
                    continue
                else:
                    # Check if buffer exceeded expected tag length without finding a tag
                    max_tag_length = max(
                        len(f"<{self.valves.thought_tag}>"),
                        len(f"<{self.valves.output_tag}>"),
                    )
                    if len(self.buffer) > max_tag_length:
                        # Switch to standard streaming if no tags detected
                        self.mode = "standard_streaming"
                        self.log_debug(
                            "[BUFFER_PARSING] No tags detected. Switching to standard_streaming mode."
                        )
                        continue
                    else:
                        # Wait for more data
                        break

            elif self.mode == "thought_parsing":
                # In thought parsing mode, handle <Thought> tags and summaries
                thought_end_tag = f"</{self.valves.thought_tag}>"
                thought_end_idx = self.buffer.find(thought_end_tag)
                if thought_end_idx != -1:
                    thought_content = self.buffer[:thought_end_idx]
                    self.thought_buffer += thought_content
                    self.buffer = self.buffer[thought_end_idx + len(thought_end_tag) :]
                    self.inside_thought = False
                    self.log_debug(
                        f"[THOUGHT_PARSING] End of <Thought> tag detected. Content: {thought_content}"
                    )

                    if self.valves.use_collapsible:
                        # Emit collapsible thought element
                        await self.emit_collapsible(__event_emitter__)

                    await self.emit_final_status(__event_emitter__)

                    # Switch to buffer_parsing after thought parsing
                    self.mode = "buffer_parsing"
                    self.log_debug(
                        "[THOUGHT_PARSING] Switching to buffer_parsing mode after emitting collapsible thoughts."
                    )

                    continue
                else:
                    # Still inside <Thought> tag, accumulate tokens
                    # Since we are already in thought_parsing, just continue
                    break

            elif self.mode == "output_parsing":
                end_tag = f"</{self.valves.output_tag}>"
                buffer_combined = "".join(self.output_buffer_tokens) + self.buffer
                tag_index = buffer_combined.find(end_tag)

                if tag_index != -1:
                    content_before_tag = buffer_combined[:tag_index]
                    await self.emit_output(__event_emitter__, content_before_tag)
                    self.log_debug(
                        f"[OUTPUT_PARSING] Emitting content up to </Output>: {content_before_tag}"
                    )

                    # Remove emitted content and </Output> tag from the buffer
                    self.buffer = buffer_combined[tag_index + len(end_tag) :]
                    self.output_buffer_tokens = []  # Clear token buffer
                    self.log_debug(
                        f"[OUTPUT_PARSING] Remaining buffer after tag removal: {self.buffer}"
                    )
                    self.mode = "buffer_parsing"
                    continue
                elif is_final_chunk:
                    # Emit remaining buffer as final chunk
                    if buffer_combined:
                        await self.emit_output(__event_emitter__, buffer_combined)
                        self.log_debug(
                            f"[OUTPUT_PARSING] Final chunk: Emitting remaining buffer: {buffer_combined}"
                        )
                    self.buffer = ""
                    self.output_buffer_tokens = []
                    break
                else:
                    # **New Condition: Emit first half if buffer is too large and end tag not found**
                    if (
                        len(buffer_combined) >= 2 * len(end_tag)
                        and end_tag not in buffer_combined
                    ):
                        half_length = len(buffer_combined) // 2
                        content_to_emit = buffer_combined[:half_length]
                        await self.emit_output(__event_emitter__, content_to_emit)
                        self.log_debug(
                            f"[OUTPUT_PARSING] Buffer exceeded twice the end tag length. Emitting first half: {content_to_emit}"
                        )
                        # Update buffer_combined to keep the second half
                        self.buffer = buffer_combined[half_length:]
                        self.output_buffer_tokens = []
                        continue  # Continue processing with updated buffer

                    # Buffer content until </Output> tag is detected
                    if len(buffer_combined) > self.valves.output_buffer_limit:
                        emit_content = buffer_combined[
                            : -self.valves.output_buffer_limit
                        ]
                        self.output_buffer_tokens = list(
                            buffer_combined[-self.valves.output_buffer_limit :]
                        )
                    else:
                        emit_content = ""
                        self.output_buffer_tokens = list(buffer_combined)

                    if emit_content:
                        await self.emit_output(__event_emitter__, emit_content)
                        self.log_debug(
                            f"[OUTPUT_PARSING] Emitting partial content: {emit_content}"
                        )

                    self.buffer = "".join(self.output_buffer_tokens)
                    self.log_debug(
                        f"[OUTPUT_PARSING] Retaining buffer tokens for next iteration: {self.buffer}"
                    )
                    break

            elif self.mode == "standard_streaming":
                # In standard streaming mode, emit tokens as they come
                if self.buffer:
                    token = self.buffer[0]
                    await self.emit_output(__event_emitter__, token)
                    self.buffer = self.buffer[1:]
                    continue
                else:
                    break

        # After processing all buffer, emit final content if this is the final chunk
        if is_final_chunk and self.buffer:
            self.log_debug(
                f"[FINAL_CHUNK] Emitting remaining buffer as standard output: {self.buffer}"
            )
            await self.emit_output(__event_emitter__, self.buffer)
            self.buffer = ""
