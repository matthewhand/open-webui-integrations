"""
title: Open-WebUI Reasoning Manifold
version: 0.4.8c

- [x] Updated to work on OWUI 0.4.x
- [x] OpenAI streaming
- [x] Ollama streaming
- [x] Thought expansion
- [x] LLM summary generation
- [x] Advanced parsing logic
- [x] Optimised emission timing
- [x] Valve to append system prompt to ever request

"""

import os
import json
import time
import asyncio
from typing import List, Union, Optional, Callable, Awaitable, Dict, Any
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message, add_or_update_system_message
from starlette.responses import StreamingResponse
from open_webui.main import (
    generate_chat_completions,
    get_task_model_id,
)
from open_webui.utils.misc import get_last_assistant_message, get_content_from_message
from open_webui.config import TASK_MODEL, TASK_MODEL_EXTERNAL

import logging
import inspect  # For inspecting function signatures
import re  # For regex operations


# Mock the user object as expected by the function (OWUI 0.4.x constraint)
mock_user = {
    "id": "reasoning_manifold",
    "username": "reasoning_manifold",
    "role": "admin",
}

THOUGHT_SUMMARY_PLACEHOLDER = "your-task-model-id-goes-here"
SYSTEM_MESSAGE_DEFAULT = "Respond using <Thought> and <Output> tags."


class Pipe:
    """
    The Pipe class handles streaming responses from OpenAI or Ollama models,
    processes internal thought tags, manages output parsing, generates summaries,
    and ensures clean and user-friendly message emission without exposing internal tags.
    """

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
        self.tracked_tokens = []  # Tracked tokens for dynamic summaries
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
        self.task_model_id = None  # Will be set in pipe method
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
        self.log = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)
        self.log.setLevel(logging.DEBUG if self.valves.debug_valve else logging.INFO)

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
        Returns the `thought_summary_model_id` if specified and not default; otherwise, falls back to `task_model_id`.
        """
        if (
            self.valves.thought_summary_model_id
            and self.valves.thought_summary_model_id.lower()
            != THOUGHT_SUMMARY_PLACEHOLDER
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

        Returns:
            Union[str, StreamingResponse]: The response to be sent back.
        """
        # Reset state variables for each new request
        self.reset_state()

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

            # Determine the task model ID dynamically, considering valve overrides
            try:
                self.task_model_id = self.get_summary_model_id()
                self.log_debug(f"Resolved task_model_id: {self.task_model_id}")
            except RuntimeError as e:
                self.log_debug(f"Failed to resolve task model ID: {e}")
                self.task_model_id = model_id  # Fallback to original model_id

            self.log_debug(f"Selected task_model_id: {self.task_model_id}")

            # Handle system messages if present (assumes system messages are separated)
            body_messages = body.get("messages", [])
            system_message, messages = pop_system_message(body_messages)

            # Append system prompt to user messages if the valve is enabled
            if self.valves.append_system_prompt:
                system_prompt = (
                    self.valves.system_message_override
                    if self.valves.system_message_override_enabled
                    else get_content_from_message(system_message)
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
                add_or_update_system_message(
                    self.valves.system_message_override, messages
                )
                self.log_debug(
                    f"Overriding system message with system_message_override: {self.valves.system_message_override}"
                )
            elif system_message:
                # Append the provided system message to the beginning of the list
                system_message_content = get_content_from_message(system_message)
                add_or_update_system_message(system_message_content, messages)
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

            payload = {
                "model": model_id,
                "messages": processed_messages,
            }

            self.log_debug(f"Payload prepared for model '{model_id}': {payload}")

            if self.valves.streaming_enabled:
                self.log_debug("Streaming is enabled. Handling stream response.")
                response = await self.stream_response(payload, __event_emitter__)
            else:
                self.log_debug("Streaming is disabled. Handling non-stream response.")
                response = await self.non_stream_response(payload, __event_emitter__)

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

    async def stream_response(self, payload, __event_emitter__) -> StreamingResponse:
        """
        Handle streaming responses from generate_chat_completions.

        Steps:
        1. Call the generate_chat_completions function with streaming enabled.
        2. Iterate over each chunk in the streaming response.
        3. Process each chunk based on its content and type.
        4. Manage operational modes and emit messages accordingly.
        5. Handle the finalization of the stream upon receiving [DONE].

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

                # Call the generate_chat_completions function
                stream_response = await generate_chat_completions(
                    form_data=payload, user=mock_user, bypass_filter=True
                )
                self.log_debug(
                    "[STREAM_RESPONSE] generate_chat_completions called successfully."
                )

                # Ensure the response is a StreamingResponse
                if isinstance(stream_response, StreamingResponse):
                    self.log_debug(
                        "[STREAM_RESPONSE] Received StreamingResponse from generate_chat_completions."
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
                                                "",
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
                                                "[TOKEN_SUMMARY] Summary already in progress. Skipping additional summary trigger."
                                            )

                            else:
                                # Handle unexpected data format
                                self.log_debug(
                                    f"[STREAM_RESPONSE] Unexpected chunk format: {chunk_str}"
                                )

                        except Exception as e:
                            # Handle errors in processing chunks
                            self.log_debug(
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
                self.log_debug(f"[STREAM_RESPONSE] Error in response_generator: {e}")
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

        if not (self.valves.enable_llm_summaries):
            self.log_debug("[TRIGGER_SUMMARY] LLM summaries are disabled.")
            return

        if not (self.tags_detected):
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
        if summary_model_id != "":
            payload = {
                "model": self.get_summary_model_id(),
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

            try:
                self.log_debug(
                    f"[GENERATE_SUMMARY] Generating summary with payload: {payload}"
                )
                response = await generate_chat_completions(
                    form_data=payload, user=mock_user, bypass_filter=True
                )
                # Ensure that the summary does not contain trailing newlines
                summary = (
                    response["choices"][0]["message"]["content"].rstrip("\n").strip()
                )
                self.log_debug(f"[SUMMARY_GENERATED] Generated summary: {summary}")
                return summary
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
        await __event_emitter__(message_event)

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

    async def non_stream_response(self, payload, __event_emitter__):
        """
        Handle non-streaming responses from generate_chat_completions.

        Steps:
        1. Call the generate_chat_completions function without streaming.
        2. Emit the assistant's message to the user.
        3. Emit the final status message indicating the duration of the 'Thinking' phase.

        Args:
            payload (dict): The payload to send to the model.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.

        Returns:
            Union[str, Dict[str, Any]]: The response content or error message.
        """
        self.log_debug("[NON_STREAM_RESPONSE] Entered non_stream_response function.")
        try:
            if self.valves.debug_valve:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Calling generate_chat_completions for non-streaming with payload: {payload}"
                )
            response_content = await generate_chat_completions(form_data=payload)
            self.log_debug(
                f"[NON_STREAM_RESPONSE] generate_chat_completions response: {response_content}"
            )

            assistant_message = response_content["choices"][0]["message"][
                "content"
            ].rstrip("\n")
            response_event = {
                "type": "message",
                "data": {"content": assistant_message},
            }

            if __event_emitter__:
                if self.tags_detected:
                    await self.emit_status(
                        __event_emitter__, "Info", "Thinking", initial=True
                    )
                await __event_emitter__(response_event)
                self.log_debug(
                    "[NON_STREAM_RESPONSE] Non-streaming message event emitted successfully."
                )

            # Emit the final status or modify message content based on valve
            await self.emit_final_status(__event_emitter__)

            if self.valves.debug_valve:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Emitted response event: {response_event}"
                )

            return response_content
        except Exception as e:
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__, "Error", f"Error: {str(e)}", True
                )
            if self.valves.debug_valve:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Error in non-stream response handling: {e}"
                )
            return f"Error: {e}"

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

                    # Switch to standard streaming after thought parsing
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

    async def non_stream_response(self, payload, __event_emitter__):
        """
        Handle non-streaming responses from generate_chat_completions.

        Steps:
        1. Call the generate_chat_completions function without streaming.
        2. Emit the assistant's message to the user.
        3. Emit the final status message indicating the duration of the 'Thinking' phase.

        Args:
            payload (dict): The payload to send to the model.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.

        Returns:
            Union[str, Dict[str, Any]]: The response content or error message.
        """
        self.log_debug("[NON_STREAM_RESPONSE] Entered non_stream_response function.")
        try:
            if self.valves.debug_valve:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Calling generate_chat_completions for non-streaming with payload: {payload}"
                )
            response_content = await generate_chat_completions(form_data=payload)
            self.log_debug(
                f"[NON_STREAM_RESPONSE] generate_chat_completions response: {response_content}"
            )

            assistant_message = response_content["choices"][0]["message"][
                "content"
            ].rstrip("\n")
            response_event = {
                "type": "message",
                "data": {"content": assistant_message},
            }

            if __event_emitter__:
                if self.tags_detected:
                    await self.emit_status(
                        __event_emitter__, "Info", "Thinking", initial=True
                    )
                await __event_emitter__(response_event)
                self.log_debug(
                    "[NON_STREAM_RESPONSE] Non-streaming message event emitted successfully."
                )

            # Emit the final status or modify message content based on valve
            await self.emit_final_status(__event_emitter__)

            if self.valves.debug_valve:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Emitted response event: {response_event}"
                )

            return response_content
        except Exception as e:
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__, "Error", f"Error: {str(e)}", True
                )
            if self.valves.debug_valve:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Error in non-stream response handling: {e}"
                )
            return f"Error: {e}"
