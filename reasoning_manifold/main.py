"""
title: Open-WebUI Reasoning Manifold
version: 0.4.3

20241125 - 0.4.1 - Updated to work on OWUI 0.4.x and Ollama Streaming
20241125 - 0.4.2 - Fixed emit_collapsible

"""

import os
import json
import time
import asyncio
from typing import List, Union, Optional, Callable, Awaitable, Dict, Any
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
from starlette.responses import StreamingResponse
from open_webui.main import (
    generate_chat_completions,
    get_task_model_id,
    get_all_base_models,
)
from open_webui.utils.misc import get_last_assistant_message, get_content_from_message
from open_webui.config import TASK_MODEL, TASK_MODEL_EXTERNAL

import logging
import inspect  # Import to inspect the function signature


class Pipe:

    # Compatibility layer to handle 0.3.x to 0.4.x change to signature
    def resolve_task_model_id(
        model_id: str,
        task_model: str = None,
        task_model_external: str = None,
        models: dict = None,
        valve_override: Optional[str] = None,
    ) -> str:
        """
        Resolve the task model ID dynamically based on the version of get_task_model_id.

        Supports:
        - Older single-argument version
        - Updated multi-argument version
        - Overrides via valve configuration.
        """
        try:
            # Check for valve override
            if valve_override != "default-task-model":
                self.log_debug(
                    f"[THOUGHT_SUMMARY] valve override will be used: {valve_override}"
                )
                return valve_override

            # Get the signature of the `get_task_model_id` function
            sig = inspect.signature(get_task_model_id)
            params = sig.parameters

            task_model_id = ""

            # Check the number of parameters and their names
            if len(params) == 1:
                # Single-argument version (0.3.x)
                self.log_debug(f"[THOUGHT_SUMMARY] detected OWUI <=0.3.x")
                task_model_id = get_task_model_id(model_id)
            elif len(params) == 4:
                # Multi-argument version (0.4.x)
                self.log_debug(f"[THOUGHT_SUMMARY] detected OWUI >=0.4.x")
                self.log_debug(
                    f"[THOUGHT_SUMMARY] selecting model using params: {model_id}, {task_model}, {task_model_external}, {models}"
                )
                task_model_id = get_task_model_id(
                    default_model_id=model_id,
                    task_model=task_model,
                    task_model_external=task_model_external,
                    models=models,
                )
            else:
                raise TypeError("Unexpected number of arguments in get_task_model_id")
            return model
        except Exception as e:
            raise RuntimeError(f"Error resolving task model ID: {e}")

    # Mock the user object as expected by the function (OWUI 0.4.x constraint)
    mock_user = {
        "id": "dummy_user",
        "username": "dummy_user",
        "role": "admin",
    }

    class Valves(BaseModel):
        """
        Configuration for the Open-WebUI Reasoning Manifold.
        """

        # Model Configuration
        model_ids: str = Field(
            default="marco-o1",
            description="Comma-separated list of model IDs to be used by the manifold.",
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
        thought_summary_model_id: str = Field(
            default="default-task-model",
            description=(
                "Optional override for the model ID used to generate thought summaries. "
                "If 'default-task-model', will use the relevant task model from Admin Panel > Settings > Interface."
            ),
        )
        thought_summary_interval_tokens: int = Field(
            default=10,
            description="Number of tokens after which to generate a thought summary.",
        )

        # Status Update Configuration
        use_collapsible: bool = Field(
            default=False,
            description="(WIP) Use collapsible UI for updates. If disabled, emits status updates.",
        )
        enable_llm_summaries: bool = Field(
            default=False,
            description="Enable LLM-generated summaries for updates.",
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

        # Debugging
        debug: bool = Field(default=True, description="Enable debug logging.")

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
        self.determined = False
        self.inside_thought = False
        self.inside_output = False
        self.token_count_since_last_summary = 0
        self.start_time = None
        self.buffer = ""  # Initialize the buffer
        self.tracked_tokens = []  # Track tokens for dynamic summaries
        self.last_summary_time = time.time()  # Last time a summary was emitted
        self.task_model_id = None  # Will be set in pipe method
        self.thought_summary_task = None  # Task for generating summaries
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
        self.log.setLevel(logging.DEBUG if self.valves.debug else logging.INFO)

        # Parse tags based on model_ids
        self.parse_tags()

    def log_debug(self, message: str):
        """Log debug messages if debugging is enabled."""
        if self.valves.debug:
            self.log.debug(message)

    def parse_tags(self):
        """
        Parse the thought_tag and output_tag fields into lists matching the model_ids.
        Supports both singular tags (applied to all models) and comma-separated lists.
        """
        model_ids = [m.strip() for m in self.valves.model_ids.split(",") if m.strip()]
        model_count = len(model_ids)
        self.log_debug(f"Parsing tags for {model_count} models.")

        # Parse thought_tags
        thought_tags = [tag.strip() for tag in self.valves.thought_tag.split(",")]
        if len(thought_tags) == 1:
            self.thought_tags = thought_tags * model_count
        elif len(thought_tags) == model_count:
            self.thought_tags = thought_tags
        else:
            self.log.debug(
                f"[TAG_ERROR] Number of thought_tags ({len(thought_tags)}) does not match number of model_ids ({model_count}). "
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
            self.log.debug(
                f"[TAG_ERROR] Number of output_tags ({len(output_tags)}) does not match number of model_ids ({model_count}). "
                f"Defaulting all output_tags to '{output_tags[0]}'"
            )
            self.output_tags = [output_tags[0]] * model_count

        self.log_debug(f"Parsed output_tags: {self.output_tags}")

    def get_models(self) -> List[dict]:
        """List of models derived from the valve configuration."""
        model_ids = [
            model_id.strip()
            for model_id in self.valves.model_ids.split(",")
            if model_id.strip()
        ]
        return [{"id": model_id, "name": model_id} for model_id in model_ids]

    def pipes(self) -> List[dict]:
        """Return the list of model pipes."""
        return self.get_models()

    def get_summary_model_id(self) -> str:
        """
        Determine the model ID to use for generating summaries.
        Returns the `thought_summary_model_id` if specified; otherwise, falls back to `task_model_id`.
        """
        if self.valves.thought_summary_model_id:
            self.log_debug(
                f"[SUMMARY_CHECK] Using override model ID for summary: {self.valves.thought_summary_model_id}"
            )
            return self.valves.thought_summary_model_id
        else:
            self.log_debug(
                f"[SUMMARY_CHECK] Using dynamic task model ID: {self.task_model_id}"
            )
            return self.task_model_id or ""

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> Union[str, StreamingResponse]:
        """
        Main handler for processing requests.
        """
        # Reset state variables for each new request
        self.thought_buffer = ""
        self.output_buffer = ""
        self.determined = False
        self.inside_thought = False
        self.inside_output = False
        self.token_count_since_last_summary = 0
        self.tracked_tokens = []
        self.last_summary_time = time.time()
        self.start_time = time.time()
        self.stop_emitter.clear()
        self.thought_summary_task = None
        self.buffer = ""  # Reset buffer
        self.task_model_id = None  # Reset task model ID
        self.messages = []  # Reset messages list

        # Reset current tags
        self.current_thought_tag = (
            self.valves.thought_tag
        )  # Default to valve's thought_tag
        self.current_output_tag = (
            self.valves.output_tag
        )  # Default to valve's output_tag

        try:
            # Emit the initial "Thinking" status immediately
            if __event_emitter__:
                await self.emit_status(__event_emitter__, level="Info", initial=True)

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
                self.task_model_id = self.resolve_task_model_id(
                    model_id,
                    task_model=body.get("task_model", TASK_MODEL),
                    task_model_external=body.get(
                        "task_model_external", TASK_MODEL_EXTERNAL
                    ),
                    models=await get_all_base_models(),
                    valve_override=self.valves.thought_summary_model_id,  # Pass valve override
                )
                self.log_debug(f"Resolved task_model_id: {self.task_model_id}")
            except RuntimeError as e:
                self.log_debug(f"Failed to resolve task model ID: {e}")
                self.task_model_id = model_id  # Fallback to original model_id

            self.log_debug(f"Selected task_model_id: {self.task_model_id}")

            # Handle system messages if present (assumes system messages are separated)
            system_message, messages = pop_system_message(body.get("messages", []))
            self.log_debug(f"System message: {system_message}")
            self.log_debug(
                f"Number of messages after popping system message: {len(messages)}"
            )

            self.messages = messages  # Store messages for later modification

            processed_messages = [
                {"role": message["role"], "content": message.get("content", "")}
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
                                chunk_str = chunk.decode("utf-8").rstrip("\n")
                            elif isinstance(chunk, str):
                                chunk_str = chunk.rstrip("\n")
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

                                    # Process any remaining buffer content
                                    if self.buffer.strip():
                                        try:
                                            self.log_debug(
                                                "[STREAM_RESPONSE] Processing remaining buffer content."
                                            )
                                            await self.process_streaming_data(
                                                "", __event_emitter__
                                            )
                                        except AttributeError as e:
                                            self.log_debug(
                                                f"[STREAM_RESPONSE] Error: {e}. Ensure process_streaming_data is defined and accessible."
                                            )
                                        self.buffer = ""

                                    # Emit any remaining thought content
                                    if self.thought_buffer.strip():
                                        self.log_debug(
                                            f"[STREAM_RESPONSE] Emitting remaining thought content: {self.thought_buffer}"
                                        )
                                        if self.valves.use_collapsible:
                                            await self.emit_thought(
                                                __event_emitter__, self.thought_buffer
                                            )
                                        self.thought_buffer = ""

                                    # Emit final status or collapsible enclosure
                                    await self.emit_final_status_or_update(
                                        __event_emitter__
                                    )
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

                                    # Process the data for tag detection and buffering
                                    await self.process_streaming_data(
                                        data, __event_emitter__
                                    )

                                    # Check if it's time to trigger an update
                                    current_time = time.time()
                                    time_since_last_summary = (
                                        current_time - self.last_summary_time
                                    )

                                    if (
                                        time_since_last_summary
                                        >= self.valves.emit_interval
                                    ):
                                        self.last_summary_time = current_time
                                        await self.trigger_update(__event_emitter__)

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

    async def process_streaming_data(self, data: str, __event_emitter__) -> None:
        """
        Process incoming streaming data, handling <Thought> and <Output> tags,
        and ensure thought content is not emitted directly.
        """
        self.log_debug(f"[PROCESS_DATA] Processing streaming data: {data}")

        # Buffer incoming data
        self.buffer += data
        self.log_debug(f"[PROCESS_DATA] Buffer updated: {self.buffer}")

        # Ensure current_output_tag is initialized
        if not self.current_output_tag:
            self.current_output_tag = (
                self.output_tags[0] if self.output_tags else "Output"
            )

        # Ensure current_thought_tag is initialized
        if not self.current_thought_tag:
            self.current_thought_tag = (
                self.thought_tags[0] if self.thought_tags else "Thought"
            )

        while self.buffer:
            if self.inside_thought:
                end_tag = f"</{self.current_thought_tag}>"
                end_idx = self.buffer.find(end_tag)
                if end_idx != -1:
                    # Found end of thought
                    thought_content = self.buffer[:end_idx]
                    self.thought_buffer += thought_content
                    self.buffer = self.buffer[end_idx + len(end_tag) :]
                    self.inside_thought = False
                    self.log_debug(
                        f"[PROCESS_DATA] End of <{self.current_thought_tag}> tag detected."
                    )
                    if self.valves.use_collapsible:
                        # Generate a summary and modify the last message's content
                        self.log_debug("[TRIGGER_UPDATE] Emitting collapsible UI.")
                        await self.emit_collapsible(__event_emitter__)
                    continue  # Go back to top of loop to process any remaining buffer
                else:
                    # No end tag yet, buffer everything and break
                    break  # Wait for more data

            elif self.inside_output:
                end_tag = f"</{self.current_output_tag}>"
                end_idx = self.buffer.find(end_tag)
                if end_idx != -1:
                    # Found end of output
                    output_content = self.buffer[:end_idx]
                    self.log_debug(
                        f"[PROCESS_DATA] Raw output content before cleaning: {output_content}"
                    )
                    await self.emit_output(__event_emitter__, output_content)

                    # Remove end tag and reset buffer
                    self.buffer = self.buffer[end_idx + len(end_tag) :].strip()
                    self.log_debug(
                        f"[PROCESS_DATA] Buffer after cleaning end tag: {self.buffer}"
                    )

                    self.inside_output = False
                    self.log_debug(
                        f"[PROCESS_DATA] End of <{self.current_output_tag}> tag detected."
                    )
                    continue  # Process remaining buffer
                else:
                    # No end tag yet, handle partial content
                    possible_end_tag_start = self.buffer.rfind("</")
                    if possible_end_tag_start != -1 and possible_end_tag_start > 0:
                        # Emit everything before the possible end tag start
                        partial_content = self.buffer[:possible_end_tag_start]
                        partial_content_cleaned = re.sub(
                            rf"</?{re.escape(self.current_output_tag)}>",
                            "",
                            partial_content,
                        )
                        await self.emit_output(
                            __event_emitter__, partial_content_cleaned
                        )
                        self.buffer = self.buffer[possible_end_tag_start:]
                        self.log_debug(
                            f"[PROCESS_DATA] Emitting partial content up to last tag boundary: {partial_content_cleaned}"
                        )
                    else:
                        # No clear end tag start, emit buffer and clean tags
                        cleaned_buffer = re.sub(
                            rf"</?{re.escape(self.current_output_tag)}>",
                            "",
                            self.buffer,
                        )
                        await self.emit_output(__event_emitter__, cleaned_buffer)
                        self.buffer = ""
                        self.log_debug(
                            f"[PROCESS_DATA] Buffer emitted entirely after cleaning tags: {cleaned_buffer}"
                        )
                    break

            else:
                # Not inside any tag, look for start tags
                found_tag = False
                for idx, tag in enumerate(self.thought_tags):
                    start_tag = f"<{tag}>"
                    if self.buffer.startswith(start_tag):
                        self.current_thought_tag = tag
                        self.current_output_tag = self.output_tags[idx]
                        self.buffer = self.buffer[len(start_tag) :]
                        self.inside_thought = True
                        self.determined = True
                        self.log_debug(f"[PROCESS_DATA] Detected <{tag}> tag.")
                        found_tag = True
                        break
                if not found_tag:
                    for idx, tag in enumerate(self.output_tags):
                        start_tag = f"<{tag}>"
                        if self.buffer.startswith(start_tag):
                            self.current_output_tag = tag
                            self.buffer = self.buffer[len(start_tag) :]
                            self.inside_output = True
                            self.determined = True
                            self.log_debug(f"[PROCESS_DATA] Detected <{tag}> tag.")
                            found_tag = True
                            break
                if not found_tag:
                    # No tags detected, check if buffer is long enough to decide
                    max_tag_length = max(
                        len(f"<{tag}>") for tag in self.thought_tags + self.output_tags
                    )
                    if len(self.buffer) >= max_tag_length:
                        # No start tag, discard one character and continue
                        self.log_debug(
                            "[PROCESS_DATA] No start tag detected in buffer, discarding one character."
                        )
                        self.buffer = self.buffer[1:]
                        continue  # Re-attempt to find a tag in the updated buffer
                    else:
                        # Buffer is not long enough to determine, wait for more data
                        break  # Wait for more data

    async def trigger_update(self, __event_emitter__) -> None:
        """
        Trigger updates based on the current valve configuration.
        """
        self.log_debug(
            "[TRIGGER_UPDATE] Triggering update based on valve configuration."
        )

        # Emit status update
        self.log_debug("[TRIGGER_UPDATE] Emitting status update.")
        await self.emit_status_update(__event_emitter__)

    async def emit_status_update(self, __event_emitter__) -> None:
        if self.valves.enable_llm_summaries:
            self.log_debug("[GENERATE_SUMMARY] Generating summary for status update.")
            summary = await self.generate_summary()
            status_message = summary or "Processing..."
        else:
            elapsed_time = time.time() - self.start_time
            status_message = f"Processing... ({int(elapsed_time)}s)"
        await self.emit_status(__event_emitter__, "Info", status_message, False)

    async def emit_collapsible(self, __event_emitter__) -> None:
        """
        Update the UI with a collapsible section, replacing the last message's content.
        Ensures the collapsible has content from the start.
        """

        # Accumulate all thoughts
        thoughts = self.thought_buffer.strip()
        if not thoughts:
            thoughts = ""

        # Create the collapsible enclosure with the generated summary
        enclosure = f"""
<details>
<summary>Click to expand thoughts</summary>
{thoughts}
</details>
""".strip()

        self.log_debug(
            f"[UPDATE_MESSAGES] Preparing to update message with: {enclosure}"
        )

        self.log_debug("[UPDATE_MESSAGES] No previous message, emitting new message.")
        message_event = {
            "type": "message",
            "data": {"content": enclosure},
        }
        await __event_emitter__(message_event)

    async def generate_summary(self) -> Optional[str]:
        if not self.tracked_tokens:
            return None

        payload = {
            "model": self.get_summary_model_id(),
            "messages": [
                {"role": "system", "content": self.valves.dynamic_status_system_prompt},
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
            summary = response["choices"][0]["message"]["content"].strip()
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
        """
        if __event_emitter__:
            elapsed_time = time.time() - self.start_time
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
                    "level": level,
                    "description": formatted_message,
                    "done": done,
                },
            }
            self.log_debug(f"[EMIT_STATUS] Emitting status: {formatted_message}")

            try:
                await __event_emitter__(event)
                self.log_debug("[EMIT_STATUS] Status emitted successfully.")
            except Exception as e:
                self.log_debug(f"[EMIT_STATUS] Error emitting status: {e}")

    async def emit_thought(self, __event_emitter__, thought_content: str) -> None:
        """
        Emit the buffered <Thought> content as a collapsible UI element.
        """
        if __event_emitter__ and thought_content.strip():
            collapsible_event = {
                "type": "thought",
                "data": {
                    "content": thought_content,
                    "collapsible": True,  # Indicates to the UI that this is collapsible
                },
            }
            self.log_debug(
                f"[EMIT_THOUGHT] Emitting <{self.current_thought_tag}> content as collapsible: {thought_content}"
            )
            try:
                await __event_emitter__(collapsible_event)
                self.log_debug("[EMIT_THOUGHT] Thought event emitted successfully.")
            except Exception as e:
                self.log_debug(f"[EMIT_THOUGHT] Error emitting thought event: {e}")

    async def emit_output(self, __event_emitter__, content: str) -> None:
        """
        Emit <Output> content as a message to the UI.
        """
        if __event_emitter__ and content.strip():
            if not self.current_output_tag:
                self.current_output_tag = (
                    self.output_tags[0] if self.output_tags else "Output"
                )

            # Clean content by removing <Output> and </Output> tags
            content_cleaned = (
                content.replace(f"<{self.current_output_tag}>", "")
                .replace(f"</{self.current_output_tag}>", "")
                .strip()
            )

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

    async def emit_final_status_or_update(self, __event_emitter__):
        """
        Emit the final status or update the collapsible section at the end of streaming.
        """
        elapsed_time = time.time() - self.start_time
        if self.valves.enable_llm_summaries:
            final_status = await self.generate_summary()
            if not final_status:
                final_status = f"Processing... ({int(elapsed_time)}s)"
        else:
            final_status = f"Processing... ({int(elapsed_time)}s)"

        # Explicitly clean any residual tags
        if self.output_buffer.strip():
            cleaned_output = self.output_buffer.replace(
                f"</{self.current_output_tag}>", ""
            ).strip()
            self.log_debug(
                f"[FINAL_STATUS] Cleaning residual tags from output: {cleaned_output}"
            )
            self.output_buffer = cleaned_output

        await self.emit_status(__event_emitter__, "Info", final_status, True)

    async def non_stream_response(self, payload, __event_emitter__):
        """
        Handle non-streaming responses from generate_chat_completions.
        """
        self.log_debug("[NON_STREAM_RESPONSE] Entered non_stream_response function.")
        try:
            if self.valves.debug:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Calling generate_chat_completions for non-streaming with payload: {payload}"
                )
            response_content = await generate_chat_completions(form_data=payload)
            self.log_debug(
                f"[NON_STREAM_RESPONSE] generate_chat_completions response: {response_content}"
            )

            assistant_message = response_content["choices"][0]["message"]["content"]
            response_event = {
                "type": "message",
                "data": {"content": assistant_message},
            }

            if __event_emitter__:
                await __event_emitter__(response_event)
                self.log_debug(
                    "[NON_STREAM_RESPONSE] Non-streaming message event emitted successfully."
                )

            # Emit the final status or modify message content based on valve
            await self.emit_final_status_or_update(__event_emitter__)

            if self.valves.debug:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Emitted response event: {response_event}"
                )

            return response_content
        except Exception as e:
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__, "Error", f"Error: {str(e)}", True
                )
            if self.valves.debug:
                self.log_debug(
                    f"[NON_STREAM_RESPONSE] Error in non-stream response handling: {e}"
                )
            return f"Error: {e}"
