import os
import json
import time
import asyncio
import random
import uuid
import html
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Callable, Awaitable, Dict, Any, Union

from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
from starlette.responses import StreamingResponse, JSONResponse
from open_webui.main import (
    generate_chat_completions,
    generate_moa_response,
    get_models,
)
import aiohttp  # Added to handle ClientResponseError


# Define the User class with required attributes
@dataclass
class User:
    id: str
    username: str
    name: str  # Added the 'name' attribute
    role: str
    email: str  # Ensures 'email' attribute is present


# Instantiate the mock_user with all necessary attributes, including 'name'
mock_user = User(
    id="consensus_manifold",
    username="consensus_manifold",
    name="Consensus Manifold",
    role="admin",
    email="root@host.docker.internal",
)

# Global Placeholders
CONSENSUS_PLACEHOLDER = "your-consensus-model-id-goes-here"
PROGRESS_SUMMARY_PLACEHOLDER = "your-progress-model-id-goes-here"
TAGS_PLACEHOLDER = "comma-separated-list-of-model-tags-goes-here"
CONSENSUS_SYSTEM_PROMPT_DEFAULT = (
    "Only report on the consensus across the output provided from multiple models."
)
PROGRESS_SUMMARY_STATUS_PROMPT_DEFAULT = "Summarize progress in exactly 4 words."


class Pipe:
    """
    The Pipe class manages the consensus-based reasoning process.
    It queries multiple contributing models (explicit, random, or tag-based) based on configuration,
    generates progress summaries, handles interruptions, and synthesizes a consensus response using the designated consensus model.
    """

    class Valves(BaseModel):
        """
        Configuration for the Consensus Manifold.
        """

        enable_random_contributors: bool = Field(
            default=True,
            description="Enable random contributing model selection (true by default).",
        )
        random_contributing_models_number: int = Field(
            default=3,
            description="How many models should be selected for contribution.",
        )
        random_contributing_models_blacklist_regex: str = Field(
            default="(embed|vision|whisper)",
            description="Regular expression to exclude models from random selection.",
        )
        enable_explicit_contributors: bool = Field(
            default=False,
            description="Enable explicit (user-specified) contributing model selection if provided (false by default).",
        )
        explicit_contributing_models: List[str] = Field(
            default=[],
            description="List of specific contributing models to use when explicit selection is enabled.",
        )
        enable_tagged_contributors: bool = Field(
            default=False,
            description="Enable tag-based contributing model selection (false by default).",
        )
        contributor_tags: str = Field(
            default=TAGS_PLACEHOLDER,
            description="Comma-separated list of tags for contributing model selection.",
        )
        manifold_prefix: str = Field(
            default="consensus/",
            description="Prefix used for consensus models.",
        )
        consensus_model_id: str = Field(
            default=CONSENSUS_PLACEHOLDER,
            description="Model ID override for generating final consensus.",
        )
        consensus_system_prompt: str = Field(
            default=CONSENSUS_SYSTEM_PROMPT_DEFAULT,
            description="System prompt for consensus generation.",
        )
        enable_llm_summaries: bool = Field(
            default=True, description="Enable LLM-based progress summaries."
        )
        progress_summary_model_id: str = Field(
            default=PROGRESS_SUMMARY_PLACEHOLDER,
            description="Model ID used for generating progress summaries.",
        )
        progress_summary_interval_seconds: float = Field(
            default=5.0,
            description="Interval in seconds between progress summaries.",
        )
        progress_summary_status_prompt: str = Field(
            default=PROGRESS_SUMMARY_STATUS_PROMPT_DEFAULT,
            description="Prompt for generating progress summaries.",
        )
        progress_summary_system_prompt: str = Field(
            default="You are a helpful assistant summarizing thoughts concisely, only respond with the summary.",
            description="System prompt for dynamic status generation.",
        )
        use_collapsible: bool = Field(
            default=True,
            description="Enable collapsible UI for displaying model outputs.",
        )
        emit_interval: float = Field(
            default=3.0,
            description="Interval in seconds between progress status updates.",
        )
        interrupt: bool = Field(
            default=False, description="Enable or disable interruption logic."
        )
        minimum_number_of_models: int = Field(
            default=2,
            description="Minimum number of models required to induce interruption.",
        )
        interruption_grace_period: float = Field(
            default=3.0,
            description="Grace period (in seconds) to wait for additional models after interruption.",
        )
        max_reroll_retries: int = Field(
            default=3,
            description="Maximum number of reroll attempts for random models.",
        )
        reroll_backoff_delay: int = Field(
            default=2,
            description="Base delay (in seconds) between reroll attempts.",
        )
        strip_collapsible_tags: bool = Field(
            default=False,
            description="Enable or disable stripping of <details> HTML tags from assistant messages.",  # ADJUSTED: Controls stripping behavior
        )
        append_system_prompt: bool = Field(
            default=False,
            description=(
                "If True, appends the system message to every user request, enclosed in parentheses."
            ),
        )
        debug_valve: bool = Field(default=True, description="Enable debug logging.")
        max_models_per_provider: int = Field(
            default=1,
            description="Maximum number of models to select per provider to avoid rate limiting."
        )

    def __init__(self):
        """
        Initialize the Pipe with default valves and necessary state variables.
        """
        self.type = "manifold"
        self.id = "consensus"
        self.valves = self.Valves()
        self.name = self.valves.manifold_prefix
        self.stop_emitter = asyncio.Event()

        # State Variables
        self.model_outputs: Dict[str, str] = {}
        self.completed_contributors: set = set()
        self.interrupted_contributors: set = set()
        self.consensus_output: Optional[str] = None
        self.last_summary_model_index: int = 0
        self.last_emit_time: float = 0.0
        self.current_mode: Optional[str] = None  # Stores current selection mode
        self.buffer: str = ""  # Initialize buffer
        self.output_buffer_tokens: List[str] = []  # Buffer for output parsing
        self.output_buffer_limit: int = (
            5  # Number of tokens to buffer for tag detection
        )
        self.summary_in_progress: bool = False  # Flag to indicate summary generation
        self.final_status_emitted: bool = (
            False  # Flag to indicate if final status is emitted
        )
        self.tags_detected: bool = False  # Flag for tag detection
        self.consensus_streamed: bool = False  # Flag to indicate if consensus is streamed

        # Logging Setup
        self.log = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)
        self.log.setLevel(logging.DEBUG if self.valves.debug_valve else logging.INFO)

        self.log_debug(
            "[Pipe INIT] Consensus Pipe initialized with the following valves:"
        )
        self.log_debug(f"{self.valves}")

        # Initialize last_emit_time and start_time
        self.reset_state()

    def reset_state(self):
        """Reset state variables for reuse in new requests."""
        self.model_outputs = {}
        self.completed_contributors = set()
        self.interrupted_contributors = set()
        self.consensus_output = None
        self.last_summary_model_index = 0
        self.last_emit_time = 0.0
        self.stop_emitter.clear()
        self.output_buffer_tokens = []
        self.buffer = ""
        self.summary_in_progress = False
        self.final_status_emitted = False
        self.tags_detected = False
        self.consensus_streamed = False  # Reset the consensus streamed flag
        self.log_debug("[RESET_STATE] State variables have been reset.")
        self.start_time = time.time()  # Start time of the current pipe execution

    def log_debug(self, message: str):
        """Log debug messages with Request ID if available."""
        if self.valves.debug_valve:
            if hasattr(self, "request_id") and self.request_id:
                self.log.debug(f"[Request {self.request_id}] {message}")
            else:
                self.log.debug(message)

    def parse_tags(self, tags_str: str) -> List[str]:
        """
        Parse the comma-separated tags string into a list of tags.

        Args:
            tags_str (str): Comma-separated tags.

        Returns:
            List[str]: List of trimmed tags, excluding the placeholder.
        """
        if tags_str.strip() == TAGS_PLACEHOLDER:
            self.log_debug(
                "[PARSE_TAGS] Placeholder detected. No meaningful tags provided."
            )
            return []  # Return an empty list if the placeholder is detected

        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        self.log_debug(f"[PARSE_TAGS] Parsed tags: {tags}")
        return tags

    def determine_mode(self, meta_model_id: str) -> str:
        """
        Determine the selection mode based on the meta-model ID.

        Args:
            meta_model_id (str): The meta-model ID stripped of the manifold prefix.

        Returns:
            str: The selection mode ('random', 'explicit', 'tags', 'unsupported').
        """
        if meta_model_id.startswith("random"):
            return "random"
        elif meta_model_id.startswith("explicit"):
            return "explicit"
        elif meta_model_id.startswith("tag"):
            return "tags"
        else:
            return "unsupported"

    async def determine_contributing_models(
        self, body: dict, __user__: Optional[dict] = None
    ) -> List[Dict[str, str]]:
        """
        Determine the contributing models to query based on the selected meta-model.

        Args:
            body (dict): The incoming request payload.
            __user__ (Optional[dict]): User information.

        Returns:
            List[Dict[str, str]]: List of contributing models to query.
        """
        self.log_debug("[DETERMINE_CONTRIBUTORS] Starting model selection process.")

        # Extract and strip the meta-model ID using the manifold prefix
        model_id = body.get("model", "")
        if self.valves.manifold_prefix in model_id:
            meta_model = model_id.split(self.valves.manifold_prefix, 1)[1]  # STRIP_OPERATION: Strips prefix
            self.log_debug(f"Stripped meta-model ID: {meta_model}")
        else:
            meta_model = model_id
            self.log_debug(f"Meta-model ID remains unchanged: {meta_model}")

        # Determine the mode
        mode = self.determine_mode(meta_model)
        self.current_mode = mode  # Store the mode in self for later use
        self.log_debug(f"[DETERMINE_CONTRIBUTORS] Determined selection mode: {mode}")

        # Select models based on the mode
        if mode == "random":
            # Fetch random contributing models dynamically
            try:
                selected_model_ids = await self.fetch_available_contributing_models()

                # Log the selected model IDs
                self.log_debug(f"[DETERMINE_CONTRIBUTORS] Selected model IDs: {selected_model_ids}")

                # Create a list of model dictionaries with 'id' and 'name'
                selected_models = [
                    {"id": model_id, "name": f"random-{model_id}"}
                    for model_id in selected_model_ids
                ]

                self.log_debug(
                    f"[DETERMINE_CONTRIBUTORS] Selected random contributing models: {selected_models}"
                )
                return selected_models
            except Exception as e:
                self.log_debug(
                    f"[DETERMINE_CONTRIBUTORS] Error selecting random contributing models: {e}"
                )
                raise ValueError(
                    f"[DETERMINE_CONTRIBUTORS] Failed to fetch random contributing models: {e}"
                )

        elif mode == "explicit":
            # Use explicit contributing models defined in the valves
            if not self.valves.explicit_contributing_models:
                raise ValueError(
                    "[DETERMINE_CONTRIBUTORS] No explicit contributing models configured."
                )
            models = [
                {"id": model, "name": f"explicit-{model}"}
                for model in self.valves.explicit_contributing_models
            ]
            self.log_debug(
                f"[DETERMINE_CONTRIBUTORS] Selected explicit contributing models: {models}"
            )
            return models

        elif mode == "tags":
            # Use tag-based contributing model selection
            tags = self.parse_tags(self.valves.contributor_tags)
            if not tags:
                raise ValueError(
                    "[DETERMINE_CONTRIBUTORS] No valid tags provided for contributing models."
                )
            models = [{"id": tag, "name": f"tag-{tag}"} for tag in tags]
            self.log_debug(
                f"[DETERMINE_CONTRIBUTORS] Selected tag-based contributing models: {models}"
            )
            return models

        else:
            # Invalid or unsupported mode
            error_message = (
                f"[DETERMINE_CONTRIBUTORS] Unsupported selection mode: {mode}"
            )
            self.log_debug(error_message)
            raise ValueError(error_message)

    async def fetch_available_contributing_models(self) -> List[str]:
        """
        Fetch available contributing models from the Models router and select a random subset
        with an emphasis on distributing selection across unique urlIdx values,
        while excluding models matching the blacklist regex.

        Returns:
            List[str]: List of selected contributing model IDs.

        Raises:
            ValueError: If no models are available or insufficient models are found.
        """
        try:
            self.log_debug("[FETCH_CONTRIBUTORS] Fetching available contributing models...")
            available_models_response = await get_models(user=mock_user)

            # Extract the list of models from the "data" key
            if isinstance(available_models_response, dict) and "data" in available_models_response:
                available_models = available_models_response["data"]
            else:
                raise ValueError("[FETCH_CONTRIBUTORS] Invalid response structure from get_models.")

            self.log_debug(f"[FETCH_CONTRIBUTORS] Raw available_models: {available_models}")

            # Apply blacklist regex to exclude certain models
            blacklist_pattern = self.valves.random_contributing_models_blacklist_regex
            try:
                blacklist_regex = re.compile(blacklist_pattern, re.IGNORECASE)
                self.log_debug(f"[FETCH_CONTRIBUTORS] Compiled blacklist regex: {blacklist_pattern}")
            except re.error as e:
                self.log_debug(f"[FETCH_CONTRIBUTORS] Invalid blacklist regex pattern: {blacklist_pattern}. Error: {e}")
                raise ValueError(f"[FETCH_CONTRIBUTORS] Invalid blacklist regex pattern: {blacklist_pattern}. Error: {e}")

            # Normalize and filter out models that match the blacklist regex
            filtered_models = [
                model for model in available_models
                if not blacklist_regex.search(model.get("id", "").lower())  # Normalize ID to lowercase
            ]

            self.log_debug(f"[FETCH_CONTRIBUTORS] Models after applying blacklist: {[model['id'] for model in filtered_models]}")

            if not filtered_models:
                raise ValueError("[FETCH_CONTRIBUTORS] No models available after applying blacklist.")

            # Extract models and group by urlIdx
            models_by_urlIdx = {}
            for model in filtered_models:
                urlIdx = model.get("urlIdx", -1)  # Default to -1 if urlIdx is missing
                if urlIdx not in models_by_urlIdx:
                    models_by_urlIdx[urlIdx] = []
                models_by_urlIdx[urlIdx].append(model["id"])

            self.log_debug(f"[FETCH_CONTRIBUTORS] Models grouped by urlIdx: {models_by_urlIdx}")

            # Ensure there are enough unique urlIdx groups
            unique_urlIdx_count = len(models_by_urlIdx)
            if unique_urlIdx_count == 0:
                raise ValueError("[FETCH_CONTRIBUTORS] No valid urlIdx found in available models.")

            # Determine how many models to select
            num_models_to_select = min(
                self.valves.random_contributing_models_number,
                sum(len(models) for models in models_by_urlIdx.values())  # Total available models
            )
            self.log_debug(f"[FETCH_CONTRIBUTORS] Number of models to select: {num_models_to_select}")

            if num_models_to_select <= 0:
                raise ValueError("[FETCH_CONTRIBUTORS] Number of models to select must be at least 1.")

            # Distribute selection across urlIdx groups
            selected_model_ids = []
            for urlIdx, models in models_by_urlIdx.items():
                # Calculate the proportion of models to select from this group
                proportion = max(1, num_models_to_select // unique_urlIdx_count)
                selected_models = random.sample(models, min(proportion, len(models)))
                selected_model_ids.extend(selected_models)
                self.log_debug(f"[FETCH_CONTRIBUTORS] Selected models from urlIdx {urlIdx}: {selected_models}")

            # Shuffle to add more randomness
            random.shuffle(selected_model_ids)

            # Limit to the desired number of models
            selected_model_ids = selected_model_ids[:num_models_to_select]
            self.log_debug(f"[FETCH_CONTRIBUTORS] Final selected models: {selected_model_ids}")

            return selected_model_ids

        except Exception as e:
            self.log_debug(f"[FETCH_CONTRIBUTORS] Error fetching contributing models: {e}")
            raise ValueError(f"[FETCH_CONTRIBUTORS] Failed to fetch random contributing models: {e}")

    async def prepare_payloads(
        self, models: List[Dict[str, str]], messages: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Prepare payloads for each contributing model to be queried.

        Args:
            models (List[Dict[str, str]]): List of contributing models with 'id' and 'name'.
            messages (List[Dict[str, Any]]): Messages to include in the payloads.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of payloads keyed by model ID.
        """
        self.log_debug(
            f"[PREPARE_PAYLOADS] Preparing payloads for contributing models: {models}"
        )

        # Prepare payloads for each model
        payloads = {
            model["id"]: {
                "model": model["id"],
                "messages": [
                    {
                        "role": message["role"],
                        "content": (
                            self.clean_message_content(message.get("content", ""))
                            if self.valves.strip_collapsible_tags
                            and message["role"] == "assistant"  # STRIP_OPERATION: Conditional stripping
                            else message.get("content", "")
                        ),
                    }
                    for message in messages
                ],
                "stream": True,  # Enable streaming for contributing models
            }
            for model in models
        }

        self.log_debug(
            f"[PREPARE_PAYLOADS] Prepared payloads for contributing models: {list(payloads.keys())}"
        )
        return payloads

    def pipes(self) -> List[dict]:
        """
        Register the categories of models available for consensus.

        Returns:
            List[dict]: A list of model categories with their IDs and descriptions.
        """
        model_categories = []

        # Register random contributing models if enabled
        if self.valves.enable_random_contributors:
            model_categories.append(
                {"id": "consensus/random", "name": "Random"}
            )
            self.log_debug("[PIPES] Registered category: consensus/random")

        # Register explicit contributing models if enabled
        if (
            self.valves.enable_explicit_contributors
            and self.valves.explicit_contributing_models
        ):
            model_categories.append(
                {"id": "consensus/explicit", "name": "Explicit"}
            )
            self.log_debug("[PIPES] Registered category: consensus/explicit")

        # Register tag-based contributing models if enabled
        if self.valves.enable_tagged_contributors and self.valves.contributor_tags:
            tags = self.parse_tags(self.valves.contributor_tags)
            for tag in tags:
                model_categories.append(
                    {
                        "id": f"consensus/tag-{tag}",
                        "name": f"Tag: {tag}",
                    }
                )
                self.log_debug(f"[PIPES] Registered category: consensus/tag-{tag}")

        if not model_categories:
            self.log_debug("[PIPES] No categories registered.")
            raise ValueError("[PIPES] No categories available to register.")

        self.log_debug(f"[PIPES] Final registered categories: {model_categories}")
        return model_categories

    async def query_contributing_model(
        self, model_id: str, payload: dict, __event_emitter__: Callable[[dict], Awaitable[None]]
    ) -> Dict[str, str]:
        """
        Query a contributing model and emit its collapsible output immediately upon receiving the response.

        Args:
            model_id (str): The ID of the contributing model.
            payload (dict): The payload to send to the contributing model.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.

        Returns:
            Dict[str, str]: The output from the contributing model or an error message.
        """
        self.log_debug(
            f"[QUERY_CONTRIBUTOR] Querying contributing model: {model_id} with payload: {payload}"
        )
        try:
            # Log that we're about to query the model
            self.log_debug(f"[QUERY_CONTRIBUTOR] Attempting to query model {model_id}...")

            response = await generate_chat_completions(
                form_data=payload, user=mock_user, bypass_filter=True
            )

            if isinstance(response, StreamingResponse):
                self.log_debug(
                    f"[QUERY_CONTRIBUTOR] Received StreamingResponse from {model_id}. Processing stream."
                )
                collected_output = await self.handle_streaming_response(response)
                self.log_debug(
                    f"[QUERY_CONTRIBUTOR] Collected output from {model_id}: {collected_output}"
                )
            elif isinstance(response, dict):
                # Correctly access the nested "content" key
                try:
                    collected_output = response["choices"][0]["message"]["content"]
                    self.log_debug(
                        f"[QUERY_CONTRIBUTOR] Received output from {model_id}: {collected_output}"
                    )
                except (KeyError, IndexError) as e:
                    self.log_debug(
                        f"[QUERY_CONTRIBUTOR] Missing 'content' key in response from {model_id}: {e}"
                    )
                    return {"model": model_id, "error": "Missing 'content' key in response."}
            else:
                self.log_debug(
                    f"[QUERY_CONTRIBUTOR] Unexpected response type: {type(response)}"
                )
                return {"model": model_id, "error": "Unexpected response type."}

            # Emit collapsible for this contributing model without <pre> tags
            collapsible_content = f"""
<details>
<summary>Model {model_id} Output</summary>
{self.escape_html(collected_output)}
</details>
""".strip()

            self.log_debug(f"[QUERY_CONTRIBUTOR] Emitting collapsible for model {model_id}.")

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": collapsible_content},
                }
            )
            self.log_debug(f"[QUERY_CONTRIBUTOR] Emitted collapsible for model {model_id}.")
            return {"model": model_id, "output": collected_output}

        except Exception as e:
            self.log_debug(
                f"[QUERY_CONTRIBUTOR] Error querying contributing model {model_id}: {e}"
            )
            # Check if the exception is a 429 error
            if isinstance(e, aiohttp.ClientResponseError) and e.status == 429:
                return {"model": model_id, "error": "rate_limit"}
            return {"model": model_id, "error": str(e)}

    async def handle_json_response(self, response: Union[JSONResponse, bytes, str]) -> dict:
        """
        Handle and parse a JSON response from a JSONResponse object or raw content.

        Args:
            response (JSONResponse or bytes or str): The response to parse.

        Returns:
            dict: Parsed JSON data.

        Raises:
            TypeError: If the response content is not bytes or str.
            ValueError: If the response does not contain valid JSON.
        """
        if isinstance(response, JSONResponse):
            try:
                raw_content = response.body  # Access the bytes content
                decoded_body = raw_content.decode("utf-8")
                self.log_debug(f"[HANDLE_JSON_RESPONSE] Decoded body: {decoded_body}")
            except UnicodeDecodeError as e:
                self.log_debug(f"[HANDLE_JSON_RESPONSE] Chunk decode error: {e}")
                raise ValueError("Failed to decode bytes to string.") from e
        elif isinstance(response, bytes):
            try:
                decoded_body = response.decode("utf-8")
            except UnicodeDecodeError as e:
                self.log_debug(f"[HANDLE_JSON_RESPONSE] Chunk decode error: {e}")
                raise ValueError("Failed to decode bytes to string.") from e
        elif isinstance(response, str):
            decoded_body = response
        else:
            self.log_debug(
                f"[HANDLE_JSON_RESPONSE] Unexpected content type: {type(response)}"
            )
            raise TypeError("Expected response content to be bytes or str.")

        try:
            # Parse the JSON content
            response_data = json.loads(decoded_body)
            self.log_debug(f"[HANDLE_JSON_RESPONSE] Parsed response data: {response_data}")
            return response_data
        except json.JSONDecodeError as e:
            self.log_debug(
                f"[HANDLE_JSON_RESPONSE] JSON decoding error: {e}. Response content: {decoded_body[:100]}"
            )
            raise ValueError("Invalid JSON in response.") from e

    async def handle_streaming_response(self, response: StreamingResponse) -> str:
        """
        Handle and accumulate a StreamingResponse.

        Args:
            response (StreamingResponse): The StreamingResponse object.

        Returns:
            str: The accumulated response content.
        """
        collected_output = ""

        async for chunk in response.body_iterator:
            try:
                # Decode chunk to string if it's bytes
                if isinstance(chunk, bytes):
                    try:
                        chunk_str = chunk.decode("utf-8")
                    except UnicodeDecodeError as e:
                        self.log_debug(f"[HANDLE_STREAMING_RESPONSE] Chunk decode error: {e}")
                        continue
                elif isinstance(chunk, str):
                    chunk_str = chunk
                else:
                    self.log_debug(f"[HANDLE_STREAMING_RESPONSE] Unexpected chunk type: {type(chunk)}")
                    continue

                self.log_debug(f"[HANDLE_STREAMING_RESPONSE] Received chunk: {chunk_str}")

                # Strip 'data: ' prefix if present
                if chunk_str.startswith("data: "):
                    chunk_str = chunk_str[6:]

                # End the stream if '[DONE]' signal is received
                if chunk_str.strip() == "[DONE]":
                    self.log_debug("[HANDLE_STREAMING_RESPONSE] Received [DONE] signal. Ending stream.")
                    break

                # Skip empty chunks
                if not chunk_str.strip():
                    self.log_debug("[HANDLE_STREAMING_RESPONSE] Empty chunk received. Skipping.")
                    continue

                # Parse JSON content
                try:
                    data = json.loads(chunk_str)
                    choice = data.get("choices", [{}])[0]
                    message_content = ""

                    if "delta" in choice and "content" in choice["delta"]:
                        message_content = choice["delta"]["content"]
                    elif "message" in choice and "content" in choice["message"]:
                        message_content = choice["message"]["content"]

                    # Append valid content to the collected output
                    if message_content:
                        collected_output += message_content
                        self.log_debug(f"[HANDLE_STREAMING_RESPONSE] Accumulated content: {message_content}")
                    else:
                        self.log_debug(f"[HANDLE_STREAMING_RESPONSE] No content found in chunk: {chunk_str}")

                except json.JSONDecodeError as e:
                    self.log_debug(f"[HANDLE_STREAMING_RESPONSE] JSON decoding error: {e} for chunk: {chunk_str}")
                    # Append the raw chunk to collected_output if it isn't JSON
                    collected_output += f"[Unparsed Chunk]: {chunk_str}\n"
                except Exception as e:
                    self.log_debug(f"[HANDLE_STREAMING_RESPONSE] Error processing JSON chunk: {e}")

            except Exception as e:
                self.log_debug(f"[HANDLE_STREAMING_RESPONSE] Unexpected error: {e}")

        self.log_debug(f"[HANDLE_STREAMING_RESPONSE] Final collected output: {collected_output.strip()}")
        return collected_output.strip()

    async def generate_consensus(self, __event_emitter__, consensus_model_id: str):
        """
        Generate consensus using the designated consensus model.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            consensus_model_id (str): The ID of the consensus model to use.
        """
        self.log_debug(
            f"[GENERATE_CONSENSUS] Called with emitter: {__event_emitter__} and model ID: {consensus_model_id}"
        )

        # Validate model outputs
        if not self.model_outputs:
            self.log_debug(
                "[GENERATE_CONSENSUS] No model outputs available for consensus."
            )
            await self.emit_status(
                __event_emitter__,
                "Error",
                "No model outputs available for consensus.",
                done=True,
            )
            return

        # Prepare a clear delineation of outputs
        delineated_outputs = "\n\n".join(
            f"Model {model_id}:\n{output}"
            for model_id, output in self.model_outputs.items()
        )

        # Prepare the payload for the consensus model
        payload = {
            "model": consensus_model_id,
            "messages": [
                {
                    "role": "system",
                    "content": self.valves.consensus_system_prompt
                    or CONSENSUS_SYSTEM_PROMPT_DEFAULT,
                },
                {
                    "role": "user",
                    "content": delineated_outputs,
                },
            ],
            "stream": True,  # Enable streaming for consensus
        }

        try:
            self.log_debug(f"[GENERATE_CONSENSUS] Payload for consensus model: {payload}")

            # Call the consensus generation function directly
            consensus_response = await generate_chat_completions(
                form_data=payload, user=mock_user
            )

            # Handle based on response type
            if isinstance(consensus_response, dict):
                self.log_debug(f"[GENERATE_CONSENSUS] Response as dict: {consensus_response}")
                try:
                    self.consensus_output = consensus_response["choices"][0]["message"]["content"]
                    await self.emit_output(
                        __event_emitter__,
                        self.consensus_output,
                        include_collapsible=False,  # Do not emit collapsibles here
                    )
                    await self.emit_status(__event_emitter__, "Completed", done=True)
                except (KeyError, IndexError) as e:
                    error_detail = consensus_response.get("detail", str(e))
                    self.log_debug(f"[GENERATE_CONSENSUS] Consensus response missing key: {error_detail}")
                    await self.emit_status(
                        __event_emitter__,
                        "Error",
                        f"Consensus generation failed: {error_detail}",
                        done=True,
                    )

            elif isinstance(consensus_response, StreamingResponse):
                self.log_debug("[GENERATE_CONSENSUS] Handling StreamingResponse")
                self.consensus_streamed = True  # Set the flag to indicate streaming
                self.consensus_output = await self.handle_streaming_response(consensus_response)
                await self.emit_output(
                    __event_emitter__,
                    self.consensus_output,
                    include_collapsible=False,  # Do not emit collapsibles here
                )
                await self.emit_status(__event_emitter__, "Completed", done=True)

            else:
                self.log_debug(f"[GENERATE_CONSENSUS] Unexpected response type: {type(consensus_response)}")
                await self.emit_status(
                    __event_emitter__,
                    "Error",
                    f"Unexpected response type: {type(consensus_response)}",
                    done=True,
                )

        except Exception as e:
            # Log and emit the error
            self.log_debug(f"[GENERATE_CONSENSUS] Error generating consensus: {e}")
            await self.emit_status(
                __event_emitter__,
                "Error",
                f"Consensus generation failed: {str(e)}",
                done=True,
            )

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
            formatted_message = "Seeking Consensus"
        elif done:
            formatted_message = f"Consensus took {time_suffix}"
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
        self.log_debug(f"[EMIT_STATUS] Attempting to emit status: {formatted_message}")

        # Verify __event_emitter__ is callable
        if not callable(__event_emitter__):
            self.log_debug(f"[EMIT_STATUS] __event_emitter__ is not callable: {type(__event_emitter__)}")
            raise TypeError(f"__event_emitter__ must be callable, got {type(__event_emitter__)}")

        try:
            await __event_emitter__(event)
            self.log_debug("[EMIT_STATUS] Status emitted successfully.")
            # Update last_emit_time after successful emission
            self.last_emit_time = current_time
        except Exception as e:
            self.log_debug(f"[EMIT_STATUS] Error emitting status: {e}")

    async def emit_collapsible(self, __event_emitter__) -> None:
        """
        Emit a collapsible section containing outputs from all contributing models.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        if not self.model_outputs:
            self.log_debug("[EMIT_COLLAPSIBLE] No model outputs to emit.")
            return

        # Generate collapsible sections for each model without <pre> tags
        collapsible_content = "\n".join(
            f"""
<details>
<summary>Model {model_id} Output</summary>
{self.escape_html(output)}
</details>
""".strip()
            for model_id, output in self.model_outputs.items()
        )

        self.log_debug("[EMIT_COLLAPSIBLE] Collapsible content generated.")

        # Prepare the collapsible message
        message_event = {
            "type": "message",
            "data": {"content": collapsible_content},
        }

        # Verify __event_emitter__ is callable
        if not callable(__event_emitter__):
            self.log_debug(f"[EMIT_COLLAPSIBLE] __event_emitter__ is not callable: {type(__event_emitter__)}")
            raise TypeError(f"__event_emitter__ must be callable, got {type(__event_emitter__)}")

        try:
            await __event_emitter__(message_event)
            self.log_debug("[EMIT_COLLAPSIBLE] Collapsible emitted successfully.")
        except Exception as e:
            self.log_debug(f"[EMIT_COLLAPSIBLE] Error emitting collapsible: {e}")

    async def emit_output(
        self, __event_emitter__, content: str, include_collapsible: bool = False
    ) -> None:
        """
        Emit content and optionally include a collapsible section with model outputs.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            content (str): The main content to emit.
            include_collapsible (bool): Whether to include collapsibles with the output.
        """
        if __event_emitter__ and content:
            # Verify __event_emitter__ is callable
            if not callable(__event_emitter__):
                self.log_debug(f"[EMIT_OUTPUT] __event_emitter__ is not callable: {type(__event_emitter__)}")
                raise TypeError(f"__event_emitter__ must be callable, got {type(__event_emitter__)}")

            # Clean the main content by removing only newline characters
            content_cleaned = content.replace("\n", "")  # ADJUSTED: Removes only newline characters
            self.log_debug(f"[EMIT_OUTPUT] Cleaned content: {content_cleaned}")

            # Prepare the message event
            main_message_event = {
                "type": "message",
                "data": {"content": content_cleaned},
            }

            try:
                # Emit the main content
                await __event_emitter__(main_message_event)
                self.log_debug(
                    "[EMIT_OUTPUT] Main output message emitted successfully."
                )

                # Optionally emit collapsible content
                if include_collapsible and self.valves.use_collapsible:
                    await self.emit_collapsible(__event_emitter__)
            except Exception as e:
                self.log_debug(
                    f"[EMIT_OUTPUT] Error emitting output or collapsible: {e}"
                )

    def escape_html(self, text: str) -> str:
        """
        Escape HTML characters in the text to prevent rendering issues.

        Args:
            text (str): The text to escape.

        Returns:
            str: Escaped text.
        """
        return html.escape(text)

    def clean_message_content(self, content: str) -> str:
        """
        Remove <details> HTML tags and their contents from the message content.

        Args:
            content (str): The message content.

        Returns:
            str: Cleaned message content.
        """
        # STRIP_OPERATION: Removes <details> tags and their contents using regex
        details_pattern = r"<details>\s*<summary>.*?</summary>\s*.*?</details>"
        return re.sub(details_pattern, "", content, flags=re.DOTALL | re.IGNORECASE)  # ADJUSTED: Added re.IGNORECASE

    def format_elapsed_time(self, elapsed: float) -> str:
        """
        Format elapsed time into a readable string.

        Args:
            elapsed (float): Elapsed time in seconds.

        Returns:
            str: Formatted time string.
        """
        minutes, seconds = divmod(int(elapsed), 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    async def emit_final_status(self, __event_emitter__):
        """
        Emit the final status message indicating the duration of the consensus process.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        if self.final_status_emitted:
            self.log_debug("[FINAL_STATUS] Final status already emitted.")
            return

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time

        # Format elapsed time
        minutes, seconds = divmod(int(elapsed_time), 60)
        if minutes > 0:
            time_suffix = f"{minutes}m {seconds}s"
        else:
            time_suffix = f"{seconds}s"

        # Prepare the final status message
        formatted_message = f"Consensus completed in {time_suffix}"
        self.log_debug(f"[FINAL_STATUS] Emitting final status: {formatted_message}")

        # Emit the final status
        await self.emit_status(__event_emitter__, "Info", formatted_message, done=True)
        self.final_status_emitted = True

    def check_active_tasks(self) -> bool:
        """
        Check if there are any active tasks or completed tasks in the pipeline.

        Returns:
            bool: True if tasks are active or completed, False otherwise.
        """
        if self.completed_contributors or self.model_outputs:
            return True
        return False

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> Dict[str, Any]:
        """
        Main handler for processing requests with retry logic for random contributing models.

        Steps:
        1. Reset state variables for each new request.
        2. Validate selection methods.
        3. Determine contributing models to query based on valves and configuration.
        4. Handle system messages, including overrides if enabled.
        5. Prepare payloads for the selected contributing models.
        6. Emit initial status: "Seeking Consensus".
        7. Query the selected contributing models while managing retries for random mode.
        8. Emit contribution completion status.
        9. Generate consensus using the selected consensus model after all contributing models complete.
        10. Emit consensus completion status.
        11. Return the consensus output or an error message in a standardized format.

        Args:
            body (dict): The incoming request payload.
            __user__ (Optional[dict]): The user information.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.

        Returns:
            Dict[str, Any]: The consensus output or an error message.
        """
        # Reset state for the new request
        self.reset_state()
        self.request_id = uuid.uuid4()
        self.log_debug(f"[PIPE] Starting new request with ID: {self.request_id}")

        try:
            # Step 1: Validate Selection Methods
            if not (
                self.valves.enable_random_contributors
                or self.valves.enable_explicit_contributors
                or self.valves.enable_tagged_contributors
            ):
                raise ValueError(
                    "[PIPE] At least one contributor selection method must be enabled."
                )

            # Step 2: Determine Contributing Models
            contributing_models = await self.determine_contributing_models(
                body, __user__
            )
            if not contributing_models:
                raise ValueError("[PIPE] No contributing models selected for query.")
            self.log_debug(
                f"[PIPE] Selected contributing models: {contributing_models}"
            )

            # Step 3: Handle System Messages
            body_messages = body.get("messages", [])
            system_message, messages = pop_system_message(body_messages)

            self.messages = messages

            # Preprocess messages
            processed_messages = [
                {
                    "role": message["role"],
                    "content": (
                        self.clean_message_content(message.get("content", ""))
                        if self.valves.strip_collapsible_tags
                        and message["role"] == "assistant"  # STRIP_OPERATION: Conditional stripping
                        else message.get("content", "")
                    ),
                }
                for message in messages
            ]

            # Step 4: Prepare Payloads for Contributing Models
            payloads = await self.prepare_payloads(
                contributing_models, processed_messages
            )
            self.log_debug("[PIPE] Prepared payloads for contributing models.")

            # Step 5: Emit Initial Status
            await self.emit_status(
                __event_emitter__,
                "Info",
                "Seeking Consensus",
                initial=True,
                done=False,
            )
            self.log_debug("[PIPE] Initial status 'Seeking Consensus' emitted.")

            # Step 6: Query Contributing Models with Retry Logic
            retries = {model["id"]: 0 for model in contributing_models}
            max_retries = self.valves.max_reroll_retries
            remaining_models = contributing_models.copy()

            active_tasks = []
            # To keep track of models already tried to avoid infinite loops
            failed_models = set()

            while remaining_models:
                query_tasks = [
                    asyncio.create_task(
                        self.query_contributing_model(
                            model["id"], payloads[model["id"]], __event_emitter__
                        )
                    )
                    for model in remaining_models
                ]
                active_tasks.extend(query_tasks)
                self.log_debug(
                    f"[PIPE] Querying contributing models: {[model['id'] for model in remaining_models]}"
                )

                results = await asyncio.gather(*query_tasks, return_exceptions=True)

                failed_contributors = []
                new_remaining_models = []

                for i, result in enumerate(results):
                    model = remaining_models[i]
                    model_id = model["id"]

                    if isinstance(result, Exception):
                        self.log_debug(
                            f"[PIPE] Error querying model {model_id}: {result}"
                        )
                        if retries[model_id] < max_retries:
                            retries[model_id] += 1
                            failed_contributors.append(model)
                            self.log_debug(
                                f"[PIPE] Retrying model {model_id} (Attempt {retries[model_id]})"
                            )
                            await asyncio.sleep(self.valves.reroll_backoff_delay)
                        else:
                            self.log_debug(
                                f"[PIPE] Model {model_id} exceeded retry limit."
                            )
                            self.interrupted_contributors.add(model_id)
                    else:
                        output = result.get("output", "")
                        error = result.get("error", "")
                        if error:
                            if error == "rate_limit":
                                self.log_debug(
                                    f"[PIPE] Model {model_id} rate limited (429). Marking as failed."
                                )
                                self.interrupted_contributors.add(model_id)
                                # Attempt to select a replacement model
                                replacement_model = await self.select_replacement_model(model_id, failed_models)
                                if replacement_model:
                                    new_remaining_models.append(replacement_model)
                                    self.log_debug(
                                        f"[PIPE] Selected replacement model {replacement_model['id']} for {model_id}."
                                    )
                            elif retries[model_id] < max_retries:
                                retries[model_id] += 1
                                failed_contributors.append(model)
                                self.log_debug(
                                    f"[PIPE] Retrying model {model_id} due to error: {error} (Attempt {retries[model_id]})"
                                )
                                await asyncio.sleep(self.valves.reroll_backoff_delay)
                            else:
                                self.log_debug(
                                    f"[PIPE] Model {model_id} encountered error: {error} and exceeded retry limit."
                                )
                                self.interrupted_contributors.add(model_id)
                        elif output:
                            self.model_outputs[model_id] = output
                            self.completed_contributors.add(model_id)
                            self.log_debug(
                                f"[PIPE] Model {model_id} completed successfully with output."
                            )
                        else:
                            self.log_debug(
                                f"[PIPE] Model {model_id} returned no output."
                            )
                            self.interrupted_contributors.add(model_id)

                # Update remaining_models with failed contributors and new replacements
                remaining_models = failed_contributors + new_remaining_models

            # Ensure all active tasks are completed before proceeding
            await asyncio.gather(*active_tasks, return_exceptions=True)
            self.log_debug(f"[PIPE] All contributing model tasks are completed.")

            # Step 7: Emit Contribution Completion Status
            contribution_time = time.time() - self.start_time
            contribution_time_str = self.format_elapsed_time(contribution_time)
            await self.emit_status(
                __event_emitter__,
                "Info",
                f"Contribution took {contribution_time_str}",
                done=False,
            )
            self.log_debug(
                f"[PIPE] Emitted contribution completion status: Contribution took {contribution_time_str}"
            )

            # Step 8: Generate Consensus
            consensus_model_id = (
                self.valves.consensus_model_id
                if self.valves.consensus_model_id != CONSENSUS_PLACEHOLDER
                else body.get("model")
            )
            if not consensus_model_id:
                raise ValueError("[PIPE] No valid consensus model ID provided.")

            self.log_debug(f"[PIPE] Using consensus model ID: {consensus_model_id}")
            await self.generate_consensus(__event_emitter__, consensus_model_id)

            # Step 9: Emit Consensus Completion Status
            consensus_time = time.time() - self.start_time
            consensus_time_str = self.format_elapsed_time(consensus_time)
            await self.emit_status(
                __event_emitter__,
                "Info",
                f"Consensus took {consensus_time_str}",
                done=True,
            )
            self.log_debug(
                f"[PIPE] Emitted consensus completion status: Consensus took {consensus_time_str}"
            )

            # Step 10: Emit Final Consensus Output as a Message (Prevent Duplicate Emission)
            if self.consensus_output and not self.consensus_streamed:
                await self.emit_output(__event_emitter__, self.consensus_output)
                self.log_debug("[PIPE] Final consensus output emitted as a message.")

            # Return the consensus output
            if self.consensus_output:
                self.log_debug("[PIPE] Consensus generation successful.")
                return {"status": "success", "data": self.consensus_output}
            else:
                self.log_debug("[PIPE] Consensus generation failed.")
                return {
                    "status": "error",
                    "message": "No consensus could be generated.",
                }

        except Exception as e:
            self.log_debug(f"[PIPE] Unexpected error: {e}")
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__,
                    "Error",
                    f"Error processing request: {str(e)}",
                    done=True,
                )
            return {"status": "error", "message": str(e)}

    async def select_replacement_model(self, failed_model_id: str, failed_models: set) -> Optional[Dict[str, str]]:
        """
        Select a replacement model that hasn't been failed yet.

        Args:
            failed_model_id (str): The ID of the model that failed.
            failed_models (set): Set of model IDs that have already failed.

        Returns:
            Optional[Dict[str, str]]: The replacement model dictionary or None if no replacement is available.
        """
        self.log_debug(f"[SELECT_REPLACEMENT] Selecting replacement for failed model {failed_model_id}.")
        try:
            # Fetch available contributing models again
            available_models_response = await get_models(user=mock_user)

            # Extract the list of models from the "data" key
            if isinstance(available_models_response, dict) and "data" in available_models_response:
                available_models = available_models_response["data"]
            else:
                self.log_debug("[SELECT_REPLACEMENT] Invalid response structure from get_models.")
                return None

            # Extract model IDs
            model_ids = [model["id"] for model in available_models if "id" in model]

            # Exclude already failed models and interrupted models
            excluded_models = failed_models.union(self.interrupted_contributors).union({failed_model_id})
            eligible_models = [model_id for model_id in model_ids if model_id not in excluded_models]

            if not eligible_models:
                self.log_debug("[SELECT_REPLACEMENT] No eligible models available for replacement.")
                return None

            # Select a random eligible model
            replacement_model_id = random.choice(eligible_models)
            self.log_debug(f"[SELECT_REPLACEMENT] Selected replacement model: {replacement_model_id}")

            return {"id": replacement_model_id, "name": f"random-{replacement_model_id}"}

        except Exception as e:
            self.log_debug(f"[SELECT_REPLACEMENT] Error selecting replacement model: {e}")
            return None
