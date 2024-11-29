"""
title: Consensus Manifold
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.5
"""

import os
import json
import time
import asyncio
import random
import uuid
import html
from typing import List, Union, Optional, Callable, Awaitable, Dict, Any
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message, add_or_update_system_message
from starlette.responses import StreamingResponse
from open_webui.main import (
    generate_chat_completions,
    generate_moa_response,
    get_task_model_id,
)
from open_webui.utils.misc import get_last_assistant_message, get_content_from_message
from open_webui.config import TASK_MODEL, TASK_MODEL_EXTERNAL
from open_webui.apps.webui.routers.chats import get_all_user_tags

import logging
import re  # For regex operations

from fastapi import HTTPException, status

# Mock the user object as expected by the function (OWUI 0.5.x constraint)
mock_user = {
    "id": "consensus_manifold",
    "username": "consensus_manifold",
    "role": "admin",
}

# Global Placeholders
CONSENSUS_PLACEHOLDER = "your-consensus-model-id-goes-here"
PROGRESS_SUMMARY_PLACEHOLDER = "your-progress-model-id-goes-here"
TAGS_PLACEHOLDER = "comma-separated-list-of-model-tags-goes-here"
CONSENSUS_SYSTEM_PROMPT_DEFAULT = (
    "You are finding consensus among the output of multiple models."
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
        enable_explicit_contributors: bool = Field(
            default=False,
            description="Enable explicit (user-specified) contributing model selection if provided (false by default).",
        )
        enable_tagged_contributors: bool = Field(
            default=False,
            description="Enable tag-based contributing model selection (false by default).",
        )
        explicit_contributing_models: List[str] = Field(
            default=[],
            description="List of specific contributing models to use when explicit selection is enabled.",
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
            description="Enable or disable stripping of <details> HTML tags from assistant messages.",
        )
        debug_valve: bool = Field(default=True, description="Enable debug logging.")

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

        # Initialize last_emit_time
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
        self.log_debug("[RESET_STATE] State variables have been reset.")

    def log_debug(self, message: str):
        """Log debug messages with Request ID if available."""
        if self.valves.debug_valve:
            if hasattr(self, "request_id") and self.request_id:
                self.log.debug(f"[Request {self.request_id}] {message}")
            else:
                self.log.debug(message)

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
            meta_model = model_id.split(self.valves.manifold_prefix, 1)[
                1
            ]  # Strip prefix
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
            # Fetch random contributing models dynamically (using explicit list)
            try:
                available_models = await self.fetch_available_contributing_models()
                if not available_models:
                    raise ValueError(
                        "[DETERMINE_CONTRIBUTORS] No random contributing models available."
                    )

                # Shuffle the available models for randomness
                random.shuffle(available_models)

                # Select unique models up to the minimum required
                selected_models = []
                selected_ids = set()
                for model in available_models:
                    if model not in selected_ids:
                        selected_models.append({"id": model, "name": f"random-{model}"})
                        selected_ids.add(model)
                    if len(selected_models) >= self.valves.minimum_number_of_models:
                        break

                if not selected_models:
                    raise ValueError(
                        "[DETERMINE_CONTRIBUTORS] No unique random contributing models could be selected."
                    )

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
        Fetch available contributing models from providers.
        For now, returns the explicit contributing models to simulate random selection.

        Returns:
            List[str]: List of available contributing model IDs.

        Raises:
            ValueError: If no explicit contributing models are configured.
        """
        try:
            if not self.valves.explicit_contributing_models:
                raise ValueError(
                    "[FETCH_CONTRIBUTORS] No explicit contributing models configured."
                )

            # Use explicit contributing models as the pool for random selection
            available_models = self.valves.explicit_contributing_models.copy()

            self.log_debug(
                f"[FETCH_CONTRIBUTORS] Available contributing models (explicit list): {available_models}"
            )
            return available_models
        except Exception as e:
            self.log_debug(
                f"[FETCH_CONTRIBUTORS] Error fetching contributing models: {e}"
            )
            raise

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
                            and message["role"] == "assistant"
                            else message.get("content", "")
                        ),
                    }
                    for message in messages
                ],
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
                {"id": "consensus/random", "name": "Random Contributing Models"}
            )
            self.log_debug("[PIPES] Registered category: consensus/random")

        # Register explicit contributing models if enabled
        if (
            self.valves.enable_explicit_contributors
            and self.valves.explicit_contributing_models
        ):
            model_categories.append(
                {"id": "consensus/explicit", "name": "Explicit Contributing Models"}
            )
            self.log_debug("[PIPES] Registered category: consensus/explicit")

        # Register tag-based contributing models if enabled
        if self.valves.enable_tagged_contributors and self.valves.contributor_tags:
            tags = self.parse_tags(self.valves.contributor_tags)
            for tag in tags:
                model_categories.append(
                    {
                        "id": f"consensus/tag-{tag}",
                        "name": f"Contributing Models for Tag: {tag}",
                    }
                )
                self.log_debug(f"[PIPES] Registered category: consensus/tag-{tag}")

        if not model_categories:
            self.log_debug("[PIPES] No categories registered.")
            raise ValueError("[PIPES] No categories available to register.")

        self.log_debug(f"[PIPES] Final registered categories: {model_categories}")
        return model_categories

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
        6. Query the selected contributing models while managing retries for random mode.
        7. Generate consensus using the selected consensus model after all contributing models complete.
        8. Return the consensus output or an error message in a standardized format.
        """
        self.reset_state()
        self.request_id = uuid.uuid4()
        self.log_debug(f"[PIPE] Initiating new pipe with Request ID: {self.request_id}")

        try:
            # Step 1: Validate Selection Methods
            if not (
                self.valves.enable_random_contributors
                or self.valves.enable_explicit_contributors
                or self.valves.enable_tagged_contributors
            ):
                raise ValueError(
                    "[PIPE] At least one of enable_random_contributors, enable_explicit_contributors, or enable_tagged_contributors must be enabled."
                )

            # Step 2: Determine Contributing Models to Query
            contributing_models = await self.determine_contributing_models(
                body, __user__
            )
            if not contributing_models:
                raise ValueError("[PIPE] No contributing models selected for query.")
            self.log_debug(
                f"[PIPE] Selected contributing models for query: {contributing_models}"
            )

            # Step 3: Handle System Messages
            body_messages = body.get("messages", [])
            system_message, messages = pop_system_message(body_messages)

            if (
                self.valves.enable_llm_summaries
                and self.valves.progress_summary_system_prompt
            ):
                add_or_update_system_message(
                    self.valves.progress_summary_system_prompt, messages
                )
                self.log_debug(
                    f"[PIPE] Overriding system message with: {self.valves.progress_summary_system_prompt}"
                )
            elif system_message:
                system_message_content = get_content_from_message(system_message)
                add_or_update_system_message(system_message_content, messages)
                self.log_debug(
                    f"[PIPE] Using provided system message: {system_message_content}"
                )

            self.log_debug(
                f"[PIPE] Total number of messages after processing system message: {len(messages)}"
            )

            # Step 4: Prepare Payloads
            payloads = await self.prepare_payloads(contributing_models, messages)

            # Step 5: Start the Progress Summarizer before querying
            if self.valves.enable_llm_summaries:
                progress_summary_task = asyncio.create_task(
                    self.progress_summary_manager(
                        contributing_models, __event_emitter__
                    )
                )
                self.log_debug("[PIPE] Progress summarizer task started.")
            else:
                progress_summary_task = None

            # Step 6: Query Contributing Models with Retry for Random Mode
            retries = {model["id"]: 0 for model in contributing_models}
            max_retries = self.valves.max_reroll_retries
            query_results = {}

            remaining_models = contributing_models.copy()

            while remaining_models:
                query_tasks = [
                    asyncio.create_task(
                        self.query_contributing_model(
                            model["id"], payloads[model["id"]]
                        )
                    )
                    for model in remaining_models
                ]

                results = await asyncio.gather(*query_tasks, return_exceptions=True)

                # Process results
                failed_contributors = []
                for i, result in enumerate(results):
                    model_id = remaining_models[i]["id"]

                    if isinstance(result, Exception):
                        self.log_debug(
                            f"[PIPE] Error querying contributing model {model_id}: {result}"
                        )
                        if retries[model_id] < max_retries:
                            retries[model_id] += 1
                            failed_contributors.append(
                                remaining_models[i]
                            )  # Retry this model
                            self.log_debug(
                                f"[PIPE] Retrying contributing model {model_id} (Retry {retries[model_id]})"
                            )
                            await asyncio.sleep(self.valves.reroll_backoff_delay)
                        else:
                            self.log_debug(
                                f"[PIPE] Contributing model {model_id} exceeded retry limit."
                            )
                            self.interrupted_contributors.add(model_id)
                    else:
                        output = result.get("output", "")
                        if "error" in result:
                            self.log_debug(
                                f"[PIPE] Contributing model {model_id} returned an error: {result['error']}"
                            )
                            self.interrupted_contributors.add(model_id)
                        else:
                            query_results[model_id] = output
                            self.completed_contributors.add(model_id)
                            self.log_debug(
                                f"[PIPE] Contributing model {model_id} completed successfully."
                            )

                # Update the list of contributing models to retry
                remaining_models = failed_contributors

            # Ensure all contributing models have completed
            self.model_outputs.update(query_results)

            # Step 7: Generate Consensus after all contributing models have completed
            consensus_model_id = (
                body.get("model")
                if body.get("model") != CONSENSUS_PLACEHOLDER
                else self.valves.consensus_model_id
            )
            self.log_debug(f"[PIPE] Using consensus model ID: {consensus_model_id}")

            await self.generate_consensus(__event_emitter__, consensus_model_id)

            # Wait for the progress summarizer to finish if it was started
            if self.valves.enable_llm_summaries and progress_summary_task:
                progress_summary_task.cancel()
                try:
                    await progress_summary_task
                except asyncio.CancelledError:
                    self.log_debug("[PIPE] Progress summarizer task cancelled.")

            # Step 8: Return the Consensus Output in a Standardized Format
            if self.consensus_output:
                return {"status": "success", "data": self.consensus_output}
            else:
                return {
                    "status": "error",
                    "message": "No consensus could be generated.",
                }

        except ValueError as ve:
            self.log_debug(f"[PIPE] ValueError: {ve}")
            if __event_emitter__:
                await self.emit_status(__event_emitter__, "Error", str(ve), done=True)
            return {"status": "error", "message": str(ve)}
        except Exception as e:
            self.log_debug(f"[PIPE] Exception: {e}")
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__,
                    "Error",
                    f"Error: {str(e)}",
                    done=True,
                )
            return {"status": "error", "message": str(e)}

    async def query_contributing_model(
        self, model_id: str, payload: dict
    ) -> Dict[str, str]:
        """
        Query a single contributing model with the provided payload.

        Args:
            model_id (str): The ID of the contributing model to query.
            payload (dict): The payload to send to the model.

        Returns:
            Dict[str, str]: A dictionary containing the model ID and its output or error.
        """
        self.log_debug(
            f"[QUERY_CONTRIBUTOR] Querying contributing model: {model_id} with payload: {payload}"
        )
        try:
            response = await generate_chat_completions(
                form_data=payload, user=mock_user, bypass_filter=True
            )
            output = response["choices"][0]["message"]["content"].rstrip("\n")
            self.log_debug(
                f"[QUERY_CONTRIBUTOR] Received output from {model_id}: {output}"
            )
            return {"model": model_id, "output": output}
        except Exception as e:
            self.log_debug(
                f"[QUERY_CONTRIBUTOR] Error querying contributing model {model_id}: {e}"
            )
            return {"model": model_id, "error": str(e)}

    async def progress_summary_manager(
        self,
        contributing_models: List[Dict[str, str]],
        __event_emitter__: Callable[[dict], Awaitable[None]],
    ):
        """
        Manage the emission of progress summaries.

        Args:
            contributing_models (List[Dict[str, str]]): List of contributing models being queried.
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        try:
            while True:
                await asyncio.sleep(self.valves.progress_summary_interval_seconds)
                if self.valves.enable_llm_summaries:
                    # Select the contributing model with the most buffered tokens
                    model_id = self.select_model_with_most_buffered_tokens()
                    if not model_id:
                        self.log_debug(
                            "[PROGRESS_SUMMARY] No contributing models left to summarize."
                        )
                        break  # Exit if no models are left to summarize

                    output = self.model_outputs.get(model_id, "No output yet.")
                    summary_payload = {
                        "model": self.valves.progress_summary_model_id,
                        "messages": [
                            {
                                "role": "system",
                                "content": self.valves.progress_summary_system_prompt,
                            },
                            {
                                "role": "user",
                                "content": f"{self.valves.progress_summary_status_prompt} Output from {model_id}: {output}",
                            },
                        ],
                    }
                    try:
                        summary_response = await generate_chat_completions(
                            form_data=summary_payload,
                            user=mock_user,
                            bypass_filter=True,
                        )
                        summary = (
                            summary_response["choices"][0]["message"]["content"]
                            .rstrip("\n")
                            .strip()
                        )
                        await self.emit_status(
                            __event_emitter__,
                            "Info",
                            f"Progress Summary for {model_id}: {summary}",
                        )
                        self.log_debug(
                            f"[PROGRESS_SUMMARY] Emitted summary for {model_id}: {summary}"
                        )
                    except Exception as e:
                        self.log_debug(
                            f"[PROGRESS_SUMMARY] Error generating summary for {model_id}: {e}"
                        )

                    # Update index for round-robin
                    self.last_summary_model_index = (
                        self.last_summary_model_index + 1
                    ) % len(contributing_models)
                else:
                    self.log_debug(
                        "[PROGRESS_SUMMARY_MANAGER] LLM summaries are disabled."
                    )
        except asyncio.CancelledError:
            self.log_debug(
                "[PROGRESS_SUMMARY_MANAGER] Progress summarizer task cancelled."
            )
        except Exception as e:
            self.log_debug(f"[PROGRESS_SUMMARY_MANAGER] Unexpected error: {e}")

    def select_model_with_most_buffered_tokens(self) -> Optional[str]:
        """
        Select the contributing model with the most buffered tokens for summarization.

        Returns:
            Optional[str]: The ID of the selected model, or None if no models are available.
        """
        if not self.model_outputs:
            return None

        # Get all contributing models that have completed
        available_models = list(self.model_outputs.keys())

        if not available_models:
            return None

        # Select the next model in round-robin
        model_id = available_models[
            self.last_summary_model_index % len(available_models)
        ]
        return model_id

    async def generate_consensus(self, __event_emitter__, consensus_model_id: str):
        """
        Generate a consensus response using the outputs from all completed contributing models.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): Event emitter for sending messages.
            consensus_model_id (str): The ID of the consensus model to use.
        """
        self.log_debug(
            f"[GENERATE_CONSENSUS] Called with emitter: {__event_emitter__} and model ID: {consensus_model_id}"
        )

        if not self.model_outputs:
            self.log_debug(
                "[GENERATE_CONSENSUS] No model outputs available for consensus."
            )
            return

        if not consensus_model_id or consensus_model_id == CONSENSUS_PLACEHOLDER:
            self.log_debug("[GENERATE_CONSENSUS] Falling back to use_moa_response.")
            await self.use_moa_response(__event_emitter__, consensus_model_id)
            return

        # Prepare the payload for the consensus model
        payload = {
            "model": consensus_model_id,
            "prompt": "Aggregate the following outputs to form a consensus response:",
            "responses": [
                {"model": model, "content": output}
                for model, output in self.model_outputs.items()
            ],
            "stream": False,
        }

        try:
            self.log_debug(
                f"[GENERATE_CONSENSUS] Payload for consensus model: {payload}"
            )

            # Launch the consensus generation coroutine
            consensus_response = await generate_moa_response(
                form_data=payload, user=mock_user
            )

            # Log and handle the raw response object
            self.log_debug(
                f"[GENERATE_CONSENSUS] Raw response object: {consensus_response}"
            )
            self.log_debug(
                f"[GENERATE_CONSENSUS] Response type: {type(consensus_response)}"
            )

            # Decode the response
            if hasattr(consensus_response, "json"):
                response_data = await consensus_response.json()
            else:
                response_data = consensus_response  # Assume it's already a dict

            self.log_debug(
                f"[GENERATE_CONSENSUS] Parsed response data: {response_data}"
            )

            # Extract the consensus output
            self.consensus_output = response_data.get("consensus", "").strip()
            self.log_debug(
                f"[GENERATE_CONSENSUS] Consensus output: {self.consensus_output}"
            )

            # Emit the consensus output
            await self.emit_output(
                __event_emitter__, self.consensus_output, include_collapsible=True
            )
        except Exception as e:
            self.log_debug(f"[GENERATE_CONSENSUS] Error generating consensus: {e}")
            await self.emit_status(
                __event_emitter__,
                "Error",
                f"Consensus generation failed: {str(e)}",
                done=True,
            )

    async def use_moa_response(self, __event_emitter__, consensus_model_id: str):
        """
        Use the current consensus_model_id for MOA response synthesis.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            consensus_model_id (str): The ID of the consensus model to use.
        """
        try:
            if not consensus_model_id or consensus_model_id == CONSENSUS_PLACEHOLDER:
                raise ValueError("[USE_MOA_RESPONSE] Model ID is not properly set.")

            payload = {
                "model": consensus_model_id,  # Use the consensus model ID
                "prompt": "Summarize the outputs below into a single response:",
                "responses": [
                    {"model": model, "content": output}
                    for model, output in self.model_outputs.items()
                ],
                "stream": False,
            }
            self.log_debug(f"[USE_MOA_RESPONSE] Fallback payload: {payload}")

            # Call the generate_moa_response function
            response = await generate_moa_response(form_data=payload, user=mock_user)

            # Handle the response format
            if hasattr(response, "json"):
                response_data = await response.json()
            else:
                response_data = response  # Assume it's already a dict

            self.consensus_output = response_data.get("consensus", "").strip()
            self.log_debug(
                f"[USE_MOA_RESPONSE] Consensus output: {self.consensus_output}"
            )
            await self.emit_output(
                __event_emitter__, self.consensus_output, include_collapsible=True
            )
        except Exception as e:
            self.log_debug(f"[USE_MOA_RESPONSE] Error synthesizing response: {e}")
            await self.emit_status(
                __event_emitter__,
                "Error",
                f"Fallback consensus generation failed: {str(e)}",
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
        elapsed_time = current_time - self.last_emit_time

        if elapsed_time < self.valves.emit_interval and not done:
            # Skip emitting if the interval hasn't passed and it's not a final update
            self.log_debug(
                f"[EMIT_STATUS] Skipping emission. Elapsed time: {elapsed_time:.2f}s, Emit interval: {self.valves.emit_interval}s."
            )
            return

        if __event_emitter__:
            if initial:
                formatted_message = "Seeking Consensus"
            elif done:
                time_suffix = self.format_elapsed_time(elapsed_time)
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
            self.log_debug(
                f"[EMIT_STATUS] Attempting to emit status: {formatted_message}"
            )

            try:
                await __event_emitter__(event)
                self.log_debug("[EMIT_STATUS] Status emitted successfully.")
                self.last_emit_time = current_time  # Update the last emission time
            except Exception as e:
                self.log_debug(f"[EMIT_STATUS] Error emitting status: {e}")

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

    async def emit_collapsible(self, __event_emitter__) -> None:
        """
        Emit a collapsible section containing outputs from all contributing models.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        if not self.model_outputs:
            self.log_debug("[EMIT_COLLAPSIBLE] No model outputs to emit.")
            return

        # Generate collapsible sections for each model
        collapsible_content = "\n".join(
            f"""
<details>
    <summary>Model {model_id} Output</summary>
    <pre>{self.escape_html(output.strip())}</pre>
</details>
""".strip()
            for model_id, output in self.model_outputs.items()
        )

        self.log_debug("[EMIT_COLLAPSIBLE] Collapsible content generated.")

        # Emit the collapsible message
        message_event = {
            "type": "message",
            "data": {"content": collapsible_content},
        }
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
            # Clean the main content
            content_cleaned = content.strip("\n").strip()
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
        # This ensures that existing collapsible tags are not nested or interfere with new ones
        details_pattern = r"<details>\s*<summary>.*?</summary>\s*.*?</details>"
        return re.sub(details_pattern, "", content, flags=re.DOTALL).strip()

    async def generate_consensus(self, __event_emitter__, consensus_model_id: str):
        """
        Generate a consensus response using the outputs from all completed contributing models.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): Event emitter for sending messages.
            consensus_model_id (str): The ID of the consensus model to use.
        """
        self.log_debug(
            f"[GENERATE_CONSENSUS] Called with emitter: {__event_emitter__} and model ID: {consensus_model_id}"
        )

        if not self.model_outputs:
            self.log_debug(
                "[GENERATE_CONSENSUS] No model outputs available for consensus."
            )
            return

        if not consensus_model_id or consensus_model_id == CONSENSUS_PLACEHOLDER:
            self.log_debug("[GENERATE_CONSENSUS] Falling back to use_moa_response.")
            await self.use_moa_response(__event_emitter__, consensus_model_id)
            return

        # Prepare the payload for the consensus model
        payload = {
            "model": consensus_model_id,
            "prompt": "Aggregate the following outputs to form a consensus response:",
            "responses": [
                {"model": model, "content": output}
                for model, output in self.model_outputs.items()
            ],
            "stream": False,
        }

        try:
            self.log_debug(
                f"[GENERATE_CONSENSUS] Payload for consensus model: {payload}"
            )

            # Launch the consensus generation coroutine
            consensus_response = await generate_moa_response(
                form_data=payload, user=mock_user
            )

            # Log and handle the raw response object
            self.log_debug(
                f"[GENERATE_CONSENSUS] Raw response object: {consensus_response}"
            )
            self.log_debug(
                f"[GENERATE_CONSENSUS] Response type: {type(consensus_response)}"
            )

            # Decode the response
            if hasattr(consensus_response, "json"):
                response_data = await consensus_response.json()
            else:
                response_data = consensus_response  # Assume it's already a dict

            self.log_debug(
                f"[GENERATE_CONSENSUS] Parsed response data: {response_data}"
            )

            # Extract the consensus output
            self.consensus_output = response_data.get("consensus", "").strip()
            self.log_debug(
                f"[GENERATE_CONSENSUS] Consensus output: {self.consensus_output}"
            )

            # Emit the consensus output
            await self.emit_output(
                __event_emitter__, self.consensus_output, include_collapsible=True
            )
        except Exception as e:
            self.log_debug(f"[GENERATE_CONSENSUS] Error generating consensus: {e}")
            await self.emit_status(
                __event_emitter__,
                "Error",
                f"Consensus generation failed: {str(e)}",
                done=True,
            )

    async def use_moa_response(self, __event_emitter__, consensus_model_id: str):
        """
        Use the current consensus_model_id for MOA response synthesis.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
            consensus_model_id (str): The ID of the consensus model to use.
        """
        try:
            if not consensus_model_id or consensus_model_id == CONSENSUS_PLACEHOLDER:
                raise ValueError("[USE_MOA_RESPONSE] Model ID is not properly set.")

            payload = {
                "model": consensus_model_id,  # Use the consensus model ID
                "prompt": "Summarize the outputs below into a single response:",
                "responses": [
                    {"model": model, "content": output}
                    for model, output in self.model_outputs.items()
                ],
                "stream": False,
            }
            self.log_debug(f"[USE_MOA_RESPONSE] Fallback payload: {payload}")

            # Call the generate_moa_response function
            response = await generate_moa_response(form_data=payload, user=mock_user)

            # Handle the response format
            if hasattr(response, "json"):
                response_data = await response.json()
            else:
                response_data = response  # Assume it's already a dict

            self.consensus_output = response_data.get("consensus", "").strip()
            self.log_debug(
                f"[USE_MOA_RESPONSE] Consensus output: {self.consensus_output}"
            )
            await self.emit_output(
                __event_emitter__, self.consensus_output, include_collapsible=True
            )
        except Exception as e:
            self.log_debug(f"[USE_MOA_RESPONSE] Error synthesizing response: {e}")
            await self.emit_status(
                __event_emitter__,
                "Error",
                f"Fallback consensus generation failed: {str(e)}",
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
        elapsed_time = current_time - self.last_emit_time

        if elapsed_time < self.valves.emit_interval and not done:
            # Skip emitting if the interval hasn't passed and it's not a final update
            self.log_debug(
                f"[EMIT_STATUS] Skipping emission. Elapsed time: {elapsed_time:.2f}s, Emit interval: {self.valves.emit_interval}s."
            )
            return

        if __event_emitter__:
            if initial:
                formatted_message = "Seeking Consensus"
            elif done:
                time_suffix = self.format_elapsed_time(elapsed_time)
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
            self.log_debug(
                f"[EMIT_STATUS] Attempting to emit status: {formatted_message}"
            )

            try:
                await __event_emitter__(event)
                self.log_debug("[EMIT_STATUS] Status emitted successfully.")
                self.last_emit_time = current_time  # Update the last emission time
            except Exception as e:
                self.log_debug(f"[EMIT_STATUS] Error emitting status: {e}")

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

    async def emit_collapsible(self, __event_emitter__) -> None:
        """
        Emit a collapsible section containing outputs from all contributing models.

        Args:
            __event_emitter__ (Callable[[dict], Awaitable[None]]): The event emitter for sending messages.
        """
        if not self.model_outputs:
            self.log_debug("[EMIT_COLLAPSIBLE] No model outputs to emit.")
            return

        # Generate collapsible sections for each model
        collapsible_content = "\n".join(
            f"""
<details>
    <summary>Model {model_id} Output</summary>
    <pre>{self.escape_html(output.strip())}</pre>
</details>
""".strip()
            for model_id, output in self.model_outputs.items()
        )

        self.log_debug("[EMIT_COLLAPSIBLE] Collapsible content generated.")

        # Emit the collapsible message
        message_event = {
            "type": "message",
            "data": {"content": collapsible_content},
        }
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
            # Clean the main content
            content_cleaned = content.strip("\n").strip()
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
        # This ensures that existing collapsible tags are not nested or interfere with new ones
        details_pattern = r"<details>\s*<summary>.*?</summary>\s*.*?</details>"
        return re.sub(details_pattern, "", content, flags=re.DOTALL).strip()
