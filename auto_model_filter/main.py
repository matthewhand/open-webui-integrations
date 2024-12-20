"""
title: Auto Model Filter, for routing to the best model
author: matthewh
version: 0.2.30
license: MIT
required_open_webui_version: 0.4.4

Description:
    This module defines the Filter class, which routes user queries to the most suitable model based on predefined criteria.
    It includes robust payload validation, streamlined debug logging, comprehensive error handling, and valves to control
    whether model selection is automated or triggered by <prefix><tag> keywords or trigger keywords in user messages.
    Additionally, it controls the emission of status events and inclusion of routing LLM responses in the output as
    collapsible UI elements.
"""
import json
import re
import asyncio
import logging
import time
import html  # For unescaping HTML entities
from typing import List, Dict, Any, Optional, Callable, Awaitable, Tuple
from pydantic import BaseModel, Field, ValidationError
from fastapi import HTTPException  # For proper error handling in FastAPI

# Assuming get_models and generate_chat_completions are available from open_webui.main
from open_webui.main import get_models, generate_chat_completions  # Update the import path as necessary

# Constants
TAGS_PLACEHOLDER = "comma-separated-list-of-model-tags-goes-here"

# Define User, Message, and ChatCompletionPayload models
class User(BaseModel):
    """
    Represents a user with an ID and role.
    """
    id: str
    role: str


class Message(BaseModel):
    """
    Represents a single message in the chat conversation.
    """
    role: str
    content: str


class ChatCompletionPayload(BaseModel):
    """
    Represents the payload structure for chat completions.
    """
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


class Filter:
    """
    The Filter class routes user queries to the most suitable model based on predefined criteria.
    It supports both automated model selection and triggered selection via <prefix><tag> keywords or trigger keywords.
    Additionally, it allows control over the emission of status events and inclusion of routing LLM responses in the output as
    collapsible UI elements.
    """

    class Valves(BaseModel):
        """
        Configuration valves for the Filter class, grouped by relevance.
        Each group starts with an enable/disable flag.
        """
        # Model Filtering
        filter_models_enabled: bool = Field(
            default=False,
            description="If true, filters models to only include those matching the filter_models_whitelist patterns.",
        )
        filter_models_whitelist: str = Field(
            default="your-comma-separated-model-id-whitelist",
            description="Comma-separated list of acceptable model regex patterns.",
        )

        # Model Type Filtering
        filter_model_types_enabled: bool = Field(
            default=True,
            description="If true, filters models to only include those matching the filter_model_types_whitelist types.",
        )
        filter_model_types_whitelist: str = Field(
            default="CUSTOM",
            description="Comma-separated list of acceptable model types (e.g., 'CUSTOM,OPENAI').",
        )

        # Filter Models with Non-Empty System Prompt
        filter_non_empty_system_prompt_enabled: bool = Field(
            default=True,
            description="If true, filters out models that have an empty 'system_prompt'.",
        )

        # Routing Model Configuration
        routing_model_override: bool = Field(
            default=False,
            description="If true, overrides the default routing model with the value specified in routing_model_id.",
        )
        routing_model_id: str = Field(
            default="model_id_1",  # Set a valid default or ensure it's provided
            description="ID of the model to use for routing if override is enabled.",
        )
        routing_model_system_prompt: str = Field(
            default="Based on the query provided by the user, select the most suitable model to address their needs. Specify the model ID, description, and provide a reason for choosing that model in 25 words or less.",
            description="System prompt for routing the query to the best model, requiring model ID, description, and reasoning in 25 words or less.",
        )

        # Trigger Keywords
        trigger_keywords_enabled: bool = Field(
            default=True,
            description="If true, automatically trigger model consultation based on trigger keywords.",
        )
        trigger_keywords: str = Field(
            default="consult,amf",
            description="Comma-separated list of trigger keywords to initiate model consultation. Do not include the prefix.",
        )

        # Tag-Based Routing
        tag_based_routing_enabled: bool = Field(
            default=True,
            description="Enable tag-based model routing via <prefix><tag> in user messages.",
        )

        # Trigger Prefix
        trigger_prefix: str = Field(
            default="!",
            description="Prefix used to trigger keywords/tags. Defaults to '!'.",
        )

        # Consultation Mode
        consult_every_request: bool = Field(
            default=False,
            description="If true, consults the routing model on every request. If false, consults only based on trigger keywords or tags.",
        )

        # Logging and Status
        debug_valve: bool = Field(
            default=True,
            description="Enable standard debug logging.",
        )
        show_tech_support: bool = Field(
            default=False,
            description="Enable technical support debug logging for detailed internal state information.",
        )
        status_emits_enabled: bool = Field(
            default=True,
            description="If true, emits status events to inform the user about the model selection process.",
        )
        show_routing_response_enabled: bool = Field(
            default=True,
            description="If true, includes the routing LLM's response in the output payload as a collapsible UI element.",
        )

    def __init__(self):
        """
        Initializes the Filter class, setting up logging and valves.
        """
        self.valves = self.Valves()
        self.log = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)
        self.log.setLevel(logging.DEBUG if self.valves.debug_valve else logging.INFO)

        # Initial essential debug logs
        self.log_debug(
            "[Filter INIT] Auto Model Filter initialized with the following valves:"
        )
        self.log_debug(f"{self.valves}")

        # Verify that routing_model_id is a string
        if not isinstance(self.valves.routing_model_id, str):
            self.log_error("routing_model_id must be a string.")
            raise ValueError("routing_model_id must be a string.")
        else:
            self.log_debug(f"[INIT] routing_model_id is a string: {self.valves.routing_model_id}")

        # Event emitter placeholder
        self.__event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None

    def log_debug(self, message: str):
        """
        Logs debug messages if the debug_valve is enabled.
        Only essential debug messages are logged under normal debug.
        """
        if self.valves.debug_valve:
            self.log.debug(message)

    def log_tech_support(self, message: str):
        """
        Logs technical support debug messages if the show_tech_support valve is enabled.
        These are detailed and less critical logs.
        """
        if self.valves.show_tech_support:
            self.log.debug(f"[TECH SUPPORT] {message}")

    def log_error(self, message: str):
        """
        Logs error messages.
        """
        self.log.error(message)

    def set_event_emitter(self, emitter: Callable[[dict], Awaitable[None]]):
        """
        Sets the event emitter function.
        """
        self.__event_emitter__ = emitter
        self.log_debug("[set_event_emitter] Event emitter set.")

    def validate_payload(self, body: dict) -> None:
        """
        Validates that the incoming payload conforms to the expected message structure.
        """
        try:
            self.log_debug("[validate_payload] Starting payload validation.")
            self.log_debug(f"[validate_payload] Payload: {json.dumps(body, indent=2)}")
            if not isinstance(body, dict):
                self.log_debug("[validate_payload] Payload is not a dictionary.")
                raise ValueError("Payload must be a dictionary.")

            # Validate 'model' field
            if "model" not in body:
                self.log_debug("[validate_payload] 'model' key missing from payload.")
                raise ValueError("Payload must contain 'model' key.")
            if not isinstance(body["model"], str) or not body["model"].strip():
                self.log_debug("[validate_payload] 'model' is not a non-empty string.")
                raise ValueError("'model' must be a non-empty string.")

            # Validate 'messages' field
            if "messages" not in body:
                self.log_debug("[validate_payload] 'messages' key missing from payload.")
                raise ValueError("Payload must contain 'messages' key.")

            messages = body["messages"]
            if not isinstance(messages, list):
                self.log_debug("[validate_payload] 'messages' is not a list.")
                raise ValueError("'messages' must be a list.")

            for idx, message in enumerate(messages):
                if not isinstance(message, dict):
                    self.log_debug(f"[validate_payload] Message at index {idx} is not a dictionary.")
                    raise ValueError(f"Each message must be a dictionary. Error at index {idx}.")
                if "role" not in message:
                    self.log_debug(f"[validate_payload] 'role' missing in message at index {idx}.")
                    raise ValueError(f"Each message must contain a 'role' key. Error at index {idx}.")
                if "content" not in message:
                    self.log_debug(f"[validate_payload] 'content' missing in message at index {idx}.")
                    raise ValueError(f"Each message must contain a 'content' key. Error at index {idx}.")
                if not isinstance(message["role"], str):
                    self.log_debug(f"[validate_payload] 'role' is not a string in message at index {idx}.")
                    raise ValueError(f"'role' must be a string in message at index {idx}.")
                if not isinstance(message["content"], str):
                    self.log_debug(f"[validate_payload] 'content' is not a string in message at index {idx}.")
                    raise ValueError(f"'content' must be a string in message at index {idx}.")

            self.log_debug("[validate_payload] Payload validation passed.")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_status("Payload validation passed.", done=False))
        except ValueError as ve:
            self.log_debug(f"[validate_payload] ValueError: {ve}")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_error(f"Payload validation error: {ve}"))
            raise
        except Exception as e:
            self.log_debug(f"[validate_payload] Unexpected exception: {e}")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_error(f"Unexpected payload validation error: {e}"))
            raise
        finally:
            self.log_debug("[validate_payload] Completed payload validation.")

    def parse_tags_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts unique tags from user messages prefixed with <prefix><tag>.
        """
        tags = set()  # Use a set to ensure uniqueness
        prefix = self.valves.trigger_prefix
        pattern = re.compile(rf'^{re.escape(prefix)}(\w+)', re.IGNORECASE)
        for message in messages:
            if message.get("role") != "user":
                continue  # Only process user messages
            content = message.get("content", "")
            match = pattern.match(content)
            if match:
                tag = match.group(1).lower()
                tags.add(tag)
                # Remove the trigger prefix and tag from the message content
                message["content"] = pattern.sub('', content, count=1).strip()
                self.log_tech_support(f"Detected tag '{prefix}{tag}' and updated message content to '{message['content']}'")
        tags_list = sorted(tags)  # Sort for consistent ordering
        if tags_list:
            self.log_debug(f"[parse_tags_from_messages] Extracted unique tags: {', '.join(tags_list)}")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_status(f"Extracted tags: {', '.join(tags_list)}", done=False))
        else:
            self.log_debug("[parse_tags_from_messages] No tags extracted from messages.")
        return tags_list

    def check_trigger_keywords(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Checks if any of the trigger keywords are present in user messages.
        """
        trigger_present = False
        # Normalize trigger keywords
        trigger_keywords = [
            keyword.strip().lower()
            for keyword in self.valves.trigger_keywords.split(",")
            if keyword.strip()
        ]
        if not trigger_keywords:
            self.log_debug("[check_trigger_keywords] No trigger keywords specified.")
            return False

        # Compile a regex pattern that matches any of the trigger keywords as whole words
        trigger_regex = re.compile(r'\b(' + '|'.join(map(re.escape, trigger_keywords)) + r')\b', re.IGNORECASE)
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if trigger_regex.search(content):
                trigger_present = True
                self.log_tech_support(f"Trigger keyword detected in message: '{content}'")
                # Strip the trigger keyword(s) from the message content
                message["content"] = trigger_regex.sub('', content).strip()
                self.log_tech_support(f"Stripped trigger keyword(s) from message content: '{message['content']}'")
                break  # No need to check further once a trigger is found
        return trigger_present

    def filter_models_by_type(self, available_models: List[Dict[str, Any]], default_type: str = "CUSTOM") -> List[Dict[str, Any]]:
        """
        Filters models based on their 'type' field if filtering is enabled.
        Also logs the count before and after filtering.
        """
        if not self.valves.filter_model_types_enabled:
            self.log_debug("[filter_models_by_type] Model type filtering is disabled.")
            return available_models

        try:
            # Split and clean the type whitelist
            acceptable_types = [
                type_str.strip().upper()
                for type_str in self.valves.filter_model_types_whitelist.split(",")
                if type_str.strip()
            ]
            if not acceptable_types:
                self.log_debug("[filter_models_by_type] No acceptable model types specified. Skipping type filtering.")
                return available_models

            self.log_debug(
                f"[filter_models_by_type] Applying type filtering. Number of models before filtering: {len(available_models)}"
            )
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_status("Applying model type filtering...", done=False))

            # Filter models matching any of the acceptable types
            filtered_models = [
                model
                for model in available_models
                if model.get("type", "").upper() in acceptable_types
            ]

            self.log_debug(
                f"[filter_models_by_type] Number of models after type filtering: {len(filtered_models)}"
            )
            if self.valves.status_emits_enabled:
                filtered_count = len(filtered_models)
                asyncio.create_task(self.emit_status(
                    f"Filtered models based on types. Before: {len(available_models)}, After: {filtered_count}",
                    done=False
                ))
            return filtered_models
        except Exception as e:
            self.log_tech_support(f"[filter_models_by_type] Unexpected exception: {e}")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_error(f"Unexpected error during model type filtering: {e}"))
            raise
        finally:
            self.log_tech_support("[filter_models_by_type] Completed model type filtering.")

    def filter_models_by_non_empty_system_prompt(self, available_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters models to exclude those with an empty 'system_prompt' if the valve is enabled.
        Additionally, logs all models with their 'system_prompt' before filtering for debugging purposes.
        """
        if not self.valves.filter_non_empty_system_prompt_enabled:
            self.log_debug("[filter_models_by_non_empty_system_prompt] Filtering by non-empty system_prompt is disabled.")
            return available_models

        try:
            # Log all models with their 'system_prompt' before filtering
            self.log_tech_support("[filter_models_by_non_empty_system_prompt] Listing all models with non-empty 'system_prompt':")
            for model in available_models:
                system_prompt = model.get("system_prompt", "").strip()
                if system_prompt:
                    self.log_tech_support(f"Model ID: {model['id']}, System Prompt: {system_prompt}")
                else:
                    self.log_tech_support(f"Model ID: {model['id']}, System Prompt: <EMPTY>")

            self.log_debug(
                f"[filter_models_by_non_empty_system_prompt] Applying system_prompt filtering. Number of models before filtering: {len(available_models)}"
            )
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_status("Filtering models with non-empty system prompts...", done=False))

            filtered_models = [
                model for model in available_models
                if model.get("system_prompt", "").strip()
            ]

            self.log_debug(
                f"[filter_models_by_non_empty_system_prompt] Number of models after system_prompt filtering: {len(filtered_models)}"
            )
            if self.valves.status_emits_enabled:
                filtered_count = len(filtered_models)
                asyncio.create_task(self.emit_status(
                    f"Filtered models with non-empty system prompts. Before: {len(available_models)}, After: {filtered_count}",
                    done=False
                ))
            return filtered_models
        except Exception as e:
            self.log_tech_support(f"[filter_models_by_non_empty_system_prompt] Unexpected exception: {e}")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_error(f"Unexpected error during system_prompt filtering: {e}"))
            raise
        finally:
            self.log_tech_support("[filter_models_by_non_empty_system_prompt] Completed system_prompt filtering.")

    def filter_models_by_tags(self, available_models: List[Dict[str, Any]], required_tags: List[str]) -> List[Dict[str, Any]]:
        """
        Filters models based on the provided tags.
        Also logs the count before and after filtering.
        """
        if not required_tags:
            self.log_debug("[filter_models_by_tags] No tags provided for filtering.")
            return available_models

        self.log_debug(f"[filter_models_by_tags] Applying tag-based filtering. Number of models before filtering: {len(available_models)}")
        if self.valves.status_emits_enabled:
            asyncio.create_task(self.emit_status(f"Filtering models with tags: {', '.join(required_tags)}", done=False))

        filtered_models = []
        for model in available_models:
            try:
                # Extract model tags directly from 'tags' field
                model_tags = set(tag.lower() for tag in model.get("tags", []) if isinstance(tag, str))
                self.log_debug(f"[filter_models_by_tags] Model '{model['id']}' tags: {model_tags}")

                # Check for tag intersection
                if set(required_tags).intersection(model_tags):
                    self.log_debug(f"[filter_models_by_tags] Model '{model['id']}' matches tags.")
                    filtered_models.append(model)
                else:
                    # Move less critical debug logs under show_tech_support
                    self.log_tech_support(f"Model '{model['id']}' does not match tags.")
            except Exception as e:
                self.log_tech_support(f"Error processing model '{model.get('id', 'unknown-id')}': {e}")
                if self.valves.status_emits_enabled:
                    asyncio.create_task(self.emit_error(f"Error processing model '{model.get('id', 'unknown-id')}': {e}"))
                continue  # Skip models with unexpected errors

        self.log_debug(f"[filter_models_by_tags] Number of models after tag filtering: {len(filtered_models)}")
        if self.valves.status_emits_enabled:
            filtered_count = len(filtered_models)
            asyncio.create_task(self.emit_status(
                f"Filtered models based on tags. Before: {len(available_models)}, After: {filtered_count}",
                done=False
            ))
        return filtered_models

    def filter_models_by_whitelist(self, available_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters models based on acceptable regex patterns if filtering is enabled.
        Also logs the count before and after filtering.
        """
        if not self.valves.filter_models_enabled:
            self.log_debug("[filter_models_by_whitelist] Filtering is disabled.")
            return available_models

        try:
            # Compile regex patterns from the whitelist
            acceptable_patterns = [
                re.compile(pattern.strip(), re.IGNORECASE)
                for pattern in self.valves.filter_models_whitelist.split(",")
                if pattern.strip()
            ]
            if not acceptable_patterns:
                self.log_debug("[filter_models_by_whitelist] No valid regex patterns found in whitelist. Skipping regex filtering.")
                return available_models

            self.log_debug(
                f"[filter_models_by_whitelist] Applying whitelist regex filtering. Number of models before filtering: {len(available_models)}"
            )
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_status("Applying whitelist regex filtering to models...", done=False))

            # Filter models matching any of the acceptable patterns
            filtered_models = [
                model for model in available_models
                if any(pattern.search(model["id"]) for pattern in acceptable_patterns)
            ]
            self.log_debug(
                f"[filter_models_by_whitelist] Number of models after whitelist filtering: {len(filtered_models)}"
            )
            if self.valves.status_emits_enabled:
                filtered_count = len(filtered_models)
                asyncio.create_task(self.emit_status(
                    f"Filtered models based on whitelist regex. Before: {len(available_models)}, After: {filtered_count}",
                    done=False
                ))
            return filtered_models
        except re.error as e:
            self.log_tech_support(f"[filter_models_by_whitelist] Regex error: {e}")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_error(f"Regex error in filter_models_whitelist: {str(e)}"))
            raise ValueError(f"Invalid regex pattern in filter_models_whitelist: {e}")
        except Exception as e:
            self.log_tech_support(f"[filter_models_by_whitelist] Unexpected exception: {e}")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_error(f"Unexpected error during model whitelist filtering: {e}"))
            raise
        finally:
            self.log_tech_support("[filter_models_by_whitelist] Completed whitelist filtering.")

    async def get_all_models(self) -> List[Dict[str, Any]]:
        """
        Fetches all available models using the get_models function.
        """
        try:
            self.log_debug("[get_all_models] Fetching all models.")
            if self.valves.status_emits_enabled:
                await self.emit_status("Fetching all available models...", done=False)
            response = await get_models(user=User(id="system", role="admin"))

            self.log_debug(f"[get_all_models] Raw response type: {type(response).__name__}")
            self.log_debug(f"[get_all_models] Raw response: {json.dumps(response, indent=2)}")

            # Check if response is a list
            if isinstance(response, list):
                models_list = response
                self.log_debug("[get_all_models] Response is a list of models.")
            elif isinstance(response, dict):
                # Attempt to extract models list from known keys
                possible_keys = ['models', 'data', 'results']
                models_list = None
                for key in possible_keys:
                    if key in response and isinstance(response[key], list):
                        models_list = response[key]
                        self.log_debug(f"[get_all_models] Extracted models list from key '{key}'.")
                        break
                if models_list is None:
                    # If no known key is found, assume the entire dict is invalid
                    self.log_error(f"[get_all_models] Expected a list of models, but received a dict without a recognizable models key.")
                    if self.valves.status_emits_enabled:
                        await self.emit_error("Unexpected response structure: Received a dictionary without a recognizable models key.")
                    raise ValueError("Unexpected response structure: Received a dictionary without a recognizable models key.")
            else:
                self.log_error(f"[get_all_models] Expected a list of models or a dict containing models, got {type(response).__name__}.")
                if self.valves.status_emits_enabled:
                    await self.emit_error(f"Unexpected response type: {type(response).__name__}. Expected a list of models.")
                raise ValueError(f"Unexpected response type: {type(response).__name__}. Expected a list of models.")

            # Process each model to extract relevant details
            processed_models = []
            unique_tags = set()  # For sanity check
            for idx, model in enumerate(models_list):
                try:
                    if not isinstance(model, dict):
                        self.log_tech_support(f"[get_all_models] Model at index {idx} is not a dictionary. Skipping.")
                        if self.valves.status_emits_enabled:
                            await self.emit_error(f"Model at index {idx} is not a dictionary. Skipping.")
                        continue  # Skip non-dictionary entries

                    details_str = model.get("details", "{}")
                    if not isinstance(details_str, str):
                        self.log_tech_support(f"[get_all_models] 'details' field in model '{model.get('id', 'unknown-id')}' is not a string.")
                        if self.valves.status_emits_enabled:
                            await self.emit_error(f"'details' field in model '{model.get('id', 'unknown-id')}' is not a string.")
                        continue  # Skip models with invalid 'details' field

                    try:
                        # Normalize keys to lowercase for consistency
                        details_raw = json.loads(details_str)
                        if not isinstance(details_raw, dict):
                            self.log_tech_support(f"[get_all_models] 'details' field in model '{model.get('id', 'unknown-id')}' is not a dictionary after parsing.")
                            if self.valves.status_emits_enabled:
                                await self.emit_error(f"'details' field in model '{model.get('id', 'unknown-id')}' is not a dictionary after parsing.")
                            continue  # Skip if details are not a dict
                        details = {k.lower(): v for k, v in details_raw.items()}
                    except json.JSONDecodeError as jde:
                        self.log_tech_support(f"[get_all_models] JSONDecodeError for 'details' in model '{model.get('id', 'unknown-id')}': {jde}")
                        if self.valves.status_emits_enabled:
                            await self.emit_error(f"Error parsing 'details' for model '{model.get('id', 'unknown-id')}': {jde}")
                        continue  # Skip models with invalid JSON in 'details'

                    description = details.get("description", "No description")
                    model_id = model.get("id", "unknown-id")

                    # Extract tags if available
                    tags = details.get("tags", [])
                    if not isinstance(tags, list):
                        self.log_tech_support(f"[get_all_models] 'tags' in model '{model_id}' is not a list. Attempting to parse as comma-separated string.")
                        if isinstance(tags, str):
                            # Attempt to split comma-separated tags
                            tags = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
                            self.log_debug(f"[get_all_models] Parsed 'tags' for model '{model_id}': {tags}")
                        else:
                            self.log_tech_support(f"[get_all_models] 'tags' in model '{model_id}' is not a list or string. Setting to empty list.")
                            tags = []

                    # Add tags to unique_tags set for sanity check
                    unique_tags.update(tags)

                    # Extract system_prompt if available
                    # system_prompt = details.get("system_prompt", "")
                    # if not system_prompt:
                    info = model.get("info", {})
                    params = info.get("params", {})
                    system_prompt = params.get("system", "")  # Fallback to nested field

                    # Sanity check: Log the system_prompt value for each model
                    if self.valves.show_tech_support:
                        self.log_tech_support(f"[filter_models_by_non_empty_system_prompt] Model '{model.get('id', 'unknown')}' system_prompt: {repr(system_prompt)}")

                    # Extract type if available
                    model_type = details.get("type", "CUSTOM")
                    if not isinstance(model_type, str):
                        self.log_tech_support(f"[get_all_models] 'type' in model '{model_id}' is not a string. Setting to 'CUSTOM'.")
                        model_type = "CUSTOM"

                    processed_model = {
                        "id": model_id,
                        "description": description,
                        "tags": tags,  # Now ensured to be a list of strings
                        "system_prompt": system_prompt,  # Adjust if needed
                        "type": model_type.upper(),  # Ensure type is uppercase for consistency
                        # Add other necessary fields as required
                    }
                    self.log_debug(f"[get_all_models] Processed model '{model_id}': {json.dumps(processed_model, indent=2)}")
                    processed_models.append(processed_model)
                except Exception as e:
                    self.log_tech_support(f"[get_all_models] Unexpected error processing model at index {idx}: {e}")
                    if self.valves.status_emits_enabled:
                        await self.emit_error(f"Unexpected error processing model at index {idx}: {e}")
                    continue  # Skip models with unexpected errors

            # Sanity Check: Compile and log unique tags
            self.log_debug(f"[get_all_models] Unique tags across all models: {sorted(unique_tags)}")

            self.log_debug(f"[get_all_models] Total processed models: {len(processed_models)}")
            if self.valves.status_emits_enabled:
                await self.emit_status(f"Retrieved {len(processed_models)} models from the system.", done=False)
            return processed_models
        except ValidationError as ve:
            self.log_tech_support(f"[get_all_models] ValidationError occurred: {ve}")
            if self.valves.status_emits_enabled:
                await self.emit_error(f"Validation error fetching models: {ve}")
            raise
        except Exception as e:
            self.log_tech_support(f"[get_all_models] Exception occurred: {e}")
            if self.valves.status_emits_enabled:
                await self.emit_error(f"Error fetching models: {str(e)}")
            raise
        finally:
            self.log_debug("[get_all_models] Completed fetching models.")

    async def determine_best_model(
        self,
        user_query: str,
        models: List[Dict[str, Any]],
        __model__: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Determines the best model for the user query using generate_chat_completions.
        Emits status events during the process.
        """
        try:
            # Prepare model descriptions with tags for the routing model
            model_descriptions = "\n".join(
                [
                    f"Model ID: {model['id']}, Description: {model.get('description', 'No description')}, Tags: {', '.join(model.get('tags', []))}"
                    for model in models
                ]
            )
            self.log_debug(
                f"[determine_best_model] Model descriptions prepared."
            )
            if self.valves.status_emits_enabled:
                await self.emit_status("Preparing to determine the best model for your query...", done=False)

            # Construct messages for the routing model
            system_message = {
                "role": "system",
                "content": self.valves.routing_model_system_prompt,
            }
            user_message = {
                "role": "user",
                "content": f"User Query: {user_query}\nAvailable Models:\n{model_descriptions}",
            }

            # Determine which routing model to use
            routing_model = (
                self.valves.routing_model_id
                if self.valves.routing_model_override
                else (__model__ if __model__ else None)
            )

            if routing_model is None:
                self.log_error("[determine_best_model] No routing model specified.")
                if self.valves.status_emits_enabled:
                    await self.emit_error("No routing model specified for determining the best model.")
                raise ValueError("No routing model specified for determining the best model.")

            self.log_debug(f"[determine_best_model] Using routing model: {routing_model}")
            self.log_debug(f"[determine_best_model] Type of routing_model: {type(routing_model)}")

            # Ensure routing_model is a string
            if not isinstance(routing_model, str):
                self.log_debug(f"[determine_best_model] routing_model is not a string: {routing_model}")
                raise ValueError("routing_model must be a string representing the model ID.")

            if self.valves.status_emits_enabled:
                await self.emit_status(f"Using routing model: {routing_model}", done=False)

            # Prepare the payload for the routing model
            payload = {
                "messages": [system_message, user_message],
                "model": routing_model,
                "stream": False  # Set to False for routing
            }

            # Log the payload being sent
            self.log_debug(f"[determine_best_model] Payload being sent: {json.dumps(payload, indent=2)}")
            if self.valves.status_emits_enabled:
                await self.emit_status("Sending payload to routing model for best model determination...", done=False)

            # Validate payload structure before sending
            self.validate_payload(payload)

            # Emit status indicating model determination has started
            if self.valves.status_emits_enabled:
                await self.emit_status("Determining the best model for your query...", done=False)

            # Call the generate_chat_completions function to determine the best model
            response = await generate_chat_completions(
                form_data=payload, user=User(id="system", role="admin")
            )

            # Log the response received
            self.log_debug(f"[determine_best_model] Response received: {json.dumps(response, indent=2)}")
            if self.valves.status_emits_enabled:
                await self.emit_status("Received response from routing model.", done=False)

            routing_response_content = ""
            best_model = ""
            if isinstance(response, dict) and "choices" in response:
                # Ensure that 'choices' is a list with at least one element
                if isinstance(response["choices"], list) and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        response_text = choice["message"]["content"].strip()
                        self.log_debug(f"[determine_best_model] Routing model response: {response_text}")
                        routing_response_content = response_text
                        if self.valves.status_emits_enabled:
                            await self.emit_status(f"Routing model response: {response_text}", done=False)

                        # Initialize variables to track the best match
                        longest_match_length = -1

                        # Iterate through all model IDs to find all matches
                        for model in models:
                            model_id = model["id"]
                            if re.search(rf'\b{re.escape(model_id)}\b', response_text, re.IGNORECASE):
                                match_length = len(model_id)
                                self.log_debug(f"[determine_best_model] Detected model ID '{model_id}' with length {match_length}.")
                                if match_length > longest_match_length:
                                    best_model = model_id
                                    longest_match_length = match_length
                                    self.log_debug(f"[determine_best_model] '{model_id}' is currently the best match.")
                                else:
                                    self.log_tech_support(f"[determine_best_model] '{model_id}' is not a better match than '{best_model}'.")

                        # Handle cases where no model ID is matched
                        if not best_model:
                            self.log_debug("[determine_best_model] No matching model ID found in response.")
                            if self.valves.status_emits_enabled:
                                await self.emit_error("Failed to match any model ID from the routing model response.")
                            raise ValueError(
                                "Failed to match any model ID from the routing model response."
                            )

                        # Retrieve the description of the best model
                        best_model_description = next(
                            (model["description"] for model in models if model["id"] == best_model),
                            "No description available."
                        )
                        routing_response_content = f"Selected model '{best_model}' - {best_model_description}."

                        self.log_debug(f"[determine_best_model] Best model selected: '{best_model}' with description: '{best_model_description}'")
                        if self.valves.status_emits_enabled:
                            await self.emit_status(f"Best model selected: '{best_model}' - {best_model_description}", done=False)

                        return best_model, routing_response_content
                    else:
                        self.log_debug("[determine_best_model] 'message' or 'content' missing in choice.")
                else:
                    self.log_debug("[determine_best_model] 'choices' is empty or not a list.")
            self.log_debug("[determine_best_model] Invalid response structure.")
            if self.valves.status_emits_enabled:
                await self.emit_error("Failed to determine the best model due to invalid response structure.")
            raise ValueError("Failed to determine the best model.")
        except ValueError as ve:
            self.log_debug(f"[determine_best_model] ValueError: {ve}")
            if self.valves.status_emits_enabled:
                await self.emit_error(f"Model determination error: {ve}")
            raise
        except Exception as e:
            self.log_debug(f"[determine_best_model] Unexpected exception: {e}")
            if self.valves.status_emits_enabled:
                await self.emit_error(f"Unexpected error during model determination: {e}")
            raise
        finally:
            self.log_tech_support("[determine_best_model] Completed best model determination.")

    async def determine_best_model_direct(
        self,
        model_id: str,
        models: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        Directly selects the specified model without using the routing model.
        """
        try:
            self.log_debug(f"[determine_best_model_direct] Attempting to select model '{model_id}'.")
            # Verify that the model exists in the list
            model_exists = any(model["id"].lower() == model_id.lower() for model in models)
            if not model_exists:
                self.log_debug(f"[determine_best_model_direct] Model ID '{model_id}' not found in available models.")
                if self.valves.status_emits_enabled:
                    await self.emit_error(f"Model ID '{model_id}' not found.")
                raise ValueError(f"Model ID '{model_id}' not found.")

            self.log_debug(f"[determine_best_model_direct] Model '{model_id}' found.")
            if self.valves.status_emits_enabled:
                await self.emit_status(f"Selected model: '{model_id}'.", done=False)

            # Find the model dictionary
            selected_model = next(model for model in models if model["id"].lower() == model_id.lower())
            model_description = selected_model.get("description", "No description available.")
            routing_response_content = f"Directly selected model '{model_id}' - {model_description}."

            return selected_model["id"], routing_response_content
        except ValueError as ve:
            self.log_debug(f"[determine_best_model_direct] ValueError: {ve}")
            if self.valves.status_emits_enabled:
                await self.emit_error(f"Model selection error: {ve}")
            raise
        except Exception as e:
            self.log_debug(f"[determine_best_model_direct] Unexpected exception: {e}")
            if self.valves.status_emits_enabled:
                await self.emit_error(f"Unexpected error during model selection: {e}")
            raise
        finally:
            self.log_tech_support("[determine_best_model_direct] Completed direct model selection.")

    async def filter_models(
        self,
        available_models: List[Dict[str, Any]],
        required_tags: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Applies the filtering logic in the correct order:
        1. Type Filtering
        2. Non-Empty System Prompt Filtering
        3. Tag Filtering
        4. Whitelist Regex Filtering

        Args:
            available_models (List[Dict[str, Any]]): List of all available models.
            required_tags (List[str]): List of tags to filter models by.

        Returns:
            List[Dict[str, Any]]: Filtered list of models.
        """
        filtered_by_type = self.filter_models_by_type(available_models)
        self.log_debug(f"[filter_models] After type filtering: {len(filtered_by_type)}")
        
        filtered_by_sp = self.filter_models_by_non_empty_system_prompt(filtered_by_type)
        self.log_debug(f"[filter_models] After system_prompt filtering: {len(filtered_by_sp)}")
        
        filtered_by_tags = self.filter_models_by_tags(filtered_by_sp, required_tags)

        self.log_debug(f"[filter_models] After tag filtering: {len(filtered_by_tags)}")
        
        final_filtered = self.filter_models_by_whitelist(filtered_by_tags)
        self.log_debug(f"[filter_models] After whitelist filtering: {len(final_filtered)}")
        
        return final_filtered

    async def emit_output(
        self, 
        content: str, 
        include_collapsible: bool = False, 
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Emit content and optionally include a collapsible section with model outputs.

        Args:
            content (str): The main content to emit.
            include_collapsible (bool): Whether to include collapsibles with the output.
            messages (List[Dict[str, Any]], optional): The messages list to append to.
        """
        if content:
            # Prepare the message to append
            message = {"role": "assistant", "content": content}
            if messages is not None:
                messages.append(message)
                self.log_debug(f"[emit_output] Appended message to messages: {content}")

            # Prepare the message event to emit
            main_message_event = {
                "type": "message",
                "data": {"content": content},
            }

            try:
                # Emit the main content
                if self.__event_emitter__:
                    await self.__event_emitter__(main_message_event)
                    self.log_debug("[emit_output] Main output message emitted successfully.")
                else:
                    self.log_debug("[emit_output] Event emitter not set. Skipping emission.")

                # Optionally include collapsible
                if include_collapsible and self.valves.show_routing_response_enabled:
                    # Prepare the collapsible content
                    collapsible_content = f"""<details>
<summary>{content}</summary>
{html.unescape(content)}
</details>"""
                    if messages is not None:
                        messages.append({"role": "assistant", "content": collapsible_content})
                        self.log_debug(f"[emit_output] Appended collapsible message to messages: {collapsible_content}")

                    # Emit the collapsible content
                    collapsible_event = {
                        "type": "message",
                        "data": {"content": collapsible_content},
                    }
                    if self.__event_emitter__:
                        await self.__event_emitter__(collapsible_event)
                        self.log_debug("[emit_output] Collapsible content emitted successfully.")
                    else:
                        self.log_debug("[emit_output] Event emitter not set. Skipping collapsible emission.")
            except Exception as e:
                self.log_tech_support(f"[emit_output] Error emitting output or collapsible: {e}")

    async def inlet(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
    ) -> dict:
        """
        Main processing function that handles incoming requests, validates payloads, routes queries
        to appropriate models based on configuration and tags or trigger keywords, and emits status events.
        """
        start_time = time.time()
        self.log_debug("[inlet] Starting inlet processing.")

        # Assign the event emitter to an instance variable for use in emit_status and emit_error
        self.set_event_emitter(__event_emitter__)

        # Flag to indicate whether model selection was performed
        model_selected = False

        try:
            # Step 1: Validate Payload Structure
            self.validate_payload(body)
            self.log_debug("[inlet] Payload validation passed.")
            # Status emit handled within validate_payload

            # Extract user messages from the payload
            messages = body.get("messages", [])

            # Step 2: Extract tags from user messages if tag-based routing is enabled
            tags = []
            if self.valves.tag_based_routing_enabled:
                tags = self.parse_tags_from_messages(messages)
                if tags:
                    self.log_debug(f"[inlet] Extracted tags from messages: {tags}")
                    # Status emit handled within parse_tags_from_messages

            # Step 3: Extract trigger keywords from user messages if trigger_keywords_enabled
            trigger_present = False
            if self.valves.trigger_keywords_enabled:
                trigger_present = self.check_trigger_keywords(messages)
                self.log_debug(f"[inlet] Trigger keywords present: {trigger_present}")
                if trigger_present:
                    await self.emit_status("Trigger keywords detected in user messages.", done=False)

            # Step 4: Fetch all available models
            all_models = await self.get_all_models()
            self.log_tech_support(f"[inlet] Retrieved all models: {[model['id'] for model in all_models]}")
            # Status emit handled within get_all_models

            # Step 5: Apply filtering logic in the correct order
            filtered_models = await self.filter_models(all_models, tags)
            self.log_debug(f"[inlet] Models after all filtering: {len(filtered_models)}")

            # Step 6: Determine if model consultation should be triggered
            consultation_triggered = False
            consultation_mode = ""
            if self.valves.consult_every_request:
                consultation_triggered = True
                consultation_mode = "Automatic"
                self.log_debug("[inlet] 'consult_every_request' is True. Consultation will be triggered for every request.")
                if self.valves.status_emits_enabled:
                    await self.emit_status("Consultation triggered by 'consult_every_request' valve (Automatic).", done=False)
            else:
                # Consultation is triggered only if trigger keywords or tags are present
                if self.valves.trigger_keywords_enabled and trigger_present:
                    consultation_triggered = True
                    consultation_mode = "Trigger Keyword"
                    self.log_debug("[inlet] Consultation triggered by trigger keywords.")
                    if self.valves.status_emits_enabled:
                        await self.emit_status("Consultation triggered by trigger keywords.", done=False)
                elif self.valves.tag_based_routing_enabled and tags and filtered_models:
                    consultation_triggered = True
                    consultation_mode = "Tag"
                    self.log_debug("[inlet] Consultation triggered by tag-based routing.")
                    if self.valves.status_emits_enabled:
                        await self.emit_status("Consultation triggered by tag-based routing.", done=False)

            if not consultation_triggered:
                self.log_debug("[inlet] Consultation not triggered based on current configuration.")
                if self.valves.status_emits_enabled:
                    await self.emit_status("No consultation triggered based on current configuration.", done=True)
                # Calculate and log the consultation duration
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(int(elapsed_time), 60)
                elapsed_time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                self.log_debug(
                    f"[inlet] Inlet processing completed in {elapsed_time_str}."
                )
                if elapsed_time >= 1:
                    await self.emit_status(
                        f"Using model '{body['model']}' ({elapsed_time_str})",
                        done=True
                    )
                else:
                    await self.emit_status(
                        f"Using model '{body['model']}'",
                        done=True
                    )
                return body  # Return the original payload unmodified

            self.log_debug(f"[inlet] Consultation triggered by {consultation_mode}.")
            if self.valves.status_emits_enabled:
                await self.emit_status(f"Consultation triggered by {consultation_mode}.", done=False)

            # Step 7: Extract user query from the messages
            user_message = next(
                (
                    message
                    for message in messages
                    if message.get("role") == "user"
                ),
                None,
            )
            if not user_message:
                self.log_debug("[inlet] No user message found in the request body.")
                if self.valves.status_emits_enabled:
                    await self.emit_error("No user message found in the request body.")
                raise HTTPException(status_code=400, detail="No user message found in the request body.")

            user_query = user_message.get("content", "")
            self.log_debug(f"[inlet] User query: '{user_query}'")
            if self.valves.status_emits_enabled:
                await self.emit_status(f"User query received: '{user_query}'", done=False)

            # Step 8: Determine the best model for the user query
            selected_model_id = body.get("model", "")
            routing_response_content = ""
            if consultation_mode == "Tag" and len(filtered_models) == 1:
                # Directly select the model corresponding to the tag
                selected_model_id = filtered_models[0]["id"]
                self.log_debug(f"[inlet] Directly selecting model '{selected_model_id}' based on tag.")
                if self.valves.status_emits_enabled:
                    await self.emit_status(f"Directly selecting model '{selected_model_id}' based on tag.", done=False)
                selected_model_prompt = filtered_models[0].get("system_prompt", "")
                body["model"] = selected_model_id
                self.log_debug(f"[inlet] Updated payload with selected model: '{selected_model_id}'")
                if self.valves.status_emits_enabled:
                    await self.emit_status(f"Payload updated with selected model: '{selected_model_id}'", done=False)

                # Update the system role content with the new system prompt
                self.update_system_prompt(body, selected_model_prompt)
                self.log_debug(f"[inlet] Updated request body with new system prompt.")
                if self.valves.status_emits_enabled:
                    await self.emit_status("System prompt updated with the selected model's prompt.", done=False)

                # Include routing LLM's response as a collapsible UI element if the valve is enabled
                if self.valves.show_routing_response_enabled:
                    routing_response_content = f"Directly selected model '{selected_model_id}' - {filtered_models[0].get('description', 'No description available.')}"
                    await self.emit_output(
                        content=routing_response_content,
                        include_collapsible=True,
                        messages=messages
                    )
                    self.log_debug(f"[inlet] Included routing LLM's response as a collapsible element in the payload.")
                    if self.valves.status_emits_enabled:
                        await self.emit_status("Included routing LLM's response as a collapsible element in the payload.", done=False)

                # Set the flag indicating model selection was performed
                model_selected = True

            else:
                # Use the routing model to determine the best model
                best_model, routing_response_content = await self.determine_best_model(
                    user_query, filtered_models, __model__
                )
                self.log_debug(f"[inlet] Best model determined: '{best_model}'")
                if self.valves.status_emits_enabled:
                    await self.emit_status(f"Best model determined: '{best_model}'", done=False)

                # Update the payload with the selected best model
                body["model"] = best_model
                self.log_debug(f"[inlet] Updated payload with selected model: '{best_model}'")
                if self.valves.status_emits_enabled:
                    await self.emit_status(f"Payload updated with selected model: '{best_model}'", done=False)

                # Find the system prompt for the selected model
                best_model_prompt = next(
                    (
                        model.get("system_prompt", "")
                        for model in filtered_models
                        if model["id"] == best_model
                    ),
                    ""
                )
                if not best_model_prompt:
                    self.log_debug(
                        f"[inlet] No system prompt found for the selected model: '{best_model}'"
                    )
                    if self.valves.status_emits_enabled:
                        await self.emit_error(
                            f"No system prompt found for the selected model: '{best_model}'."
                        )
                    raise HTTPException(
                        status_code=400, detail=f"No system prompt found for the selected model: '{best_model}'."
                    )

                # Update the system role content with the new system prompt
                self.update_system_prompt(body, best_model_prompt)
                self.log_debug(f"[inlet] Updated request body with new system prompt.")
                if self.valves.status_emits_enabled:
                    await self.emit_status("System prompt updated with the selected model's prompt.", done=False)

                # Include routing LLM's response as a collapsible UI element if the valve is enabled
                if self.valves.show_routing_response_enabled:
                    await self.emit_output(
                        content=routing_response_content,
                        include_collapsible=True,
                        messages=messages
                    )
                    self.log_debug(f"[inlet] Included routing LLM's response as a collapsible element in the payload.")
                    if self.valves.status_emits_enabled:
                        await self.emit_status("Included routing LLM's response as a collapsible element in the payload.", done=False)

                # Set the flag indicating model selection was performed
                model_selected = True

            # Step 9: Validate the updated payload structure
            self.validate_payload(body)
            self.log_debug("[inlet] Payload validation after update passed.")
            if self.valves.status_emits_enabled:
                await self.emit_status("Final payload validation passed.", done=False)

            # Step 10: Emit final status based on elapsed time
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            elapsed_time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

            if model_selected:
                if elapsed_time >= 1:
                    status_message = f"Using model '{body['model']}' ({elapsed_time_str})"
                else:
                    status_message = f"Using model '{body['model']}'"
            else:
                # This case should not occur as model_selected is set when consultation is triggered
                status_message = f"Using model '{body['model']}'"

            self.log_debug(f"[inlet] Model consultation took {elapsed_time_str}.")
            if self.valves.status_emits_enabled:
                await self.emit_status(
                    status_message,
                    done=True
                )

            return body  # Return the updated payload with the selected model
        except HTTPException as he:
            self.log_debug(f"[inlet] HTTPException: {he.detail}")
            raise he  # Propagate HTTPException to FastAPI to handle it properly
        except ValueError as ve:
            self.log_debug(f"[inlet] ValueError: {ve}")
            if self.valves.status_emits_enabled:
                await self.emit_error(f"Error: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            self.log_debug(f"[inlet] Unexpected exception: {e}")
            if self.valves.status_emits_enabled:
                await self.emit_error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            elapsed_time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            self.log_debug(
                f"[inlet] Inlet processing completed in {elapsed_time_str}."
            )
            # Note: The actual status emission is handled above based on model_selected and elapsed_time.

    async def handle_streaming_response(self, response: Any) -> str:
        """
        Handle and accumulate a StreamingResponse.
        """
        collected_output = ""

        async for chunk in response.body_iterator:
            try:
                # Decode chunk to string if it's bytes
                if isinstance(chunk, bytes):
                    try:
                        chunk_str = chunk.decode("utf-8")
                    except UnicodeDecodeError as e:
                        self.log_tech_support(f"[handle_streaming_response] Chunk decode error: {e}")
                        continue
                elif isinstance(chunk, str):
                    chunk_str = chunk
                else:
                    self.log_tech_support(f"[handle_streaming_response] Unexpected chunk type: {type(chunk)}")
                    continue

                self.log_debug(f"[handle_streaming_response] Received chunk: {chunk_str}")

                # Strip 'data: ' prefix if present
                if chunk_str.startswith("data: "):
                    chunk_str = chunk_str[6:]

                # End the stream if '[DONE]' signal is received
                if chunk_str.strip() == "[DONE]":
                    self.log_debug("[handle_streaming_response] Received [DONE] signal. Ending stream.")
                    break

                # Skip empty chunks
                if not chunk_str.strip():
                    self.log_debug("[handle_streaming_response] Empty chunk received. Skipping.")
                    continue

                # Parse JSON content
                try:
                    data = json.loads(chunk_str)
                    choice = data.get("choices", [{}])[0]
                    delta_content = choice.get("delta", {}).get("content", "")

                    # Append valid content to the collected output
                    if delta_content:
                        collected_output += delta_content
                        self.log_debug(f"[handle_streaming_response] Accumulated content: {delta_content}")
                    else:
                        self.log_tech_support(f"[handle_streaming_response] No content found in chunk: {chunk_str}")

                except json.JSONDecodeError as e:
                    self.log_tech_support(f"[handle_streaming_response] JSON decoding error: {e} for chunk: {chunk_str}")
                    # Optionally, append unparsed chunk or handle differently
                except Exception as e:
                    self.log_tech_support(f"[handle_streaming_response] Error processing JSON chunk: {e}")

            except Exception as e:
                self.log_tech_support(f"[handle_streaming_response] Unexpected error: {e}")

        self.log_debug(f"[handle_streaming_response] Final collected output: {collected_output}")
        return collected_output  # do not .strip()

    async def emit_status(self, description: str, done: bool):
        """
        Emits a status event to inform the user about the current process.
        """
        event = {
            "type": "status",
            "data": {
                "description": description,
                "done": done
            }
        }
        self.log_debug(f"[emit_status] Emitting status: {description}, done: {done}")
        try:
            if self.__event_emitter__:
                await self.__event_emitter__(event)
            else:
                self.log_debug("[emit_status] Event emitter not set.")
        except Exception as e:
            self.log_tech_support(f"[emit_status] Failed to emit status: {e}")

    async def emit_error(self, message: str):
        """
        Emits an error event to inform the user about an issue.
        """
        event = {
            "type": "error",
            "data": {
                "message": message
            }
        }
        self.log_debug(f"[emit_error] Emitting error: {message}")
        try:
            if self.__event_emitter__:
                await self.__event_emitter__(event)
            else:
                self.log_debug("[emit_error] Event emitter not set.")
        except Exception as e:
            self.log_tech_support(f"[emit_error] Failed to emit error: {e}")

    def update_system_prompt(
        self, body: Dict[str, Any], new_system_prompt: str
    ) -> None:
        """
        Replaces the system role content in the body with the new system prompt.
        """
        try:
            messages = body.get("messages", [])
            system_found = False
            for message in messages:
                if message.get("role") == "system":
                    self.log_debug("[update_system_prompt] Updating existing system prompt.")
                    message["content"] = new_system_prompt
                    system_found = True
                    break
            if not system_found:
                # If no system message is found, add one at the beginning
                self.log_debug("[update_system_prompt] No system prompt found. Adding a new one.")
                messages.insert(0, {"role": "system", "content": new_system_prompt})
                body["messages"] = messages
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_status("System prompt updated.", done=False))
        except Exception as e:
            self.log_tech_support(f"[update_system_prompt] Exception occurred: {e}")
            if self.valves.status_emits_enabled:
                asyncio.create_task(self.emit_error(f"Error updating system prompt: {str(e)}"))
            raise
        finally:
            self.log_debug("[update_system_prompt] Completed updating system prompt.")
