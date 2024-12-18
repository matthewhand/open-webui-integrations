"""
title: Flowise Manifold, for connecting to external Flowise instance endpoint
author: matthewh
version: 3.2
license: MIT
required_open_webui_version: 0.4.4

TODO
- [x] Update to OWUI 0.4.x
- [x] Upgrade to Manifold
  - [x] Static chatflow definitions
  - [x] Dynamic chatflow definitions
- [x] Flowise Assistants
- [x] Precise prompt
  - [x] Toggle for entire chat history 
  - [x] Toggle for system prompt
- [x] LLM features
  - [x] Summarization
  - [x] Status updates
- [x] Add Citation Emission
- [x] Add Session Mapping
- [x] Ensure all try blocks have try/catch/finally
- [x] Increment version number and update TODO list
"""

import json
import time
import re
import logging
import asyncio
from typing import List, Optional, Callable, Dict, Any, Union

from pydantic import BaseModel, Field
from dataclasses import dataclass

import requests  # Synchronous HTTP requests
from open_webui.main import generate_chat_completions
from open_webui.utils.misc import pop_system_message, get_last_user_message


@dataclass
class User:
    id: str
    username: str
    name: str
    role: str
    email: str


# Instantiate a mock user (Replace with actual user handling)
mock_user = User(
    id="flowise_manifold",
    username="flowise_manifold",
    name="Flowise Manifold",
    role="admin",
    email="admin@flowise.local",
)

# Global Placeholders (Replace with actual configurations)
FLOWISE_API_ENDPOINT_PLACEHOLDER = "YOUR_FLOWISE_API_ENDPOINT"
FLOWISE_API_KEY_PLACEHOLDER = "YOUR_FLOWISE_API_KEY"
FLOWISE_CHATFLOW_IDS_PLACEHOLDER = "YOUR_FLOWISE_CHATFLOW_IDS"
FLOWISE_ASSISTANT_IDS_PLACEHOLDER = "YOUR_FLOWISE_ASSISTANT_IDS"  # NEW
MANIFOLD_PREFIX_DEFAULT = "flowise/"


class SessionManager:
    """
    Manages user sessions, including chat history and Flowise response mappings.
    """

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)

    def get_session(self, user_id: str) -> Dict[str, Any]:
        try:
            if user_id not in self.sessions:
                self.sessions[user_id] = {"chat_history": [], "session_id": self.generate_session_id()}
                self.log.debug(f"[SessionManager] Created new session for user '{user_id}'.")
            else:
                self.log.debug(f"[SessionManager] Retrieved existing session for user '{user_id}'.")
            return self.sessions[user_id]
        except Exception as e:
            self.log.error(f"[SessionManager] Error retrieving session for user '{user_id}': {e}", exc_info=True)
            raise

    def update_session(self, user_id: str, key: str, value: Any):
        try:
            session = self.get_session(user_id)
            session[key] = value
            self.log.debug(f"[SessionManager] Updated session '{key}' for user '{user_id}' with value '{value}'.")
        except Exception as e:
            self.log.error(f"[SessionManager] Error updating session '{key}' for user '{user_id}': {e}", exc_info=True)
            raise

    def append_to_history(self, user_id: str, role: str, content: str):
        try:
            session = self.get_session(user_id)
            session["chat_history"].append({"role": role, "content": content})
            self.log.debug(f"[SessionManager] Appended to chat_history for user '{user_id}': {role}: {content}")
        except Exception as e:
            self.log.error(f"[SessionManager] Error appending to history for user '{user_id}': {e}", exc_info=True)
            raise

    def generate_session_id(self) -> str:
        try:
            session_id = f"session_{int(time.time() * 1000)}"
            self.log.debug(f"[SessionManager] Generated new session ID: {session_id}")
            return session_id
        except Exception as e:
            self.log.error(f"[SessionManager] Error generating session ID: {e}", exc_info=True)
            raise


class Pipe:
    """
    The Pipe class manages interactions with Flowise APIs, including dynamic and static chatflow and assistant retrieval,
    model registration with a manifold prefix, handling requests with internal retry logic,
    integrating summarization, and managing status updates.
    """

    class Valves(BaseModel):
        """
        Configuration for the Flowise Pipe.
        """

        # Flowise Connection
        flowise_api_endpoint: str = Field(
            default="http://host.docker.internal:3030/",
            description="Base URL for the Flowise API endpoint.",
        )
        flowise_api_key: str = Field(
            default=FLOWISE_API_KEY_PLACEHOLDER,
            description="API key for Flowise. Required for dynamic chatflow and assistant retrieval.",
        )
        use_dynamic_chatflows: bool = Field(
            default=False,  # Disabled by default
            description="Enable dynamic retrieval of chatflows. Requires valid API key.",
        )
        use_static_chatflows: bool = Field(
            default=True,  # Enabled by default
            description="Enable static retrieval of chatflows from 'flowise_chatflow_ids'.",
        )
        flowise_chatflow_ids: str = Field(
            default=FLOWISE_CHATFLOW_IDS_PLACEHOLDER,
            description="Comma-separated 'Name:ID' pairs for static chatflows.",
        )
        # NEW: Assistants Configuration
        use_dynamic_assistants: bool = Field(
            default=False,  # Disabled by default
            description="Enable dynamic retrieval of assistants. Requires valid API key.",  # NEW
        )
        use_static_assistants: bool = Field(
            default=False,  # Disabled by default
            description="Enable static retrieval of assistants from 'flowise_assistant_ids'.",  # NEW
        )
        flowise_assistant_ids: str = Field(
            default=FLOWISE_ASSISTANT_IDS_PLACEHOLDER,
            description="Comma-separated 'Name:ID' pairs for static assistants.",  # NEW
        )
        manifold_prefix: str = Field(
            default=MANIFOLD_PREFIX_DEFAULT,
            description="Prefix used for Flowise models.",
        )

        chatflow_blacklist: str = Field(
            default="(wip|dev|offline|broken)",  # Added parentheses for grouping
            description="Regex to exclude certain chatflows by name.",
        )
        assistant_blacklist: str = Field(
            default="(wip|dev|offline|broken)",  # NEW: Separate blacklist for assistants
            description="Regex to exclude certain assistants by name.",  # NEW
        )

        # Summarization Configuration
        enable_summarization: bool = Field(
            default=False,
            description="Enable chat history summarization.",
        )
        summarization_output: bool = Field(
            default=False,
            description="Output summarization to user using collapsible UI element.",
        )
        summarization_model_id: str = Field(
            default="",
            description="Model ID for summarization tasks.",
        )
        summarization_system_prompt: str = Field(
            default="Provide a concise summary of user and assistant messages separately.",
            description="System prompt for summarization.",
        )

        # Status Updates
        enable_llm_status_updates: bool = Field(
            default=False,
            description="Enable LLM-generated status updates. If false, uses static messages.",
        )
        llm_status_update_model_id: str = Field(
            default="",
            description="Model ID for LLM-based status updates.",
        )
        llm_status_update_prompt: str = Field(
            default="Acknowledge the user's patience waiting for: {last_request}",
            description="System prompt for LLM status updates. {last_request} replaced at runtime.",
        )
        llm_status_update_frequency: int = Field(
            default=5,
            description="Emit LLM-based status updates every Nth interval.",
        )
        static_status_messages: List[str] = Field(
            default=[
                "Processing...",
                "Please be patient...",
                "This will take a moment...",
                "Handling your request...",
                "Working on it...",
                "Almost there...",
                "Just a sec...",
            ],
            description="Static messages if LLM updates are not used.",
        )
        enable_timer_suffix: bool = Field(
            default=True,
            description="Add elapsed time to status messages.",
        )

        include_system_prompt: bool = Field(
            default=False,
            description="Include the system prompt in the request if True.",
        )

        post_entire_chat: bool = Field(
            default=False,
            description="Include the entire chat history in the request if True; otherwise, use the latest user message.",
        )

        # Message History and Prompt
        MAX_HISTORY: int = Field(
            default=0,
            description="Max previous messages to include; 0 means no limit.",
        )

        # Debug and General Configuration
        request_timeout: int = Field(
            default=300,
            description="HTTP client timeout in seconds.",
        )
        chatflow_load_timeout: int = Field(
            default=15,
            description="Timeout in seconds for loading chatflows.",
        )
        pipe_processing_timeout: int = Field(
            default=15,
            description="Max time in seconds for processing the pipe method.",
        )
        enable_debug: bool = Field(
            default=False,
            description="Enable or disable debug logging.",
        )

        # Additional Configuration for Status Updates
        emit_interval: int = Field(
            default=5,
            description="Interval in seconds between status updates.",
        )

    def __init__(self, valves: Optional["Pipe.Valves"] = None):
        """
        Initialize the Pipe with default valves and necessary state variables.
        """
        try:
            # Setup logging
            self.log = logging.getLogger(self.__class__.__name__)
            self.log.setLevel(logging.DEBUG)  # Set to DEBUG initially
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            if not self.log.handlers:
                self.log.addHandler(handler)

            # Initialize valves
            if valves:
                self.valves = valves
                self.log_debug("[INIT] Using externally provided valves.")
            else:
                self.valves = self.Valves()
                self.log_debug("[INIT] Using default valves configuration.")

            # Update logging level based on configuration
            self.log.setLevel(logging.DEBUG if self.valves.enable_debug else logging.INFO)

            # Assign id and name using manifold_prefix
            self.id = "flowise_manifold"
            self.name = self.valves.manifold_prefix
            self.log_debug(f"[INIT] Assigned ID: {self.id}, Name: {self.name}")

            # Initialize other attributes
            self.session_manager = SessionManager()
            self.chatflows: Dict[str, str] = {}
            self.last_emit_time: float = 0.0  # For status updates

            # Log valve configurations
            self.log_debug("[INIT] Valve configurations:")
            try:
                config_dict = self.valves.dict()
                for k, v in config_dict.items():
                    if "key" in k.lower() or "password" in k.lower():
                        self.log_debug(f"  {k}: <hidden>")
                    else:
                        self.log_debug(f"  {k}: {v}")
            except Exception as e:
                self.log_error(f"[INIT] Error printing config: {e}", exc_info=True)
            finally:
                self.log_debug("[INIT] Finished logging valve configurations.")

            # Chatflows and Assistants will be loaded in pipes()
            self.log_debug(
                "[INIT] Chatflows and Assistants will be loaded in pipes() method."
            )

            self.log_debug("[INIT] Pipe initialization complete.")
        except Exception as e:
            self.log_error(f"[INIT] Unexpected error during initialization: {e}", exc_info=True)
        finally:
            self.log_debug("[INIT] Initialization process finished.")

    def log_debug(self, message: str):
        """Log debug messages."""
        if self.valves.enable_debug:
            self.log.debug(message)

    def log_error(self, message: str, exc_info: Optional[bool] = False):
        """Log error messages, optionally including exception info."""
        self.log.error(message, exc_info=exc_info)

    def emit_citation(self, __event_emitter__, tool_output: str, tool_name: str):
        """
        Emit a citation event.

        Args:
            __event_emitter__: Event handler.
            tool_output (str): Content for citation.
            tool_name (str): Source of the citation.
        """
        try:
            citation_event = {
                "type": "citation",
                "data": {
                    "document": [tool_output],
                    "metadata": [{"source": tool_name}],
                    "source": {"name": tool_name},
                },
            }
            self.log_debug(f"[emit_citation] Constructed citation event: {citation_event}")

            if asyncio.iscoroutinefunction(__event_emitter__):
                self.log_debug("[emit_citation] Detected asynchronous event emitter.")
                asyncio.create_task(__event_emitter__(citation_event))
            else:
                self.log_debug("[emit_citation] Detected synchronous event emitter.")
                __event_emitter__(citation_event)

            self.log_debug("[emit_citation] Citation event emitted successfully.")
        except Exception as e:
            self.log_error(f"[emit_citation] Error emitting citation: {e}", exc_info=True)
        finally:
            self.log_debug("[emit_citation] Finished emitting citation.")

    # Existing methods (Valves, load_chatflows, load_static_models, load_dynamic_models, pipes, register_flowise_setup_pipe, get_chatflows, get_last_user_message, call_llm, generate_summary, generate_status_update, handle_flowise_request, clean_response_text, _get_combined_prompt, reset_state, emit_status_sync, emit_output_sync, pipe)
    # Ensure that each method contains proper try/catch/finally blocks as demonstrated above.

    def is_dynamic_config_ok(self) -> bool:
        """Check if dynamic chatflow configuration is valid."""
        self.log_debug("[is_dynamic_config_ok] Checking dynamic chatflow configuration.")
        try:
            if not self.valves.use_dynamic_chatflows:
                self.log_debug("[is_dynamic_config_ok] Dynamic chatflows disabled.")
                return False
            if (
                not self.valves.flowise_api_key
                or self.valves.flowise_api_key == FLOWISE_API_KEY_PLACEHOLDER
            ):
                self.log_error(
                    "[is_dynamic_config_ok] Dynamic chatflows enabled but API key missing/placeholder."
                )
                return False
            self.log_debug("[is_dynamic_config_ok] Dynamic chatflow configuration is valid.")
            return True
        except Exception as e:
            self.log_error(f"[is_dynamic_config_ok] Unexpected error: {e}", exc_info=True)
            return False
        finally:
            self.log_debug("[is_dynamic_config_ok] Finished dynamic chatflow configuration check.")

    def is_static_config_ok(self) -> bool:
        """Check if static chatflow configuration is valid."""
        self.log_debug("[is_static_config_ok] Checking static chatflow configuration.")
        try:
            if not self.valves.use_static_chatflows:
                self.log_debug("[is_static_config_ok] Static chatflows disabled.")
                return False
            if (
                not self.valves.flowise_chatflow_ids
                or self.valves.flowise_chatflow_ids == FLOWISE_CHATFLOW_IDS_PLACEHOLDER
            ):
                self.log_error(
                    "[is_static_config_ok] Static chatflows enabled but config empty/placeholder."
                )
                return False
            self.log_debug("[is_static_config_ok] Static chatflow configuration is valid.")
            return True
        except Exception as e:
            self.log_error(f"[is_static_config_ok] Unexpected error: {e}", exc_info=True)
            return False
        finally:
            self.log_debug("[is_static_config_ok] Finished static chatflow configuration check.")

    # NEW: Check if dynamic assistant configuration is valid
    def is_dynamic_assistants_config_ok(self) -> bool:
        """Check if dynamic assistant configuration is valid."""
        self.log_debug(
            "[is_dynamic_assistants_config_ok] Checking dynamic assistant configuration."
        )
        try:
            if not self.valves.use_dynamic_assistants:
                self.log_debug("[is_dynamic_assistants_config_ok] Dynamic assistants disabled.")
                return False
            if (
                not self.valves.flowise_api_key
                or self.valves.flowise_api_key == FLOWISE_API_KEY_PLACEHOLDER
            ):
                self.log_error(
                    "[is_dynamic_assistants_config_ok] Dynamic assistants enabled but API key missing/placeholder."
                )
                return False
            self.log_debug("[is_dynamic_assistants_config_ok] Dynamic assistant configuration is valid.")
            return True
        except Exception as e:
            self.log_error(f"[is_dynamic_assistants_config_ok] Unexpected error: {e}", exc_info=True)
            return False
        finally:
            self.log_debug("[is_dynamic_assistants_config_ok] Finished dynamic assistant configuration check.")

    # NEW: Check if static assistant configuration is valid
    def is_static_assistants_config_ok(self) -> bool:
        """Check if static assistant configuration is valid."""
        self.log_debug("[is_static_assistants_config_ok] Checking static assistant configuration.")
        try:
            if not self.valves.use_static_assistants:
                self.log_debug("[is_static_assistants_config_ok] Static assistants disabled.")
                return False
            if (
                not self.valves.flowise_assistant_ids
                or self.valves.flowise_assistant_ids == FLOWISE_ASSISTANT_IDS_PLACEHOLDER
            ):
                self.log_error(
                    "[is_static_assistants_config_ok] Static assistants enabled but config empty/placeholder."
                )
                return False
            self.log_debug("[is_static_assistants_config_ok] Static assistant configuration is valid.")
            return True
        except Exception as e:
            self.log_error(f"[is_static_assistants_config_ok] Unexpected error: {e}", exc_info=True)
            return False
        finally:
            self.log_debug("[is_static_assistants_config_ok] Finished static assistant configuration check.")

    def load_chatflows(self) -> Dict[str, str]:
        """
        Load dynamic and static chatflows and assistants based on configuration.
        Returns:
            Dict[str, str]: Loaded models with names as keys and IDs as values.
        """
        self.log_debug("[load_chatflows] Starting chatflow and assistant loading process.")
        loaded_models = {}
        try:
            # Load static chatflows if enabled
            if self.valves.use_static_chatflows and self.is_static_config_ok():
                self.log_debug("[load_chatflows] Loading static chatflows.")
                static_chatflows = self.load_static_models(
                    self.valves.flowise_chatflow_ids,
                    model_type="chatflow",
                    blacklist_regex=self.valves.chatflow_blacklist,
                )
                loaded_models.update(static_chatflows)
                self.log_debug(f"[load_chatflows] Loaded static chatflows: {static_chatflows}")
            else:
                if self.valves.use_static_chatflows:
                    self.log_debug(
                        "[load_chatflows] Static chatflows enabled but configuration invalid. Skipping static loading."
                    )

            # Load dynamic chatflows if enabled
            if self.valves.use_dynamic_chatflows and self.is_dynamic_config_ok():
                self.log_debug("[load_chatflows] Loading dynamic chatflows.")
                dynamic_chatflows = self.load_dynamic_models(
                    endpoint_suffix="chatflows",
                    model_type="chatflow",
                    blacklist_regex=self.valves.chatflow_blacklist,
                )
                loaded_models.update(dynamic_chatflows)
                self.log_debug(f"[load_chatflows] Loaded dynamic chatflows: {dynamic_chatflows}")
            else:
                if self.valves.use_dynamic_chatflows:
                    self.log_debug(
                        "[load_chatflows] Dynamic chatflows enabled but configuration invalid. Skipping dynamic loading."
                    )

            # NEW: Load static assistants if enabled
            if (
                self.valves.use_static_assistants
                and self.is_static_assistants_config_ok()
            ):
                self.log_debug("[load_chatflows] Loading static assistants.")
                static_assistants = self.load_static_models(
                    self.valves.flowise_assistant_ids,
                    model_type="assistant",
                    blacklist_regex=self.valves.assistant_blacklist,
                )
                loaded_models.update(static_assistants)
                self.log_debug(f"[load_chatflows] Loaded static assistants: {static_assistants}")
            else:
                if self.valves.use_static_assistants:
                    self.log_debug(
                        "[load_chatflows] Static assistants enabled but configuration invalid. Skipping static loading."
                    )

            # NEW: Load dynamic assistants if enabled
            if (
                self.valves.use_dynamic_assistants
                and self.is_dynamic_assistants_config_ok()
            ):
                self.log_debug("[load_chatflows] Loading dynamic assistants.")
                dynamic_assistants = self.load_dynamic_models(
                    endpoint_suffix="assistants",
                    model_type="assistant",
                    blacklist_regex=self.valves.assistant_blacklist,
                )
                loaded_models.update(dynamic_assistants)
                self.log_debug(f"[load_chatflows] Loaded dynamic assistants: {dynamic_assistants}")
            else:
                if self.valves.use_dynamic_assistants:
                    self.log_debug(
                        "[load_chatflows] Dynamic assistants enabled but configuration invalid. Skipping dynamic loading."
                    )

            # Update self.chatflows
            self.chatflows = loaded_models
            self.log_debug(f"[load_chatflows] Final loaded models: {self.chatflows}")

            return self.chatflows

        except Exception as e:
            self.log_error(f"[load_chatflows] Unexpected error: {e}", exc_info=True)
            return loaded_models
        finally:
            self.log_debug("[load_chatflows] Completed chatflow and assistant loading process.")

    def load_static_models(
        self, ids_str: str, model_type: str, blacklist_regex: str
    ) -> Dict[str, str]:
        """
        Load static models (chatflows or assistants) based on provided IDs.

        Args:
            ids_str (str): Comma-separated 'Name:ID' pairs.
            model_type (str): Type of model ('chatflow' or 'assistant').
            blacklist_regex (str): Regex pattern to blacklist model names.

        Returns:
            Dict[str, str]: Loaded static models.
        """
        static_models = {}
        try:
            self.log_debug(f"[load_static_models] Starting static {model_type} retrieval.")

            pairs = [
                pair.strip()
                for pair in ids_str.split(",")
                if ":" in pair
            ]
            self.log_debug(f"[load_static_models] Extracted pairs: {pairs}")

            for pair in pairs:
                try:
                    name, model_id = map(str.strip, pair.split(":", 1))
                    self.log_debug(
                        f"[load_static_models] Processing pair: name='{name}', id='{model_id}'"
                    )

                    if not name or not model_id:
                        self.log_debug(f"[load_static_models] Skipping invalid pair: '{pair}'")
                        continue

                    # Validate model_id
                    if not re.match(r"^[a-zA-Z0-9_-]+$", model_id):
                        self.log_debug(
                            f"[load_static_models] Invalid ID '{model_id}' for pair '{pair}'. Skipping."
                        )
                        continue

                    # Apply blacklist regex
                    if re.search(blacklist_regex, name, re.IGNORECASE):
                        self.log_debug(
                            f"[load_static_models] {model_type.capitalize()} '{name}' is blacklisted. Skipping."
                        )
                        continue

                    # Sanitize name
                    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                    self.log_debug(f"[load_static_models] Sanitized name: '{sanitized_name}'")

                    if sanitized_name in static_models:
                        base_name = sanitized_name
                        suffix = 1
                        while sanitized_name in static_models:
                            sanitized_name = f"{base_name}_{suffix}"
                            suffix += 1
                        self.log_debug(
                            f"[load_static_models] Resolved duplicate name: '{name}' -> '{sanitized_name}'"
                        )

                    static_models[sanitized_name] = model_id
                    self.log_debug(
                        f"[load_static_models] Added static {model_type}: '{sanitized_name}': '{model_id}'"
                    )

                except ValueError as ve:
                    self.log_error(
                        f"[load_static_models] Error parsing pair '{pair}': {ve}",
                        exc_info=True,
                    )
                except Exception as e:
                    self.log_error(
                        f"[load_static_models] Unexpected error processing pair '{pair}': {e}",
                        exc_info=True,
                    )

            self.log_debug(
                f"[load_static_models] Successfully loaded static {model_type}s: {static_models}"
            )
            return static_models

        except Exception as e:
            self.log_error(
                f"[load_static_models] Unexpected error during static {model_type} loading: {e}",
                exc_info=True,
            )
            return static_models
        finally:
            self.log_debug(f"[load_static_models] Finished static {model_type} retrieval.")

    def load_dynamic_models(
        self, endpoint_suffix: str, model_type: str, blacklist_regex: str, retries: int = 3, delay: int = 5
    ) -> Dict[str, str]:
        """
        Load dynamic models (chatflows or assistants) using the Flowise API, with enhanced debugging and retry logic.

        Args:
            endpoint_suffix (str): API endpoint suffix ('chatflows' or 'assistants').
            model_type (str): Type of model ('chatflow' or 'assistant').
            blacklist_regex (str): Regex pattern to blacklist model names.
            retries (int): Number of retry attempts in case of failure.
            delay (int): Delay in seconds between retries.

        Returns:
            Dict[str, str]: Loaded dynamic models.
        """
        dynamic_models = {}
        try:
            self.log_debug(f"[load_dynamic_models] Starting dynamic {model_type} retrieval.")

            endpoint = (
                f"{self.valves.flowise_api_endpoint.rstrip('/')}/api/v1/{endpoint_suffix}"
            )
            headers = {"Authorization": f"Bearer {self.valves.flowise_api_key}"}

            self.log_debug(f"[load_dynamic_models] Endpoint: {endpoint}")
            self.log_debug(f"[load_dynamic_models] Headers: {headers}")

            for attempt in range(1, retries + 1):
                try:
                    self.log_debug(
                        f"[load_dynamic_models] Attempt {attempt} to retrieve {model_type}s."
                    )
                    response = requests.get(
                        endpoint,
                        headers=headers,
                        timeout=self.valves.chatflow_load_timeout,
                    )
                    self.log_debug(
                        f"[load_dynamic_models] Response status: {response.status_code}"
                    )
                    raw_response = response.text
                    self.log_debug(f"[load_dynamic_models] Raw response: {raw_response}")

                    if response.status_code != 200:
                        self.log_error(
                            f"[load_dynamic_models] API call failed with status: {response.status_code}."
                        )
                        raise ValueError(f"HTTP {response.status_code}: {raw_response}")

                    data = json.loads(raw_response)
                    self.log_debug(f"[load_dynamic_models] Parsed data: {data}")

                    for item in data:
                        if model_type == "chatflow":
                            name = item.get("name", "").strip()
                            model_id = item.get("id", "").strip()
                        elif model_type == "assistant":
                            details = json.loads(item.get("details", "{}"))
                            name = details.get("name", "").strip()
                            model_id = item.get("id", "").strip()
                        else:
                            self.log_debug(
                                f"[load_dynamic_models] Unknown model_type '{model_type}'. Skipping."
                            )
                            continue

                        self.log_debug(
                            f"[load_dynamic_models] Processing {model_type}: name='{name}', id='{model_id}'"
                        )

                        if not name or not model_id:
                            self.log_debug(
                                f"[load_dynamic_models] Skipping invalid entry: {item}"
                            )
                            continue

                        # Apply blacklist regex
                        if re.search(blacklist_regex, name, re.IGNORECASE):
                            self.log_debug(
                                f"[load_dynamic_models] {model_type.capitalize()} '{name}' is blacklisted. Skipping."
                            )
                            continue

                        # Sanitize name
                        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                        self.log_debug(
                            f"[load_dynamic_models] Sanitized name: '{sanitized_name}'"
                        )

                        if sanitized_name in dynamic_models:
                            base_name = sanitized_name
                            suffix = 1
                            while sanitized_name in dynamic_models:
                                sanitized_name = f"{base_name}_{suffix}"
                                suffix += 1
                            self.log_debug(
                                f"[load_dynamic_models] Resolved duplicate name: '{name}' -> '{sanitized_name}'"
                            )

                        dynamic_models[sanitized_name] = model_id
                        self.log_debug(
                            f"[load_dynamic_models] Added dynamic {model_type}: '{sanitized_name}': '{model_id}'"
                        )

                    self.log_debug(
                        f"[load_dynamic_models] Successfully loaded dynamic {model_type}s: {dynamic_models}"
                    )
                    return dynamic_models  # Exit successfully after loading

                except (requests.RequestException, ValueError) as e:
                    self.log_error(
                        f"[load_dynamic_models] Attempt {attempt} failed: {e}",
                        exc_info=True,
                    )
                    if attempt < retries:
                        self.log_debug(f"[load_dynamic_models] Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        self.log_error(
                            f"[load_dynamic_models] All retry attempts failed for {model_type}s."
                        )
                        self.flowise_available = False
                except Exception as e:
                    self.log_error(
                        f"[load_dynamic_models] Unexpected error: {e}",
                        exc_info=True,
                    )
                    self.flowise_available = False
                    break
        except Exception as e:
            self.log_error(
                f"[load_dynamic_models] Fatal error during dynamic {model_type} retrieval: {e}",
                exc_info=True,
            )
        finally:
            self.log_debug(
                f"[load_dynamic_models] Completed dynamic {model_type} retrieval process."
            )

        return dynamic_models

    def pipes(self) -> List[dict]:
        """
        Register all available chatflows and assistants, adding the setup pipe if no models are available.

        Returns:
            List[dict]: A list of models with their IDs and names.
        """
        self.log_debug("[pipes] Starting model registration.")
        models = []
        try:
            # Load models (chatflows and assistants) based on configuration
            self.load_chatflows()

            # If no models are available after loading, add the setup pipe
            if not self.chatflows:
                self.log_debug(
                    "[pipes] No models available after loading. Registering 'Flowise Setup' pipe."
                )
                self.register_flowise_setup_pipe()

            # Register all models as entries
            models = [{"id": name, "name": name} for name in self.chatflows.keys()]
            self.log_debug(f"[pipes] Registered models: {models}")
            return models

        except Exception as e:
            self.log_error(
                f"[pipes] Unexpected error during model registration: {e}",
                exc_info=True,
            )
            return models
        finally:
            self.log_debug("[pipes] Completed model registration.")

    def register_flowise_setup_pipe(self):
        """
        Register the 'Flowise Setup' pipe to emit advisory instructions.

        This pipe is used when no models are configured correctly.
        """
        try:
            advisory_message = (
                "Flowise Setup Required:\n\n"
                "No chatflows or assistants are currently registered. Please configure them by:\n"
                "1. Enabling dynamic retrieval with a valid Flowise API key.\n"
                "2. Enabling static retrieval by providing 'Name:ID' pairs in 'flowise_chatflow_ids' and/or 'flowise_assistant_ids'.\n\n"
                "Ensure that the Flowise API endpoint is correctly configured in 'flowise_api_endpoint'."
            )
            self.chatflows["Flowise Setup"] = "flowise_setup_pipe_id"
            self.log_debug(
                f"[register_flowise_setup_pipe] Registered 'Flowise Setup' pipe with ID 'flowise_setup_pipe_id'."
            )
        except Exception as e:
            self.log_error(f"[register_flowise_setup_pipe] Error registering setup pipe: {e}", exc_info=True)
        finally:
            self.log_debug("[register_flowise_setup_pipe] Finished registering 'Flowise Setup' pipe.")

    def get_chatflows(self) -> Dict[str, str]:
        """
        Retrieve all available chatflows and assistants.

        Returns:
            Dict[str, str]: A dictionary mapping model names to IDs.
        """
        self.log_debug("[get_chatflows] Retrieving all available models.")
        try:
            return self.chatflows.copy()
        except Exception as e:
            self.log_error(f"[get_chatflows] Error retrieving chatflows: {e}", exc_info=True)
            return {}

    def get_last_user_message(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Retrieve the most recent user message.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.

        Returns:
            Optional[str]: The content of the last user message, or None if not found.
        """
        try:
            last_message = get_last_user_message(messages)
            if last_message:
                self.log_debug(f"[get_last_user_message] Last user message: {last_message}")
                return last_message.get("content")
            self.log_debug("[get_last_user_message] No user message found.")
            return None
        except Exception as e:
            self.log_error(f"[get_last_user_message] Error retrieving last user message: {e}", exc_info=True)
            return None

    async def call_llm(
        self,
        base_model_id: str,
        messages: List[Dict[str, str]],
    ) -> Optional[str]:
        """
        Helper method to handle LLM chat completions with full conversation history.

        Args:
            base_model_id (str): The model ID to use.
            messages (List[Dict[str, str]]): The list of messages representing the conversation.

        Returns:
            Optional[str]: The generated content or None if failed.
        """
        try:
            payload = {
                "model": base_model_id,
                "messages": messages,
                # Removed 'max_tokens' and 'temperature' to allow base_model configuration
            }

            self.log_debug(
                f"[call_llm] Payload for generate_chat_completions: {json.dumps(payload, indent=4)}"
            )

            response = await generate_chat_completions(
                form_data=payload,
                bypass_filter=True,  # Ensure bypass_filter is included
            )
            self.log_debug(f"[call_llm] LLM Response: {response}")

            # Validate response structure
            if (
                "choices" in response
                and len(response["choices"]) > 0
                and "message" in response["choices"][0]
                and "content" in response["choices"][0]["message"]
            ):
                content = response["choices"][0]["message"]["content"].strip()
                self.log_debug(f"[call_llm] Generated Content Before Cleanup: {content}")
                cleaned_content = self.clean_response_text(content)
                self.log_debug(f"[call_llm] Generated Content After Cleanup: {cleaned_content}")
                return cleaned_content
            else:
                self.log_error("Invalid response structure from LLM.")
                self.log_debug(f"[call_llm] Full LLM Response: {json.dumps(response, indent=4)}")
                return None
        except Exception as e:
            self.log_error(f"[call_llm] Error during LLM call: {e}", exc_info=True)
            return None
        finally:
            self.log_debug("[call_llm] Finished LLM call.")

    def generate_summary(self, __user__: Optional[dict] = None) -> Optional[str]:
        """
        Generate a summary of the accumulated chat history using an LLM.

        Args:
            __user__ (Optional[dict]): The user information.

        Returns:
            Optional[str]: The generated summary if successful; otherwise, None.
        """
        try:
            if not self.valves.enable_summarization:
                self.log_debug("[generate_summary] Summarization is disabled.")
                return None

            if not self.valves.summarization_model_id:
                self.log_error("[generate_summary] Summarization model ID not configured.")
                return None

            # Collect chat history
            user_id = (
                __user__.get("user_id", "default_user") if __user__ else "default_user"
            )
            chat_session = self.session_manager.get_session(user_id)
            history = chat_session.get("chat_history", [])
            if not history:
                self.log_debug("[generate_summary] No chat history available for summarization.")
                return None

            user_messages = [msg["content"] for msg in history if msg["role"] == "user"]
            assistant_messages = [
                msg["content"] for msg in history if msg["role"] == "assistant"
            ]

            user_content = "\n".join(user_messages)
            assistant_content = "\n".join(assistant_messages)

            prompt_messages = [
                {"role": "system", "content": self.valves.summarization_system_prompt},
                {
                    "role": "user",
                    "content": f"User Messages:\n{user_content}\n\nAssistant Messages:\n{assistant_content}",
                },
            ]

            self.log_debug(
                f"[generate_summary] Generating summary with messages: {prompt_messages}"
            )

            # Run the async call_llm method synchronously
            summary = asyncio.run(
                self.call_llm(
                    base_model_id=self.valves.summarization_model_id,
                    messages=prompt_messages,
                )
            )

            if summary:
                self.log_debug(f"[generate_summary] Generated summary: {summary}")
                return summary

            self.log_debug("[generate_summary] Summary generation returned None.")
            return None

        except Exception as e:
            self.log_error(
                f"[generate_summary] Error generating summary: {e}", exc_info=True
            )
            return None
        finally:
            self.log_debug("[generate_summary] Finished generating summary.")

    def generate_status_update(self, last_request: str) -> Optional[str]:
        """
        Generate a status update using an LLM based on the last user request.

        Args:
            last_request (str): The user's last request.

        Returns:
            Optional[str]: The generated status update if successful; otherwise, None.
        """
        try:
            if not self.valves.enable_llm_status_updates:
                self.log_debug("[generate_status_update] LLM status updates are disabled.")
                return None

            if not self.valves.llm_status_update_model_id:
                self.log_error(
                    "[generate_status_update] LLM status update model ID not configured."
                )
                return None

            prompt = self.valves.llm_status_update_prompt.format(
                last_request=last_request
            )
            self.log_debug(
                f"[generate_status_update] Generating status update with prompt: {prompt}"
            )

            prompt_messages = [
                {"role": "system", "content": self.valves.llm_status_update_prompt},
                {"role": "user", "content": prompt},
            ]

            # Run the async call_llm method synchronously
            status_update = asyncio.run(
                self.call_llm(
                    base_model_id=self.valves.llm_status_update_model_id,
                    messages=prompt_messages,
                )
            )

            if status_update:
                self.log_debug(
                    f"[generate_status_update] Generated status update: {status_update}"
                )
                return status_update

            self.log_debug("[generate_status_update] Status update generation returned None.")
            return None

        except Exception as e:
            self.log_error(
                f"[generate_status_update] Error generating status update: {e}",
                exc_info=True,
            )
            return None
        finally:
            self.log_debug("[generate_status_update] Finished generating status update.")

    def handle_flowise_request(
        self,
        question: str,
        __user__: Optional[dict],
        __event_emitter__: Optional[Callable[[dict], Any]],
        chatflow_name: str = "",
    ) -> Union[Dict[str, Any], Dict[str, str]]:
        """
        Send the prompt to Flowise for processing.

        Args:
            question (str): The user's question.
            __user__ (Optional[dict]): User information dictionary.
            __event_emitter__ (Optional[Callable[[dict], Any]]): Event emitter for status updates.
            chatflow_name (str): Name of the chatflow to use.

        Returns:
            Union[Dict[str, Any], Dict[str, str]]: Response from Flowise or error message.
        """
        self.log_debug(
            f"[handle_flowise_request] Handling request for '{chatflow_name}' question: {question!r}"
        )
        text = None  # Initialize to avoid undefined variable issues

        try:
            # Prepare payload
            payload = {"question": question}
            self.log_debug(f"[handle_flowise_request] Initial payload: {payload}")

            user_id = (
                __user__.get("user_id", "default_user") if __user__ else "default_user"
            )
            self.log_debug(f"[handle_flowise_request] User ID: {user_id}")

            session = self.session_manager.get_session(user_id)
            chat_id = session.get("session_id")
            self.log_debug(f"[handle_flowise_request] Current session ID: {chat_id}")

            if chat_id:
                payload["chatId"] = chat_id
                self.log_debug(f"[handle_flowise_request] Added chatId to payload: {chat_id}")

            # Optionally include the system prompt
            if self.valves.include_system_prompt and session.get("system_message"):
                system_message = session.get("system_message", "")
                payload["systemMessage"] = system_message
                self.log_debug(f"[handle_flowise_request] Included systemMessage: {system_message}")

            self.log_debug(f"[handle_flowise_request] Final payload: {payload}")

            # Determine model to use
            if not chatflow_name or chatflow_name not in self.chatflows:
                if self.chatflows:
                    chatflow_name = list(self.chatflows.keys())[0]
                    self.log_debug(
                        f"[handle_flowise_request] No or invalid chatflow_name provided. Using '{chatflow_name}'."
                    )
                else:
                    error_message = "No chatflows or assistants configured."
                    self.log_debug(f"[handle_flowise_request] {error_message}")
                    if __event_emitter__:
                        self.emit_status_sync(
                            __event_emitter__, error_message, done=True
                        )
                    return {"error": error_message}

            model_id = self.chatflows[chatflow_name]
            self.log_debug(f"[handle_flowise_request] Selected model ID: {model_id}")

            endpoint = self.valves.flowise_api_endpoint.rstrip("/")
            url = f"{endpoint}/api/v1/prediction/{model_id}"
            headers = {
                "Content-Type": "application/json",
            }
            if self.valves.flowise_api_key:
                headers["Authorization"] = f"Bearer {self.valves.flowise_api_key}"
            self.log_debug(
                f"[handle_flowise_request] Sending request to URL: {url} with headers: {headers}"
            )

            # Make the HTTP request
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.valves.request_timeout
            )
            self.log_debug(
                f"[handle_flowise_request] Response status: {response.status_code}"
            )
            response_text = response.text
            self.log_debug(f"[handle_flowise_request] Response text: {response_text!r}")

            if response.status_code != 200:
                error_message = (
                    f"Error: Flowise API call failed with status {response.status_code}"
                )
                self.log_debug(f"[handle_flowise_request] {error_message}")
                return {"error": error_message}

            # Parse the JSON response
            try:
                data = json.loads(response_text)
                self.log_debug(f"[handle_flowise_request] Parsed data: {data!r}")
            except json.JSONDecodeError:
                error_message = "Error: Invalid JSON response from Flowise."
                self.log_debug(f"[handle_flowise_request] {error_message}")
                return {"error": error_message}

            # Extract and clean the response text
            raw_text = data.get("text", "")
            self.log_debug(f"[handle_flowise_request] Raw response text: {raw_text!r}")
            text = self.clean_response_text(raw_text)
            self.log_debug(f"[handle_flowise_request] Cleaned response text: {text!r}")

            new_chat_id = data.get("chatId", chat_id)
            self.log_debug(f"[handle_flowise_request] New chat ID: {new_chat_id}")

            if not text:
                error_message = "Error: Empty response from Flowise."
                self.log_debug(f"[handle_flowise_request] {error_message}")
                return {"error": error_message}

            # Update chat session
            self.session_manager.update_session(user_id, "session_id", new_chat_id)
            self.session_manager.append_to_history(user_id, "assistant", text)

            self.log_debug(
                f"[handle_flowise_request] Updated history for user '{user_id}': {self.session_manager.get_session(user_id)['chat_history']}"
            )

            # Emit the Flowise response via the event emitter
            if __event_emitter__:
                self.log_debug(
                    f"[handle_flowise_request] Emitting Flowise response via emit_output_sync."
                )
                self.emit_output_sync(
                    __event_emitter__, text, include_collapsible=False
                )

            # Emit citations if 'usedTools' is present
            for tool in data.get("usedTools", []):
                tool_output = tool.get("toolOutput", "")
                tool_name = tool.get("tool", "")
                self.log_debug(f"[handle_flowise_request] Emitting citation for tool '{tool_name}'.")
                self.emit_citation(__event_emitter__, tool_output, tool_name)

            return {"response": text}

        except requests.exceptions.RequestException as e:
            # Handle any request-related exceptions (e.g., connection errors)
            error_message = f"Request failed: {str(e)}"
            self.log_debug(f"[handle_flowise_request] {error_message}")
            return {"error": error_message}

        except Exception as e:
            # Handle any other unexpected exceptions
            error_message = f"Error during Flowise request handling: {e}"
            self.log_error(f"[handle_flowise_request] {error_message}", exc_info=True)
            return {"error": error_message}

        finally:
            # Final logging actions
            if text:
                self.log_debug("[handle_flowise_request] Successfully handled Flowise request.")
            else:
                self.log_debug("[handle_flowise_request] Flowise request handling completed with errors.")

    def clean_response_text(self, text: str) -> str:
        """
        Cleans the response text by removing enclosing quotes and trimming whitespace.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        self.log_debug(f"[clean_response_text] Entering with: {text!r}")
        try:
            pattern = r'^([\'"])(.*)\1$'
            match = re.match(pattern, text)
            if match:
                text = match.group(2)
                self.log_debug(f"[clean_response_text] Stripped quotes: {text!r}")
            return text.strip()
        except Exception as e:
            self.log_error(f"[clean_response_text] Error: {e}", exc_info=True)
            return text
        finally:
            self.log_debug("[clean_response_text] Finished cleaning response text.")

    def _get_combined_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Combines user and assistant messages into a structured prompt.

        Example:
            User: Hi!
            Assistant: Hello! How can I help you today?
            User: Tell me a joke.
        """
        try:
            prompt_parts = [
                f"{message.get('role', 'user').capitalize()}: {message.get('content', '')}"
                for message in messages
            ]
            combined_prompt = "\n".join(prompt_parts)
            self.log_debug(f"[get_combined_prompt] Combined prompt:\n{combined_prompt}")
            return combined_prompt
        except Exception as e:
            self.log_error(f"[get_combined_prompt] Error combining prompts: {e}", exc_info=True)
            return ""

    def reset_state(self):
        """Reset per-request state variables without clearing chat_sessions."""
        try:
            # Reset per-request variables only
            self.start_time = time.time()  # Start time of the current pipe execution
            self.last_emit_time = 0.0
            self.log_debug("[reset_state] Per-request state variables have been reset.")
        except Exception as e:
            self.log_error(f"[reset_state] Unexpected error: {e}", exc_info=True)
        finally:
            self.log_debug("[reset_state] Finished resetting state.")

    def emit_status_sync(
        self,
        __event_emitter__: Callable[[dict], Any],
        message: str,
        done: bool,
    ):
        """Emit status updates to the event emitter synchronously."""
        try:
            if __event_emitter__:
                event = {
                    "type": "status",
                    "data": {"description": message, "done": done},
                }
                self.log_debug(f"[emit_status_sync] Preparing to emit status event: {event}")

                if asyncio.iscoroutinefunction(__event_emitter__):
                    self.log_debug("[emit_status_sync] Detected asynchronous event emitter.")
                    asyncio.create_task(__event_emitter__(event))
                else:
                    self.log_debug("[emit_status_sync] Detected synchronous event emitter.")
                    __event_emitter__(event)

                self.log_debug("[emit_status_sync] Status event emitted successfully.")
        except Exception as e:
            self.log_error(
                f"[emit_status_sync] Error emitting status event: {e}",
                exc_info=True,
            )
        finally:
            self.log_debug("[emit_status_sync] Finished emitting status event.")

    def emit_output_sync(
        self,
        __event_emitter__: Callable[[dict], Any],
        content: str,
        include_collapsible: bool = False,
    ):
        """Emit message updates to the event emitter synchronously."""
        try:
            if __event_emitter__ and content:
                # Prepare the message event
                if include_collapsible:
                    content = f"""
<details>
<summary>Click to expand summary</summary>
{content}
</details>
                    """.strip()
                message_event = {
                    "type": "message",
                    "data": {"content": content},
                }
                self.log_debug(f"[emit_output_sync] Preparing to emit message event: {message_event}")

                if asyncio.iscoroutinefunction(__event_emitter__):
                    self.log_debug("[emit_output_sync] Detected asynchronous event emitter.")
                    asyncio.create_task(__event_emitter__(message_event))
                else:
                    self.log_debug("[emit_output_sync] Detected synchronous event emitter.")
                    __event_emitter__(message_event)

                self.log_debug("[emit_output_sync] Message event emitted successfully.")
        except Exception as e:
            self.log_error(
                f"[emit_output_sync] Error emitting message event: {e}",
                exc_info=True,
            )
        finally:
            self.log_debug("[emit_output_sync] Finished emitting message event.")

    def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> Union[Dict[str, Any], Dict[str, str]]:
        """
        Processes a user request by selecting the appropriate model (chatflow or assistant) or the setup pipe.

        Args:
            body (dict): The incoming request payload.
            __user__ (Optional[dict]): The user information.
            __event_emitter__ (Optional[Callable[[dict], Any]]): The event emitter for sending messages.

        Returns:
            Union[Dict[str, Any], Dict[str, str]]: The Flowise output, setup instructions, or an error message.
        """
        try:
            # Reset state for the new request
            self.reset_state()
            self.request_id = str(time.time())
            self.log_debug(f"[pipe] Starting new request with ID: {self.request_id}")

            output = {}  # Initialize output to ensure it's defined

            # Retrieve and process the model name
            model_full_name = body.get("model", "").strip()
            self.log_debug(f"[pipe] Received model name: '{model_full_name}'")

            # Strip everything before and including the first period (.)
            if "." in model_full_name:
                stripped_model_name = model_full_name.split(".", 1)[1]
                self.log_debug(
                    f"[pipe] Detected period in model name. Stripped to: '{stripped_model_name}'"
                )
            else:
                stripped_model_name = model_full_name
                self.log_debug(
                    f"[pipe] No period detected in model name. Using as is: '{stripped_model_name}'"
                )

            # Sanitize the stripped model name
            sanitized_model_name = re.sub(r"[^a-zA-Z0-9_]", "_", stripped_model_name)
            self.log_debug(f"[pipe] Sanitized model name: '{sanitized_model_name}'")

            model_name = sanitized_model_name
            self.log_debug(f"[pipe] Final model name after processing: '{model_name}'")

            # Look up the model ID
            model_id = self.chatflows.get(model_name)
            if model_id:
                self.log_debug(
                    f"[pipe] Found model ID '{model_id}' for model name '{model_name}'."
                )
            else:
                self.log_debug(
                    f"[pipe] No model found for name '{model_name}'. Assuming 'Flowise Setup' pipe."
                )
                model_name = "Flowise Setup"
                model_id = self.chatflows.get(model_name)

            # Handle setup pipe
            if model_name == "Flowise Setup":
                advisory_message = (
                    "Flowise Setup Required:\n\n"
                    "No chatflows or assistants are currently registered. Please configure them by:\n"
                    "1. Enabling dynamic retrieval with a valid Flowise API key.\n"
                    "2. Enabling static retrieval by providing 'Name:ID' pairs in 'flowise_chatflow_ids' and/or 'flowise_assistant_ids'.\n\n"
                    "Ensure that the Flowise API endpoint is correctly configured in 'flowise_api_endpoint'."
                )
                self.log_debug(f"[pipe] Advisory message: {advisory_message}")
                if __event_emitter__:
                    self.emit_output_sync(
                        __event_emitter__, advisory_message, include_collapsible=False
                    )
                return {"status": "setup", "message": advisory_message}

            # At this point, model_id should be valid
            if not model_id:
                error_message = f"No model found for name '{model_name}'."
                self.log_error(f"[pipe] {error_message}")
                if __event_emitter__:
                    self.emit_status_sync(__event_emitter__, error_message, done=True)
                return {"status": "error", "message": error_message}

            # Process the request using the selected model
            self.log_debug(f"[pipe] Using model '{model_name}' with ID '{model_id}'")
            messages = body.get("messages", [])
            if not messages:
                error_message = "No messages found in the request."
                self.log_debug(f"[pipe] {error_message}")
                if __event_emitter__:
                    self.emit_status_sync(__event_emitter__, error_message, done=True)
                return {"status": "error", "message": error_message}

            # Extract system prompt and messages using pop_system_message
            system_message, user_messages = pop_system_message(messages)
            self.log_debug(f"[pipe] System message: {system_message}")
            self.log_debug(
                f"[pipe] User messages after popping system message: {user_messages}"
            )

            # Determine the question based on valves.post_entire_chat and enable_summarization
            if self.valves.post_entire_chat:
                self.log_debug("[pipe] post_entire_chat is True. Using the entire chat history.")
                combined_prompt = self._get_combined_prompt(user_messages)
            else:
                self.log_debug("[pipe] post_entire_chat is False. Using the most recent user message.")
                last_user_message = self.get_last_user_message(user_messages)
                if last_user_message:
                    combined_prompt = last_user_message
                else:
                    combined_prompt = self._get_combined_prompt(user_messages)

            self.log_debug(
                f"[pipe] Combined prompt for model '{model_name}':\n{combined_prompt}"
            )

            # Emit initial status: Processing the request
            if __event_emitter__:
                self.emit_status_sync(
                    __event_emitter__, "Processing your request", done=False
                )
                self.log_debug("[pipe] Initial status 'Processing your request' emitted.")

            # Generate summary before sending to Flowise if enabled
            summary = None
            if self.valves.enable_summarization:
                self.log_debug("[pipe] Summarization is enabled. Generating summary.")
                summary = self.generate_summary(__user__)
                if summary:
                    self.log_debug(f"[pipe] Generated summary: {summary}")
                    # Emit the summary via the event emitter within a collapsible section
                    if self.valves.summarization_output and __event_emitter__:
                        collapsible_summary = f"**Summary:**\n{summary}"
                        self.emit_output_sync(
                            __event_emitter__,
                            collapsible_summary,
                            include_collapsible=True,
                        )

            # Modify the prompt to include the summary if available
            if summary:
                combined_prompt_with_summary = f"{combined_prompt}\n\nSummary:\n{summary}"
                self.log_debug(
                    f"[pipe] Combined prompt with summary:\n{combined_prompt_with_summary}"
                )
            else:
                combined_prompt_with_summary = combined_prompt

            # Optionally include system prompt
            if self.valves.include_system_prompt and system_message:
                combined_prompt_with_summary = f"{system_message}\n\n{combined_prompt_with_summary}"
                self.log_debug(
                    f"[pipe] Included system message in prompt:\n{combined_prompt_with_summary}"
                )

            # Handle Flowise request
            output = self.handle_flowise_request(
                question=combined_prompt_with_summary,  # Updated to include summary and system prompt if enabled
                __user__=__user__,
                __event_emitter__=__event_emitter__,
                chatflow_name=model_name,
            )
            self.log_debug(f"[pipe] handle_flowise_request output: {output}")

            # Emit status updates based on configuration
            if self.valves.enable_llm_status_updates:
                self.log_debug(
                    "[pipe] LLM Status Updates are enabled. Generating status update."
                )
                # Retrieve the last user message for status update
                last_request = (
                    user_messages[-1]["content"]
                    if user_messages
                    else "your_last_request_placeholder"
                )
                status_message = self.generate_status_update(last_request)
                if status_message and __event_emitter__:
                    self.emit_status_sync(__event_emitter__, status_message, done=False)
            else:
                # Emit a static status update if LLM updates are not enabled
                if self.valves.static_status_messages and __event_emitter__:
                    static_message = self.valves.static_status_messages[0]  # Example: use the first static message
                    self.emit_status_sync(__event_emitter__, static_message, done=False)

            # Emit final status
            if __event_emitter__:
                if "response" in output:
                    final_status_message = "Request completed successfully."
                else:
                    final_status_message = "Request completed with errors."
                self.emit_status_sync(
                    __event_emitter__, final_status_message, done=True
                )
                self.log_debug(f"[pipe] Final status '{final_status_message}' emitted.")

            return output

        except Exception as e:
            # Handle any unexpected exceptions during the pipe processing
            error_message = f"Unexpected error during pipe processing: {e}"
            self.log_error(f"[pipe] {error_message}", exc_info=True)
            if __event_emitter__:
                self.emit_status_sync(__event_emitter__, error_message, done=True)
            return {"error": error_message}

        finally:
            # Ensure the output is returned if it hasn't been already
            # This prevents 'output' from being undefined
            if not output:
                self.log_debug("[pipe] No output generated. Returning empty response.")
                return {"error": "No output generated."}
            self.log_debug("[pipe] Pipe processing completed.")
