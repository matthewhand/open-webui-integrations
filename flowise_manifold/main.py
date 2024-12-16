"""
title: Flowise Manifold, for connecting to external Flowise instance endpoint
author: matthewh
version: 3.0
license: MIT
required_open_webui_version: 0.4.4

TODO
- [x] Update to OWUI 0.4.x
- [x] Upgrade to Manifold
 - [x] Static chatflow definitions
 - [x] Dynamic chatflow definitions
- [x] Flowise Assistants
- [ ] LLM features
 - [ ] Summarization
 - [ ] Status updates
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


# Mock functions for demonstration (Replace with actual implementations)
def generate_chat_completions(form_data):
    """
    Mock implementation of generate_chat_completions.
    Returns different responses based on the input prompt to simulate summarization and status updates.
    """
    time.sleep(0.1)  # Simulate network delay
    messages = form_data.get("messages", [])

    if any("summary" in msg.get("content", "").lower() for msg in messages):
        # Simulate a summarization response
        return {
            "choices": [
                {"message": {"content": "This is a mock summary of the conversation."}}
            ]
        }
    elif any("acknowledge" in msg.get("content", "").lower() for msg in messages):
        # Simulate a status update response
        return {
            "choices": [
                {
                    "message": {
                        "content": "Thank you for your patience. We are processing your request."
                    }
                }
            ]
        }
    else:
        # Default mock response
        return {
            "choices": [
                {
                    "message": {
                        "content": "Mocked response from generate_chat_completions."
                    }
                }
            ]
        }


def pop_system_message(messages):
    # Mock implementation: Assume the first message is a system message
    if messages and messages[0].get("role") == "system":
        return messages[0], messages[1:]
    return {}, messages


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
            description="(WIP) Enable chat history summarization.",
        )
        summarization_output: bool = Field(
            default=False,
            description="(WIP) Output summarization to user using collapsible UI element.",
        )
        summarization_model_id: str = Field(
            default="",
            description="(WIP) Model ID for summarization tasks.",
        )
        summarization_system_prompt: str = Field(
            default="Provide a concise summary of user and assistant messages separately.",
            description="(WIP) System prompt for summarization.",
        )

        # Status Updates
        enable_llm_status_updates: bool = Field(
            default=False,
            description="(WIP) Enable LLM-generated status updates. If false, uses static messages.",
        )
        llm_status_update_model_id: str = Field(
            default="",
            description="(WIP) Model ID for LLM-based status updates.",
        )
        llm_status_update_prompt: str = Field(
            default="Acknowledge the user's patience waiting for: {last_request}",
            description="(WIP) System prompt for LLM status updates. {last_request} replaced at runtime.",
        )
        llm_status_update_frequency: int = Field(
            default=5,
            description="(WIP) Emit LLM-based status updates every Nth interval.",
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
        self.chat_sessions: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.flowise_available: bool = True
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

    def log_debug(self, message: str):
        """Log debug messages."""
        if self.valves.enable_debug:
            self.log.debug(message)

    def log_error(self, message: str, exc_info: Optional[bool] = False):
        """Log error messages, optionally including exception info."""
        self.log.error(message, exc_info=exc_info)

    def is_dynamic_config_ok(self) -> bool:
        """Check if dynamic chatflow configuration is valid."""
        self.log_debug(
            "[is_dynamic_config_ok] Checking dynamic chatflow configuration."
        )
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
        self.log_debug(
            "[is_dynamic_config_ok] Dynamic chatflow configuration is valid."
        )
        return True

    def is_static_config_ok(self) -> bool:
        """Check if static chatflow configuration is valid."""
        self.log_debug("[is_static_config_ok] Checking static chatflow configuration.")
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

    # NEW: Check if dynamic assistant configuration is valid
    def is_dynamic_assistants_config_ok(self) -> bool:
        """Check if dynamic assistant configuration is valid."""
        self.log_debug(
            "[is_dynamic_assistants_config_ok] Checking dynamic assistant configuration."
        )
        if not self.valves.use_dynamic_assistants:
            self.log_debug(
                "[is_dynamic_assistants_config_ok] Dynamic assistants disabled."
            )
            return False
        if (
            not self.valves.flowise_api_key
            or self.valves.flowise_api_key == FLOWISE_API_KEY_PLACEHOLDER
        ):
            self.log_error(
                "[is_dynamic_assistants_config_ok] Dynamic assistants enabled but API key missing/placeholder."
            )
            return False
        self.log_debug(
            "[is_dynamic_assistants_config_ok] Dynamic assistant configuration is valid."
        )
        return True

    # NEW: Check if static assistant configuration is valid
    def is_static_assistants_config_ok(self) -> bool:
        """Check if static assistant configuration is valid."""
        self.log_debug(
            "[is_static_assistants_config_ok] Checking static assistant configuration."
        )
        if not self.valves.use_static_assistants:
            self.log_debug(
                "[is_static_assistants_config_ok] Static assistants disabled."
            )
            return False
        if (
            not self.valves.flowise_assistant_ids
            or self.valves.flowise_assistant_ids == FLOWISE_ASSISTANT_IDS_PLACEHOLDER
        ):
            self.log_error(
                "[is_static_assistants_config_ok] Static assistants enabled but config empty/placeholder."
            )
            return False
        self.log_debug(
            "[is_static_assistants_config_ok] Static assistant configuration is valid."
        )
        return True

    def load_chatflows(self) -> Dict[str, str]:
        """
        Load dynamic and static chatflows and assistants based on configuration.
        Returns:
            Dict[str, str]: Loaded models with names as keys and IDs as values.
        """
        self.log_debug(
            "[load_chatflows] Starting chatflow and assistant loading process."
        )
        loaded_models = {}

        try:
            # Load static chatflows if enabled
            if self.valves.use_static_chatflows and self.is_static_config_ok():
                self.log_debug("[load_chatflows] Loading static chatflows.")
                static_chatflows = self.load_static_chatflows()
                loaded_models.update(static_chatflows)
                self.log_debug(
                    f"[load_chatflows] Loaded static chatflows: {static_chatflows}"
                )
            else:
                if self.valves.use_static_chatflows:
                    self.log_debug(
                        "[load_chatflows] Static chatflows enabled but configuration invalid. Skipping static loading."
                    )

            # Load dynamic chatflows if enabled
            if self.valves.use_dynamic_chatflows and self.is_dynamic_config_ok():
                self.log_debug("[load_chatflows] Loading dynamic chatflows.")
                dynamic_chatflows = self.load_dynamic_chatflows()
                loaded_models.update(dynamic_chatflows)
                self.log_debug(
                    f"[load_chatflows] Loaded dynamic chatflows: {dynamic_chatflows}"
                )
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
                static_assistants = self.load_static_assistants()
                loaded_models.update(static_assistants)
                self.log_debug(
                    f"[load_chatflows] Loaded static assistants: {static_assistants}"
                )
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
                dynamic_assistants = self.load_dynamic_assistants()
                loaded_models.update(dynamic_assistants)
                self.log_debug(
                    f"[load_chatflows] Loaded dynamic assistants: {dynamic_assistants}"
                )
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
            self.log_debug(
                "[load_chatflows] Completed chatflow and assistant loading process."
            )

    def load_dynamic_chatflows(
        self, retries: int = 3, delay: int = 5
    ) -> Dict[str, str]:
        """
        Load chatflows dynamically using the Flowise API, with enhanced debugging and retry logic.

        Args:
            retries (int): Number of retry attempts in case of failure.
            delay (int): Delay in seconds between retries.

        Returns:
            Dict[str, str]: Loaded dynamic chatflows.
        """
        dynamic_chatflows = {}
        try:
            self.log_debug(
                "[load_dynamic_chatflows] Starting dynamic chatflow retrieval."
            )

            endpoint = (
                f"{self.valves.flowise_api_endpoint.rstrip('/')}/api/v1/chatflows"
            )
            headers = {"Authorization": f"Bearer {self.valves.flowise_api_key}"}

            self.log_debug(f"[load_dynamic_chatflows] Endpoint: {endpoint}")
            self.log_debug(f"[load_dynamic_chatflows] Headers: {headers}")

            for attempt in range(1, retries + 1):
                try:
                    self.log_debug(
                        f"[load_dynamic_chatflows] Attempt {attempt} to retrieve chatflows."
                    )
                    response = requests.get(
                        endpoint,
                        headers=headers,
                        timeout=self.valves.chatflow_load_timeout,
                    )
                    self.log_debug(
                        f"[load_dynamic_chatflows] Response status: {response.status_code}"
                    )
                    raw_response = response.text
                    self.log_debug(
                        f"[load_dynamic_chatflows] Raw response: {raw_response}"
                    )

                    if response.status_code != 200:
                        self.log_error(
                            f"[load_dynamic_chatflows] API call failed with status: {response.status_code}."
                        )
                        raise ValueError(f"HTTP {response.status_code}: {raw_response}")

                    data = json.loads(raw_response)
                    self.log_debug(f"[load_dynamic_chatflows] Parsed data: {data}")

                    for flow in data:
                        name = flow.get("name", "").strip()
                        cid = flow.get("id", "").strip()

                        self.log_debug(
                            f"[load_dynamic_chatflows] Processing flow: name='{name}', id='{cid}'"
                        )

                        if not name or not cid:
                            self.log_debug(
                                f"[load_dynamic_chatflows] Skipping invalid entry: {flow}"
                            )
                            continue

                        # Apply blacklist regex
                        if re.search(
                            self.valves.chatflow_blacklist, name, re.IGNORECASE
                        ):
                            self.log_debug(
                                f"[load_dynamic_chatflows] Chatflow '{name}' is blacklisted. Skipping."
                            )
                            continue

                        # Sanitize name
                        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                        self.log_debug(
                            f"[load_dynamic_chatflows] Sanitized name: '{sanitized_name}'"
                        )

                        if sanitized_name in dynamic_chatflows:
                            base_name = sanitized_name
                            suffix = 1
                            while sanitized_name in dynamic_chatflows:
                                sanitized_name = f"{base_name}_{suffix}"
                                suffix += 1
                            self.log_debug(
                                f"[load_dynamic_chatflows] Resolved duplicate name: '{name}' -> '{sanitized_name}'"
                            )

                        dynamic_chatflows[sanitized_name] = cid
                        self.log_debug(
                            f"[load_dynamic_chatflows] Added dynamic chatflow: '{sanitized_name}': '{cid}'"
                        )

                    self.log_debug(
                        f"[load_dynamic_chatflows] Successfully loaded dynamic chatflows: {dynamic_chatflows}"
                    )
                    return dynamic_chatflows  # Exit successfully after loading

                except (requests.RequestException, ValueError) as e:
                    self.log_error(
                        f"[load_dynamic_chatflows] Attempt {attempt} failed: {e}",
                        exc_info=True,
                    )
                    if attempt < retries:
                        self.log_debug(
                            f"[load_dynamic_chatflows] Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        self.log_error(
                            "[load_dynamic_chatflows] All retry attempts failed."
                        )
                        self.flowise_available = False
                except Exception as e:
                    self.log_error(
                        f"[load_dynamic_chatflows] Unexpected error: {e}", exc_info=True
                    )
                    self.flowise_available = False
                    break
        except Exception as e:
            self.log_error(
                f"[load_dynamic_chatflows] Fatal error during dynamic chatflow retrieval: {e}",
                exc_info=True,
            )
        finally:
            self.log_debug(
                "[load_dynamic_chatflows] Completed dynamic chatflow retrieval process."
            )

        return dynamic_chatflows

    def load_dynamic_assistants(
        self, retries: int = 3, delay: int = 5
    ) -> Dict[str, str]:
        """
        Load assistants dynamically using the Flowise API, with enhanced debugging and retry logic.

        Args:
            retries (int): Number of retry attempts in case of failure.
            delay (int): Delay in seconds between retries.

        Returns:
            Dict[str, str]: Loaded dynamic assistants.
        """
        dynamic_assistants = {}
        try:
            self.log_debug(
                "[load_dynamic_assistants] Starting dynamic assistant retrieval."
            )

            endpoint = (
                f"{self.valves.flowise_api_endpoint.rstrip('/')}/api/v1/assistants"
            )
            headers = {"Authorization": f"Bearer {self.valves.flowise_api_key}"}

            self.log_debug(f"[load_dynamic_assistants] Endpoint: {endpoint}")
            self.log_debug(f"[load_dynamic_assistants] Headers: {headers}")

            for attempt in range(1, retries + 1):
                try:
                    self.log_debug(
                        f"[load_dynamic_assistants] Attempt {attempt} to retrieve assistants."
                    )
                    response = requests.get(
                        endpoint,
                        headers=headers,
                        timeout=self.valves.chatflow_load_timeout,  # Reuse chatflow_load_timeout
                    )
                    self.log_debug(
                        f"[load_dynamic_assistants] Response status: {response.status_code}"
                    )
                    raw_response = response.text
                    self.log_debug(
                        f"[load_dynamic_assistants] Raw response: {raw_response}"
                    )

                    if response.status_code != 200:
                        self.log_error(
                            f"[load_dynamic_assistants] API call failed with status: {response.status_code}."
                        )
                        raise ValueError(f"HTTP {response.status_code}: {raw_response}")

                    data = json.loads(raw_response)
                    self.log_debug(f"[load_dynamic_assistants] Parsed data: {data}")

                    for assistant in data:
                        details = json.loads(assistant.get("details", "{}"))
                        name = details.get("name", "").strip()
                        aid = assistant.get("id", "").strip()

                        self.log_debug(
                            f"[load_dynamic_assistants] Processing assistant: name='{name}', id='{aid}'"
                        )

                        if not name or not aid:
                            self.log_debug(
                                f"[load_dynamic_assistants] Skipping invalid entry: {assistant}"
                            )
                            continue

                        # Apply assistant blacklist regex
                        if re.search(
                            self.valves.assistant_blacklist, name, re.IGNORECASE
                        ):
                            self.log_debug(
                                f"[load_dynamic_assistants] Assistant '{name}' is blacklisted. Skipping."
                            )
                            continue

                        # Sanitize name
                        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                        self.log_debug(
                            f"[load_dynamic_assistants] Sanitized name: '{sanitized_name}'"
                        )

                        if sanitized_name in dynamic_assistants:
                            base_name = sanitized_name
                            suffix = 1
                            while sanitized_name in dynamic_assistants:
                                sanitized_name = f"{base_name}_{suffix}"
                                suffix += 1
                            self.log_debug(
                                f"[load_dynamic_assistants] Resolved duplicate name: '{name}' -> '{sanitized_name}'"
                            )

                        dynamic_assistants[sanitized_name] = aid
                        self.log_debug(
                            f"[load_dynamic_assistants] Added dynamic assistant: '{sanitized_name}': '{aid}'"
                        )

                    self.log_debug(
                        f"[load_dynamic_assistants] Successfully loaded dynamic assistants: {dynamic_assistants}"
                    )
                    return dynamic_assistants  # Exit successfully after loading

                except (requests.RequestException, ValueError) as e:
                    self.log_error(
                        f"[load_dynamic_assistants] Attempt {attempt} failed: {e}",
                        exc_info=True,
                    )
                    if attempt < retries:
                        self.log_debug(
                            f"[load_dynamic_assistants] Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        self.log_error(
                            "[load_dynamic_assistants] All retry attempts failed."
                        )
                        self.flowise_available = False
                except Exception as e:
                    self.log_error(
                        f"[load_dynamic_assistants] Unexpected error: {e}",
                        exc_info=True,
                    )
                    self.flowise_available = False
                    break
        except Exception as e:
            self.log_error(
                f"[load_dynamic_assistants] Fatal error during dynamic assistant retrieval: {e}",
                exc_info=True,
            )
        finally:
            self.log_debug(
                "[load_dynamic_assistants] Completed dynamic assistant retrieval process."
            )

        return dynamic_assistants

    def load_static_chatflows(self) -> Dict[str, str]:
        """
        Synchronously loads static chatflows and updates self.chatflows.

        Returns:
            Dict[str, str]: Loaded static chatflows.
        """
        static_chatflows = {}
        try:
            self.log_debug(
                "[load_static_chatflows] Starting static chatflow retrieval."
            )

            pairs = [
                pair.strip()
                for pair in self.valves.flowise_chatflow_ids.split(",")
                if ":" in pair
            ]
            self.log_debug(f"[load_static_chatflows] Extracted pairs: {pairs}")

            for pair in pairs:
                try:
                    name, flow_id = map(str.strip, pair.split(":", 1))
                    self.log_debug(
                        f"[load_static_chatflows] Processing pair: name='{name}', id='{flow_id}'"
                    )

                    if not name or not flow_id:
                        self.log_debug(
                            f"[load_static_chatflows] Skipping invalid pair: '{pair}'"
                        )
                        continue

                    # Validate flow_id
                    if not re.match(r"^[a-zA-Z0-9_-]+$", flow_id):
                        self.log_debug(
                            f"[load_static_chatflows] Invalid flow_id '{flow_id}' for pair '{pair}'. Skipping."
                        )
                        continue

                    # Sanitize name
                    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                    self.log_debug(
                        f"[load_static_chatflows] Sanitized name: '{sanitized_name}'"
                    )

                    if sanitized_name in static_chatflows:
                        base_name = sanitized_name
                        suffix = 1
                        while sanitized_name in static_chatflows:
                            sanitized_name = f"{base_name}_{suffix}"
                            suffix += 1
                        self.log_debug(
                            f"[load_static_chatflows] Resolved duplicate name: '{name}' -> '{sanitized_name}'"
                        )

                    static_chatflows[sanitized_name] = flow_id
                    self.log_debug(
                        f"[load_static_chatflows] Added static chatflow: '{sanitized_name}': '{flow_id}'"
                    )

                except ValueError as ve:
                    self.log_error(
                        f"[load_static_chatflows] Error parsing pair '{pair}': {ve}",
                        exc_info=True,
                    )
                except Exception as e:
                    self.log_error(
                        f"[load_static_chatflows] Unexpected error processing pair '{pair}': {e}",
                        exc_info=True,
                    )

            self.log_debug(
                f"[load_static_chatflows] Successfully loaded static chatflows: {static_chatflows}"
            )
            return static_chatflows

        except Exception as e:
            self.log_error(
                f"[load_static_chatflows] Unexpected error during static chatflow loading: {e}",
                exc_info=True,
            )
            return static_chatflows
        finally:
            self.log_debug(
                "[load_static_chatflows] Finished static chatflow retrieval."
            )

    # NEW: Load static assistants
    def load_static_assistants(self) -> Dict[str, str]:
        """
        Synchronously loads static assistants and updates self.chatflows.

        Returns:
            Dict[str, str]: Loaded static assistants.
        """
        static_assistants = {}
        try:
            self.log_debug(
                "[load_static_assistants] Starting static assistant retrieval."
            )

            pairs = [
                pair.strip()
                for pair in self.valves.flowise_assistant_ids.split(",")
                if ":" in pair
            ]
            self.log_debug(f"[load_static_assistants] Extracted pairs: {pairs}")

            for pair in pairs:
                try:
                    name, assistant_id = map(str.strip, pair.split(":", 1))
                    self.log_debug(
                        f"[load_static_assistants] Processing pair: name='{name}', id='{assistant_id}'"
                    )

                    if not name or not assistant_id:
                        self.log_debug(
                            f"[load_static_assistants] Skipping invalid pair: '{pair}'"
                        )
                        continue

                    # Validate assistant_id
                    if not re.match(r"^[a-zA-Z0-9_-]+$", assistant_id):
                        self.log_debug(
                            f"[load_static_assistants] Invalid assistant_id '{assistant_id}' for pair '{pair}'. Skipping."
                        )
                        continue

                    # Apply assistant blacklist regex
                    if re.search(self.valves.assistant_blacklist, name, re.IGNORECASE):
                        self.log_debug(
                            f"[load_static_assistants] Assistant '{name}' is blacklisted. Skipping."
                        )
                        continue

                    # Sanitize name
                    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                    self.log_debug(
                        f"[load_static_assistants] Sanitized name: '{sanitized_name}'"
                    )

                    if sanitized_name in static_assistants:
                        base_name = sanitized_name
                        suffix = 1
                        while sanitized_name in static_assistants:
                            sanitized_name = f"{base_name}_{suffix}"
                            suffix += 1
                        self.log_debug(
                            f"[load_static_assistants] Resolved duplicate name: '{name}' -> '{sanitized_name}'"
                        )

                    static_assistants[sanitized_name] = assistant_id
                    self.log_debug(
                        f"[load_static_assistants] Added static assistant: '{sanitized_name}': '{assistant_id}'"
                    )

                except ValueError as ve:
                    self.log_error(
                        f"[load_static_assistants] Error parsing pair '{pair}': {ve}",
                        exc_info=True,
                    )
                except Exception as e:
                    self.log_error(
                        f"[load_static_assistants] Unexpected error processing pair '{pair}': {e}",
                        exc_info=True,
                    )

            self.log_debug(
                f"[load_static_assistants] Successfully loaded static assistants: {static_assistants}"
            )
            return static_assistants

        except Exception as e:
            self.log_error(
                f"[load_static_assistants] Unexpected error during static assistant loading: {e}",
                exc_info=True,
            )
            return static_assistants
        finally:
            self.log_debug(
                "[load_static_assistants] Finished static assistant retrieval."
            )

    def load_dynamic_assistants(
        self, retries: int = 3, delay: int = 5
    ) -> Dict[str, str]:
        """
        Load assistants dynamically using the Flowise API, with enhanced debugging and retry logic.

        Args:
            retries (int): Number of retry attempts in case of failure.
            delay (int): Delay in seconds between retries.

        Returns:
            Dict[str, str]: Loaded dynamic assistants.
        """
        dynamic_assistants = {}
        try:
            self.log_debug(
                "[load_dynamic_assistants] Starting dynamic assistant retrieval."
            )

            endpoint = (
                f"{self.valves.flowise_api_endpoint.rstrip('/')}/api/v1/assistants"
            )
            headers = {"Authorization": f"Bearer {self.valves.flowise_api_key}"}

            self.log_debug(f"[load_dynamic_assistants] Endpoint: {endpoint}")
            self.log_debug(f"[load_dynamic_assistants] Headers: {headers}")

            for attempt in range(1, retries + 1):
                try:
                    self.log_debug(
                        f"[load_dynamic_assistants] Attempt {attempt} to retrieve assistants."
                    )
                    response = requests.get(
                        endpoint,
                        headers=headers,
                        timeout=self.valves.chatflow_load_timeout,  # Reuse chatflow_load_timeout
                    )
                    self.log_debug(
                        f"[load_dynamic_assistants] Response status: {response.status_code}"
                    )
                    raw_response = response.text
                    self.log_debug(
                        f"[load_dynamic_assistants] Raw response: {raw_response}"
                    )

                    if response.status_code != 200:
                        self.log_error(
                            f"[load_dynamic_assistants] API call failed with status: {response.status_code}."
                        )
                        raise ValueError(f"HTTP {response.status_code}: {raw_response}")

                    data = json.loads(raw_response)
                    self.log_debug(f"[load_dynamic_assistants] Parsed data: {data}")

                    for assistant in data:
                        details = json.loads(assistant.get("details", "{}"))
                        name = details.get("name", "").strip()
                        aid = assistant.get("id", "").strip()

                        self.log_debug(
                            f"[load_dynamic_assistants] Processing assistant: name='{name}', id='{aid}'"
                        )

                        if not name or not aid:
                            self.log_debug(
                                f"[load_dynamic_assistants] Skipping invalid entry: {assistant}"
                            )
                            continue

                        # Apply assistant blacklist regex
                        if re.search(
                            self.valves.assistant_blacklist, name, re.IGNORECASE
                        ):
                            self.log_debug(
                                f"[load_dynamic_assistants] Assistant '{name}' is blacklisted. Skipping."
                            )
                            continue

                        # Sanitize name
                        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                        self.log_debug(
                            f"[load_dynamic_assistants] Sanitized name: '{sanitized_name}'"
                        )

                        if sanitized_name in dynamic_assistants:
                            base_name = sanitized_name
                            suffix = 1
                            while sanitized_name in dynamic_assistants:
                                sanitized_name = f"{base_name}_{suffix}"
                                suffix += 1
                            self.log_debug(
                                f"[load_dynamic_assistants] Resolved duplicate name: '{name}' -> '{sanitized_name}'"
                            )

                        dynamic_assistants[sanitized_name] = aid
                        self.log_debug(
                            f"[load_dynamic_assistants] Added dynamic assistant: '{sanitized_name}': '{aid}'"
                        )

                    self.log_debug(
                        f"[load_dynamic_assistants] Successfully loaded dynamic assistants: {dynamic_assistants}"
                    )
                    return dynamic_assistants  # Exit successfully after loading

                except (requests.RequestException, ValueError) as e:
                    self.log_error(
                        f"[load_dynamic_assistants] Attempt {attempt} failed: {e}",
                        exc_info=True,
                    )
                    if attempt < retries:
                        self.log_debug(
                            f"[load_dynamic_assistants] Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        self.log_error(
                            "[load_dynamic_assistants] All retry attempts failed."
                        )
                        self.flowise_available = False
                except Exception as e:
                    self.log_error(
                        f"[load_dynamic_assistants] Unexpected error: {e}",
                        exc_info=True,
                    )
                    self.flowise_available = False
                    break
        except Exception as e:
            self.log_error(
                f"[load_dynamic_assistants] Fatal error during dynamic assistant retrieval: {e}",
                exc_info=True,
            )
        finally:
            self.log_debug(
                "[load_dynamic_assistants] Completed dynamic assistant retrieval process."
            )

        return dynamic_assistants

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

    def get_chatflows(self) -> Dict[str, str]:
        """
        Retrieve all available chatflows and assistants.

        Returns:
            Dict[str, str]: A dictionary mapping model names to IDs.
        """
        self.log_debug("[get_chatflows] Retrieving all available models.")
        return self.chatflows.copy()

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
        # Reset state for the new request
        self.reset_state()
        self.request_id = str(time.time())
        self.log_debug(f"[pipe] Starting new request with ID: {self.request_id}")

        output = {}  # Initialize output to ensure it's defined

        try:
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

            system_message, user_messages = pop_system_message(messages)
            self.log_debug(f"[pipe] System message: {system_message}")
            self.log_debug(
                f"[pipe] User messages after popping system message: {user_messages}"
            )

            combined_prompt = self._get_combined_prompt(user_messages)
            self.log_debug(
                f"[pipe] Combined prompt for model '{model_name}':\n{combined_prompt}"
            )

            # Emit initial status: Processing the request
            if __event_emitter__:
                self.emit_status_sync(
                    __event_emitter__, "Processing your request", done=False
                )
                self.log_debug(
                    "[pipe] Initial status 'Processing your request' emitted."
                )

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
                combined_prompt_with_summary = (
                    f"{combined_prompt}\n\nSummary:\n{summary}"
                )
                self.log_debug(
                    f"[pipe] Combined prompt with summary:\n{combined_prompt_with_summary}"
                )
            else:
                combined_prompt_with_summary = combined_prompt

            # Handle Flowise request
            output = self.handle_flowise_request(
                question=combined_prompt_with_summary,
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
                    static_message = self.valves.static_status_messages[
                        0
                    ]  # Example: use the first static message
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
        finally:
            return output

    def emit_status_sync(
        self,
        __event_emitter__: Callable[[dict], Any],
        message: str,
        done: bool,
    ):
        """Emit status updates to the event emitter synchronously."""
        if __event_emitter__:
            event = {
                "type": "status",
                "data": {"description": message, "done": done},
            }
            self.log_debug(
                f"[emit_status_sync] Preparing to emit status event: {event}"
            )

            try:
                if asyncio.iscoroutinefunction(__event_emitter__):
                    self.log_debug(
                        "[emit_status_sync] Detected asynchronous event emitter."
                    )
                    asyncio.create_task(__event_emitter__(event))
                else:
                    self.log_debug(
                        "[emit_status_sync] Detected synchronous event emitter."
                    )
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
            self.log_debug(
                f"[emit_output_sync] Preparing to emit message event: {message_event}"
            )

            try:
                if asyncio.iscoroutinefunction(__event_emitter__):
                    self.log_debug(
                        "[emit_output_sync] Detected asynchronous event emitter."
                    )
                    asyncio.create_task(__event_emitter__(message_event))
                else:
                    self.log_debug(
                        "[emit_output_sync] Detected synchronous event emitter."
                    )
                    __event_emitter__(message_event)

                self.log_debug("[emit_output_sync] Message event emitted successfully.")
            except Exception as e:
                self.log_error(
                    f"[emit_output_sync] Error emitting message event: {e}",
                    exc_info=True,
                )
            finally:
                self.log_debug("[emit_output_sync] Finished emitting message event.")

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
                self.log_error(
                    "[generate_summary] Summarization model ID not configured."
                )
                return None

            # Collect chat history
            user_id = (
                __user__.get("user_id", "default_user") if __user__ else "default_user"
            )
            chat_session = self.chat_sessions.get(user_id, {})
            history = chat_session.get("history", [])
            if not history:
                self.log_debug(
                    "[generate_summary] No chat history available for summarization."
                )
                return None

            user_messages = [msg["content"] for msg in history if msg["role"] == "user"]
            assistant_messages = [
                msg["content"] for msg in history if msg["role"] == "assistant"
            ]

            user_content = "\n".join(user_messages)
            assistant_content = "\n".join(assistant_messages)

            prompt = (
                self.valves.summarization_system_prompt
                + f"\n\nUser Messages:\n{user_content}\n\nAssistant Messages:\n{assistant_content}"
            )
            self.log_debug(
                f"[generate_summary] Generating summary with prompt: {prompt}"
            )

            payload = {
                "model": self.valves.summarization_model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": self.valves.summarization_system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"User Messages:\n{user_content}\nAssistant Messages:\n{assistant_content}",
                    },
                ],
                "max_tokens": 150,
                "temperature": 0.5,
            }

            response = generate_chat_completions(form_data=payload)
            summary = response["choices"][0]["message"]["content"].strip()
            self.log_debug(f"[generate_summary] Generated summary: {summary}")
            return summary

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
                self.log_debug(
                    "[generate_status_update] LLM status updates are disabled."
                )
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

            payload = {
                "model": self.valves.llm_status_update_model_id,
                "messages": [
                    {"role": "system", "content": self.valves.llm_status_update_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 50,
                "temperature": 0.7,
            }

            response = generate_chat_completions(form_data=payload)
            status_update = response["choices"][0]["message"]["content"].strip()
            self.log_debug(
                f"[generate_status_update] Generated status update: {status_update}"
            )
            return status_update

        except Exception as e:
            self.log_error(
                f"[generate_status_update] Error generating status update: {e}",
                exc_info=True,
            )
            return None

        finally:
            self.log_debug(
                "[generate_status_update] Finished generating status update."
            )

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
        try:
            # Prepare payload
            payload = {"question": question}
            self.log_debug(f"[handle_flowise_request] Initial payload: {payload}")

            user_id = (
                __user__.get("user_id", "default_user") if __user__ else "default_user"
            )
            self.log_debug(f"[handle_flowise_request] User ID: {user_id}")

            chat_session = self.chat_sessions.get(user_id, {})
            self.log_debug(
                f"[handle_flowise_request] Current chat session: {chat_session}"
            )

            chat_id = chat_session.get("chat_id")
            if chat_id:
                payload["chatId"] = chat_id
                self.log_debug(
                    f"[handle_flowise_request] Added chatId to payload: {chat_id}"
                )

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

            try:
                data = json.loads(response_text)
                self.log_debug(f"[handle_flowise_request] Parsed data: {data!r}")
            except json.JSONDecodeError:
                error_message = "Error: Invalid JSON response from Flowise."
                self.log_debug(f"[handle_flowise_request] {error_message}")
                return {"error": error_message}

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
            if user_id not in self.chat_sessions:
                self.chat_sessions[user_id] = {"chat_id": None, "history": []}
                self.log_debug(
                    f"[handle_flowise_request] Created new chat session for user '{user_id}'."
                )

            self.chat_sessions[user_id]["chat_id"] = new_chat_id
            self.chat_sessions[user_id]["history"].append(
                {"role": "assistant", "content": text}
            )
            self.log_debug(
                f"[handle_flowise_request] Updated history for user '{user_id}': {self.chat_sessions[user_id]['history']}"
            )

            # Emit the Flowise response via the event emitter
            if __event_emitter__:
                self.log_debug(
                    f"[handle_flowise_request] Emitting Flowise response via emit_output_sync."
                )
                self.emit_output_sync(
                    __event_emitter__, text, include_collapsible=False
                )

            return {"response": text}

        except requests.exceptions.RequestException as e:
            # Handle any other request exceptions (e.g., connection errors)
            error_message = f"Request failed: {str(e)}"
            self.log_debug(f"[handle_flowise_request] {error_message}")

            return {"error": error_message}
        except Exception as e:
            error_message = f"Error during Flowise request handling: {e}"
            self.log_error(f"[handle_flowise_request] {error_message}", exc_info=True)
            return {"error": error_message}
        # Removed the finally block with return statement to prevent undefined variable issues

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
        prompt_parts = [
            f"{message.get('role', 'user').capitalize()}: {message.get('content', '')}"
            for message in messages
        ]
        combined_prompt = "\n".join(prompt_parts)
        self.log_debug(f"[get_combined_prompt] Combined prompt:\n{combined_prompt}")
        return combined_prompt

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

    def format_elapsed_time(self, elapsed: float) -> str:
        """
        Format elapsed time into a readable string.

        Args:
            elapsed (float): Elapsed time in seconds.

        Returns:
            str: Formatted time string.
        """
        try:
            minutes, seconds = divmod(int(elapsed), 60)
            if minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except Exception as e:
            self.log_error(f"[format_elapsed_time] Error: {e}", exc_info=True)
            return "Unknown time"
        finally:
            self.log_debug("[format_elapsed_time] Finished formatting elapsed time.")
