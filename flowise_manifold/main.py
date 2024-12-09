import json
import time
import re
import logging
from typing import List, Optional, Callable, Dict, Any, Union

from pydantic import BaseModel, Field
from dataclasses import dataclass

import aiohttp
import asyncio

# Mock functions for demonstration (Replace with actual implementations)
async def generate_chat_completions(form_data):
    # Mock implementation
    await asyncio.sleep(0.1)  # Simulate network delay
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
MANIFOLD_PREFIX_DEFAULT = "flowise/"

class Pipe:
    """
    The Pipe class manages interactions with Flowise APIs, including dynamic and static chatflow retrieval,
    model registration with a manifold prefix, and handling requests asynchronously with internal retry logic.
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
            description="API key for Flowise. Required for dynamic chatflow retrieval.",
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
        manifold_prefix: str = Field(
            default=MANIFOLD_PREFIX_DEFAULT,
            description="Prefix used for Flowise models.",
        )

        chatflow_blacklist: str = Field(
            default="wip|dev|offline|broken",
            description="Regex to exclude certain chatflows by name.",
        )

        # Summarization Configuration (Disabled as per requirements)
        enable_summarization: bool = Field(
            default=False,
            description="Enable chat history summarization.",
        )
        summarisation_model_id: str = Field(
            default="",
            description="Model ID for summarisation tasks.",
        )
        summarisation_system_prompt: str = Field(
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
                "Just a sec..."
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
        enable_debug: bool = Field(
            default=True,
            description="Enable or disable debug logging.",
        )
        chatflow_load_timeout: int = Field(
            default=15,
            description="Timeout in seconds for loading chatflows.",
        )
        pipe_processing_timeout: int = Field(
            default=15,
            description="Max time in seconds for processing the pipe method.",
        )

    def __init__(self, valves: Optional['Pipe.Valves'] = None):
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

        # Initialize other attributes
        self.chat_sessions: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.flowise_available: bool = True
        self.chatflows: Dict[str, str] = {}

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

        # Check configurations
        self.log_debug("[INIT] Checking configurations for chatflows.")
        dynamic_ok = self.is_dynamic_config_ok()
        static_ok = self.is_static_config_ok()

        # Load chatflows asynchronously during initialization
        try:
            asyncio.create_task(self.load_chatflows(dynamic_ok, static_ok))
            self.log_debug("[INIT] Scheduled chatflow loading task.")
        except Exception as e:
            self.log_error(f"[INIT] Failed to schedule chatflow loading: {e}", exc_info=True)

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
        if not self.valves.use_dynamic_chatflows:
            self.log_debug("[is_dynamic_config_ok] Dynamic chatflows disabled.")
            return False
        if not self.valves.flowise_api_key or self.valves.flowise_api_key == FLOWISE_API_KEY_PLACEHOLDER:
            self.log_error("[is_dynamic_config_ok] Dynamic chatflows enabled but API key missing/placeholder.")
            return False
        self.log_debug("[is_dynamic_config_ok] Dynamic chatflow configuration is valid.")
        return True

    def is_static_config_ok(self) -> bool:
        """Check if static chatflow configuration is valid."""
        if not self.valves.use_static_chatflows:
            self.log_debug("[is_static_config_ok] Static chatflows disabled.")
            return False
        if not self.valves.flowise_chatflow_ids or self.valves.flowise_chatflow_ids == FLOWISE_CHATFLOW_IDS_PLACEHOLDER:
            self.log_error("[is_static_config_ok] Static chatflows enabled but config empty/placeholder.")
            return False
        self.log_debug("[is_static_config_ok] Static chatflow configuration is valid.")
        return True

    async def load_chatflows(self, dynamic_ok: bool, static_ok: bool):
        """
        Asynchronously load both dynamic and static chatflows based on configuration.
        """
        try:
            # Load dynamic chatflows if enabled and configured
            if self.valves.use_dynamic_chatflows and dynamic_ok:
                self.log_debug("[load_chatflows] Dynamic chatflows enabled. Loading now.")
                await self.load_dynamic_chatflows()
            else:
                if self.valves.use_dynamic_chatflows:
                    self.log_debug("[load_chatflows] Dynamic chatflows enabled but configuration invalid. Skipping dynamic loading.")

            # Load static chatflows if enabled and configured
            if self.valves.use_static_chatflows and static_ok:
                self.log_debug("[load_chatflows] Static chatflows enabled. Loading now.")
                self.load_static_chatflows()
            else:
                if self.valves.use_static_chatflows:
                    self.log_debug("[load_chatflows] Static chatflows enabled but configuration invalid. Skipping static loading.")

            # If neither dynamic nor static is valid, no chatflows will be loaded
            if not (self.valves.use_dynamic_chatflows and dynamic_ok) and not (self.valves.use_static_chatflows and static_ok):
                self.log_debug("[load_chatflows] Neither dynamic nor static config is valid. No chatflows loaded.")

        except Exception as e:
            self.log_error(f"[load_chatflows] Unexpected error during chatflow loading: {e}", exc_info=True)

    async def load_dynamic_chatflows(self, retries: int = 3, delay: int = 5):
        """
        Load chatflows dynamically using the Flowise API, with enhanced debugging and retry logic.

        Args:
            retries (int): Number of retry attempts in case of failure.
            delay (int): Delay in seconds between retries.

        Returns:
            None
        """
        try:
            self.log_debug("[load_dynamic_chatflows] Starting dynamic chatflow retrieval.")

            if not self.is_dynamic_config_ok():
                self.log_error("[load_dynamic_chatflows] Dynamic chatflow retrieval skipped: Invalid configuration.")
                return

            endpoint = f"{self.valves.flowise_api_endpoint.rstrip('/')}/api/v1/chatflows"
            headers = {"Authorization": f"Bearer {self.valves.flowise_api_key}"}

            self.log_debug(f"[load_dynamic_chatflows] Endpoint: {endpoint}")
            self.log_debug(f"[load_dynamic_chatflows] Headers: {headers}")

            for attempt in range(1, retries + 1):
                try:
                    self.log_debug(f"[load_dynamic_chatflows] Attempt {attempt} to retrieve chatflows.")

                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.valves.chatflow_load_timeout)) as session:
                        async with session.get(endpoint, headers=headers) as response:
                            self.log_debug(f"[load_dynamic_chatflows] Response status: {response.status}")
                            raw_response = await response.text()
                            self.log_debug(f"[load_dynamic_chatflows] Raw response: {raw_response}")

                            if response.status != 200:
                                self.log_error(f"[load_dynamic_chatflows] API call failed with status: {response.status}.")
                                raise ValueError(f"HTTP {response.status}: {raw_response}")

                            data = json.loads(raw_response)
                            chatflows = {}
                            for flow in data:
                                name = flow.get("name", "").strip()
                                cid = flow.get("id", "").strip()

                                if not name or not cid:
                                    self.log_debug(f"[load_dynamic_chatflows] Skipping invalid entry: {flow}")
                                    continue

                                # Apply blacklist regex
                                if re.search(self.valves.chatflow_blacklist, name, re.IGNORECASE):
                                    self.log_debug(f"[load_dynamic_chatflows] Chatflow '{name}' is blacklisted. Skipping.")
                                    continue

                                sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                                if sanitized_name in chatflows:
                                    base_name = sanitized_name
                                    suffix = 1
                                    while sanitized_name in chatflows:
                                        sanitized_name = f"{base_name}_{suffix}"
                                        suffix += 1

                                chatflows[sanitized_name] = cid
                                self.log_debug(f"[load_dynamic_chatflows] Added: {sanitized_name} -> {cid}")

                            self.chatflows.update(chatflows)
                            self.log_debug(f"[load_dynamic_chatflows] Successfully loaded dynamic chatflows: {self.chatflows}")
                            self.log_debug(f"[load_dynamic_chatflows] Total loaded chatflows: {len(self.chatflows)}")
                            return  # Exit successfully after loading

                except (aiohttp.ClientError, ValueError) as e:
                    self.log_error(f"[load_dynamic_chatflows] Attempt {attempt} failed: {e}", exc_info=True)
                    if attempt < retries:
                        self.log_debug(f"[load_dynamic_chatflows] Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        self.log_error("[load_dynamic_chatflows] All retry attempts failed.")
                        self.flowise_available = False
                except Exception as e:
                    self.log_error(f"[load_dynamic_chatflows] Unexpected error: {e}", exc_info=True)
                    self.flowise_available = False
                    break
        except Exception as e:
            self.log_error(f"[load_dynamic_chatflows] Fatal error during dynamic chatflow retrieval: {e}", exc_info=True)
        finally:
            self.log_debug("[load_dynamic_chatflows] Completed dynamic chatflow retrieval process.")

    def load_static_chatflows(self):
        """
        Synchronously loads static chatflows and updates self.chatflows.
        """
        try:
            self.log_debug("[load_static_chatflows] Starting static chatflow retrieval.")
            if not self.is_static_config_ok():
                self.log_error("[load_static_chatflows] Static chatflow retrieval skipped: Invalid configuration.")
                return

            chatflows = self.parse_static_chatflows()
            if chatflows:
                self.chatflows.update(chatflows)
                self.log_debug(f"[load_static_chatflows] Successfully loaded static chatflows: {chatflows}")
                self.log_debug(f"[load_static_chatflows] Total loaded chatflows: {len(self.chatflows)}")
            else:
                self.log_error("[load_static_chatflows] No valid static chatflows found.")

        except Exception as e:
            self.log_error(f"[load_static_chatflows] Unexpected error during static chatflow loading: {e}", exc_info=True)
        finally:
            self.log_debug("[load_static_chatflows] Finished static chatflow retrieval.")

    def parse_static_chatflows(self) -> Dict[str, str]:
        """
        Parse static chatflows from a comma-separated list of 'Name:ID' pairs.

        Validates and sanitizes each pair and logs detailed information about the parsing process.

        Returns:
            Dict[str, str]: A dictionary mapping sanitized chatflow names to their IDs.
        """
        try:
            self.log_debug("[parse_static_chatflows] Parsing static chatflows.")
            chatflows = {}
            pairs = [pair.strip() for pair in self.valves.flowise_chatflow_ids.split(",") if ":" in pair]
            self.log_debug(f"[parse_static_chatflows] Extracted pairs: {pairs}")

            for pair in pairs:
                try:
                    name, flow_id = map(str.strip, pair.split(":", 1))

                    if not name or not flow_id:
                        self.log_debug(f"[parse_static_chatflows] Skipping invalid pair: {pair}")
                        continue

                    # Validate flow_id
                    if not re.match(r"^[a-zA-Z0-9_-]+$", flow_id):
                        self.log_debug(f"[parse_static_chatflows] Invalid flow_id '{flow_id}' for pair '{pair}'. Skipping.")
                        continue

                    # Sanitize name
                    original_name = name
                    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

                    if name in chatflows:
                        base_name = name
                        suffix = 1
                        while name in chatflows:
                            name = f"{base_name}_{suffix}"
                            suffix += 1
                        self.log_debug(f"[parse_static_chatflows] Resolved duplicate name: {original_name} -> {name}")

                    chatflows[name] = flow_id
                    self.log_debug(f"[parse_static_chatflows] Added static chatflow: {name}: {flow_id}")

                except ValueError as ve:
                    self.log_error(f"[parse_static_chatflows] Error parsing pair '{pair}': {ve}", exc_info=True)
                except Exception as e:
                    self.log_error(f"[parse_static_chatflows] Unexpected error processing pair '{pair}': {e}", exc_info=True)

            self.log_debug(f"[parse_static_chatflows] Final parsed static chatflows: {chatflows}")
            return chatflows

        except Exception as e:
            self.log_error(f"[parse_static_chatflows] Unexpected error during parsing: {e}", exc_info=True)
            return {}
        finally:
            self.log_debug("[parse_static_chatflows] Finished parsing static chatflows.")

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

    def pipes(self) -> List[dict]:
        """
        Retrieve and register chatflow models, ensuring both dynamic and static chatflows are loaded.

        Implements retry logic for dynamic chatflows and ensures static chatflows are loaded regardless
        of dynamic loading success. Combines both dynamic and static chatflows, giving precedence to dynamic ones.

        Returns:
            List[dict]: List of chatflow models with 'id' and 'name'.
        """
        self.log_debug("[pipes] Starting asynchronous model retrieval with fallback handling.")
        max_retries = 5
        retry_delay = 2  # seconds
        attempts = 0
        dynamic_chatflows = {}
        static_chatflows = {}

        # Attempt to load dynamic chatflows
        if self.valves.use_dynamic_chatflows:
            while attempts < max_retries:
                try:
                    attempts += 1
                    self.log_debug(f"[pipes] Attempt {attempts}/{max_retries} to load dynamic chatflows.")
                    self.log_debug(f"[pipes] Endpoint: {self.valves.flowise_api_endpoint}")
                    self.log_debug(f"[pipes] API Key: {self.valves.flowise_api_key}")

                    self.load_dynamic_chatflows()
                    dynamic_chatflows = self.chatflows.copy()  # Preserve dynamic chatflows
                    if dynamic_chatflows:
                        self.log_debug(f"[pipes] Successfully loaded dynamic chatflows: {dynamic_chatflows}")
                        break  # Exit the retry loop on success
                except Exception as e:
                    self.log_error(f"[pipes] Error during dynamic chatflow loading attempt {attempts}: {e}", exc_info=True)
                    if attempts < max_retries:
                        self.log_debug(f"[pipes] Retrying in {retry_delay}s...")
                        asyncio.sleep(retry_delay)
                    else:
                        self.log_error("[pipes] Failed to load dynamic chatflows after maximum retries.")

        # Attempt to load static chatflows regardless of dynamic loading success
        if self.valves.use_static_chatflows:
            try:
                self.log_debug("[pipes] Attempting to load static chatflows.")
                self.load_static_chatflows()
                static_chatflows = self.chatflows.copy()  # Preserve static chatflows
                if static_chatflows:
                    self.log_debug(f"[pipes] Successfully loaded static chatflows: {static_chatflows}")
            except Exception as e:
                self.log_error(f"[pipes] Error during static chatflow loading: {e}", exc_info=True)

        # Combine dynamic and static chatflows, giving precedence to dynamic chatflows
        combined_chatflows = {**static_chatflows, **dynamic_chatflows}

        # Update self.chatflows to reflect the final state
        self.chatflows = combined_chatflows

        # Return the final list of models or an empty list if no chatflows were loaded
        if self.chatflows:
            models = [{"id": name, "name": name} for name in self.chatflows.keys()]
            self.log_debug(f"[pipes] Successfully retrieved models: {models}")
            return models
        else:
            self.log_error("[pipes] No chatflows loaded. Returning an empty list.")
            return []

    async def handle_flowise_request(
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
        self.log_debug(f"[handle_flowise_request] Handling request for '{chatflow_name}' question: {question!r}")
        try:
            # Prepare payload
            payload = {"question": question}
            user_id = __user__.get("user_id", "default_user") if __user__ else "default_user"
            chat_session = self.chat_sessions.get(user_id, {})
            chat_id = chat_session.get("chat_id")
            if chat_id:
                payload["chatId"] = chat_id

            self.log_debug(f"[handle_flowise_request] Payload: {payload!r}")

            # Determine chatflow to use
            if not chatflow_name or chatflow_name not in self.chatflows:
                if self.chatflows:
                    chatflow_name = list(self.chatflows.keys())[0]
                    self.log_debug(f"[handle_flowise_request] No or invalid chatflow_name provided. Using '{chatflow_name}'.")
                else:
                    error_message = "No chatflows configured."
                    self.log_debug(f"[handle_flowise_request] {error_message}")
                    if __event_emitter__:
                        await self.emit_status(__event_emitter__, error_message, done=True)
                    return {"error": error_message}

            chatflow_id = self.chatflows[chatflow_name]
            endpoint = self.valves.flowise_api_endpoint.rstrip("/")
            url = f"{endpoint}/api/v1/prediction/{chatflow_id}"
            headers = {"Content-Type": "application/json"}

            self.log_debug(f"[handle_flowise_request] Sending to {url}")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.valves.request_timeout)) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    self.log_debug(f"[handle_flowise_request] Response status: {response.status}")
                    self.log_debug(f"[handle_flowise_request] Response text: {response_text!r}")

                    if response.status != 200:
                        error_message = f"Error: Flowise API call failed with status {response.status}"
                        self.log_debug(error_message)
                        return {"error": error_message}

                    try:
                        data = json.loads(response_text)
                        self.log_debug(f"[handle_flowise_request] Parsed data: {data!r}")
                    except json.JSONDecodeError:
                        error_message = "Error: Invalid JSON response from Flowise."
                        self.log_debug(error_message)
                        return {"error": error_message}

                    raw_text = data.get("text", "")
                    text = self.clean_response_text(raw_text)
                    new_chat_id = data.get("chatId", chat_id)

                    if not text:
                        error_message = "Error: Empty response from Flowise."
                        self.log_debug(error_message)
                        return {"error": error_message}

                    # Update chat session
                    if user_id not in self.chat_sessions:
                        self.chat_sessions[user_id] = {"chat_id": None, "history": []}
                    self.chat_sessions[user_id]["chat_id"] = new_chat_id
                    self.chat_sessions[user_id]["history"].append({"role": "assistant", "content": text})
                    self.log_debug(f"[handle_flowise_request] Updated history for '{user_id}': {self.chat_sessions[user_id]['history']}")

                    # Emit the Flowise response via the event emitter
                    if __event_emitter__:
                        response_event = {
                            "type": "message",
                            "data": {"content": text},
                        }
                        self.log_debug(f"[handle_flowise_request] Emitting Flowise response event: {response_event!r}")
                        try:
                            await __event_emitter__(response_event)
                        except Exception as e:
                            self.log_error(f"[handle_flowise_request] Error emitting response event: {e}", exc_info=True)

                    return {"response": text}

        except aiohttp.ClientError as e:
            error_message = f"Error during Flowise request handling: {e}"
            self.log_error(error_message, exc_info=True)
            return {"error": error_message}
        except Exception as e:
            error_message = f"Error during Flowise request handling: {e}"
            self.log_error(error_message, exc_info=True)
            return {"error": error_message}
        finally:
            self.log_debug("[handle_flowise_request] Finished handling Flowise request.")

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> Union[Dict[str, Any], Dict[str, str]]:
        """
        Main handler for processing requests with dynamic and static chatflow support.

        Steps:
        1. Reset state variables for each new request.
        2. Validate selection methods.
        3. Retrieve chatflows with internal retry logic.
        4. Handle system messages.
        5. Prepare payload for the selected chatflow.
        6. Emit initial status: "Processing your request".
        7. Query the selected chatflow.
        8. Emit contribution completion status.
        9. Return the Flowise output or an error message in a standardized format.

        Args:
            body (dict): The incoming request payload.
            __user__ (Optional[dict]): The user information.
            __event_emitter__ (Optional[Callable[[dict], Any]]): The event emitter for sending messages.

        Returns:
            Union[Dict[str, Any], Dict[str, str]]: The Flowise output or an error message.
        """
        # Reset state for the new request
        self.reset_state()
        self.request_id = str(time.time())
        self.log_debug(f"[pipe] Starting new request with ID: {self.request_id}")

        # Check if neither dynamic nor static chatflows are enabled
        if not (self.valves.use_dynamic_chatflows or self.valves.use_static_chatflows):
            advisory_message = (
                "No chatflow methods are enabled.\n"
                "Please enable at least one method:\n"
                "1. Enable dynamic chatflows by providing a valid Flowise API key.\n"
                "2. Enable static chatflows by providing comma-separated 'Name:ID' pairs.\n\n"
                "Additionally, ensure that the Flowise API endpoint is correctly configured. "
                "If Flowise is not hosted on the default host via port 3030, update the 'flowise_api_endpoint' accordingly."
            )
            self.log_error("[pipe] No chatflow methods enabled.", exc_info=False)
            if __event_emitter__:
                await self.emit_status(__event_emitter__, advisory_message, done=True)
            return {"status": "error", "message": advisory_message}

        try:
            # Step 1: Retrieve Chatflows with internal retry logic
            models = await self.pipes()
            if not models:
                error_message = "[pipe] No chatflows loaded."
                self.log_debug(f"[pipe] {error_message}")
                if __event_emitter__:
                    await self.emit_status(__event_emitter__, error_message, done=True)
                return {"status": "error", "message": error_message}

            self.log_debug(f"[pipe] Retrieved chatflows: {models}")

            # Step 2: Handle System Messages
            messages = body.get("messages", [])
            if not messages:
                error_message = "Error: No messages found in the request."
                self.log_debug(f"[pipe] {error_message}")
                if __event_emitter__:
                    await self.emit_status(__event_emitter__, error_message, done=True)
                return {"status": "error", "message": error_message}

            system_message, user_messages = pop_system_message(messages)
            self.log_debug(f"[pipe] System message: {system_message}")
            self.log_debug(f"[pipe] User messages: {user_messages}")

            # Step 3: Prepare the prompt
            combined_prompt = self._get_combined_prompt(user_messages)
            self.log_debug(f"[pipe] Combined prompt for Flowise:\n{combined_prompt}")

            # Step 4: Emit initial status
            await self.emit_status(
                __event_emitter__,
                "Processing your request",
                done=False,
            )
            self.log_debug("[pipe] Initial status 'Processing your request' emitted.")

            # Step 5: Query the selected chatflow
            selected_chatflow = models[0]["id"]  # Selecting the first chatflow
            self.log_debug(f"[pipe] Using chatflow: {selected_chatflow}")
            output = await self.handle_flowise_request(
                combined_prompt,
                __user__,
                __event_emitter__,
                chatflow_name=selected_chatflow
            )
            if "error" in output:
                raise ValueError(f"[pipe] Error querying chatflow: {output['error']}")

            # Step 6: Emit contribution completion status
            contribution_time = time.time() - self.start_time
            contribution_time_str = self.format_elapsed_time(contribution_time)
            await self.emit_status(
                __event_emitter__,
                f"Contribution took {contribution_time_str}",
                done=False,
            )
            self.log_debug(
                f"[pipe] Emitted contribution completion status: Contribution took {contribution_time_str}"
            )

            # Step 7: Return the Flowise output
            if "response" in output:
                self.log_debug("[pipe] Flowise request successful.")
                return {"status": "success", "data": output["response"]}
            else:
                self.log_debug("[pipe] Flowise request failed.")
                return {
                    "status": "error",
                    "message": "Flowise request failed to generate a response.",
                }

        except Exception as e:
            self.log_error(f"[pipe] Unexpected error: {e}", exc_info=True)
            if __event_emitter__:
                await self.emit_status(
                    __event_emitter__,
                    f"Error processing request: {str(e)}",
                    done=True,
                )
            return {"status": "error", "message": str(e)}
        finally:
            self.log_debug("[pipe] Finished processing the request.")

    def reset_state(self):
        """Reset state variables for reuse in new requests."""
        try:
            self.chat_sessions = {}
            self.start_time = time.time()  # Start time of the current pipe execution
            self.log_debug("[reset_state] State variables have been reset.")
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

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Any],
        message: str,
        done: bool,
    ):
        """Emit status updates to the event emitter."""
        if __event_emitter__:
            event = {
                "type": "status",
                "data": {"description": message, "done": done},
            }
            self.log_debug(f"Emitting status event: {event}")
            try:
                await __event_emitter__(event)
                self.log_debug("[emit_status] Status event emitted successfully.")
            except Exception as e:
                self.log_error(f"[emit_status] Error emitting status event: {e}", exc_info=True)
            finally:
                self.log_debug("[emit_status] Finished emitting status event.")
