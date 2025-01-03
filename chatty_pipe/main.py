"""
title: Chatty Pipe
author: matthewh
version: 0.3.3
required_open_webui_version: 0.4.0

Instructions:
1. Configure and enable the Chat Completion Engine:
    - Go to Admin Panel > Functions
    - Ensure that the Pipe is enabled.
    - Click on the Chatty Pipe Config.
    - Set a base model ie "llama3.2:latest" and click Save.
2. Start a new chat using the Pipe:
    - Click "New Chat".
    - Select "Chatty Pipe" from the model dropdown list.

TODO
- [x] Follow up feature.
- [x] Internal query detection for tags and titles with proper history management.
- [ ] Status completion when interrupted by user
- [ ] Don't error when base model has a system prompt
"""
from typing import Optional, Callable, Awaitable, Dict, Any, List
from pydantic import BaseModel, Field
import asyncio
import logging
import random
import re
import json  # Ensure json is imported

from open_webui.main import generate_chat_completions, get_all_models
from open_webui.utils.misc import pop_system_message, get_last_user_message


class Pipe:
    """Pipeline for handling chat completion requests."""

    class Valves(BaseModel):
        """Configuration for the Chat Completion Pipe."""

        # Top-Level Valve: Base Model ID
        base_model_id: str = Field(
            default="gpt-4:latest",
            description="Base model to be used for generating chat completions.",
        )

        # Valve to enable follow-up message feature
        enable_follow_up: bool = Field(
            default=True,
            description="Enable or disable follow-up messages after inactivity.",
        )

        # Valve to enable status emissions
        enable_status_emits: bool = Field(
            default=True,
            description="Enable or disable status message emissions.",
        )

        # Valve for toggling the follow-up system message
        enable_follow_up_system_message: bool = Field(
            default=True,
            description="Enable or disable the follow-up system message.",
        )

        # Follow-up system message
        follow_up_system_message: str = Field(
            default="You are concerned that the user has abandoned the conversation, check that they are still there by questioning them succinctly.",
            description="System message to use for follow-up prompts when enabled.",
        )

        # Valves for follow-up timer
        fixed_delay: float = Field(
            default=30.0,
            description="Fixed delay in seconds before emitting a follow-up message.",
        )
        random_delay: float = Field(
            default=10.0,
            description="Random additional delay in seconds before emitting a follow-up message.",
        )

        # Maximum number of follow-ups before completing
        max_follow_ups: int = Field(
            default=2,
            description="Maximum number of follow-up messages before emitting a 'Completed' status.",
        )

        # Valve for tag prompt detection
        tag_prompt_detection: str = Field(
            default='JSON format: { "tags": ["tag1", "tag2", "tag3"] }',
            description="Detection pattern to identify tag generation requests.",
        )

        # Valve for title prompt detection
        title_prompt_detection: str = Field(
            default='RESPOND ONLY WITH THE TITLE TEXT.',
            description="Detection pattern to identify title generation requests.",
        )

        # Common valve
        enable_debug: bool = Field(
            default=True, description="Enable or disable debug logging."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG if self.valves.enable_debug else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)

        # Initialize conversation_history as a dict mapping user_id to their conversation data
        # Each user_id maps to a dict with:
        # - 'messages' (list of dicts with 'role' and 'content')
        # - 'timer_task' (asyncio.Task or None)
        # - 'follow_up_count' (int)
        # - 'system_prompt' (str)
        # - 'timer_version' (int)
        self.conversation_history: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

        # Ensure that conversation_history is empty upon initialization
        self.conversation_history.clear()
        self.log.debug("Initialized Pipe with empty conversation history.")

    def clean_message(self, message: str) -> str:
        """
        Remove leading and trailing quotes from the message if both exist.

        Args:
            message (str): The message to clean.

        Returns:
            str: The cleaned message.
        """
        pattern = r'^([\'"])(.*)\1$'
        match = re.match(pattern, message)
        if match:
            return match.group(2)
        return message

    async def get_model_by_id(self, base_model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the model dictionary by its ID.

        Args:
            base_model_id (str): The model ID to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The model dictionary if found, else None.
        """
        try:
            available_models = await get_all_models()
            for model in available_models:
                if model["id"] == base_model_id:
                    return model
            return None
        except Exception as e:
            self.log.error(f"Error fetching models: {e}")
            return None

    async def call_llm(
        self,
        model: Dict[str, Any],
        messages: List[Dict[str, str]],
    ) -> Optional[str]:
        """
        Helper method to handle LLM chat completions with full conversation history.

        Args:
            model (Dict[str, Any]): The model dictionary to use.
            messages (List[Dict[str, str]]): The list of messages representing the conversation.

        Returns:
            Optional[str]: The generated content or None if failed.
        """
        payload = {
            "model": model["id"],
            "messages": messages,
            # Removed 'max_tokens' and 'temperature' to allow base_model configuration
        }

        self.log.debug(
            f"Payload for generate_chat_completions: {json.dumps(payload, indent=4)}"
        )

        try:
            response = await generate_chat_completions(
                form_data=payload,
                bypass_filter=True,  # Ensure bypass_filter is included
            )
            self.log.debug(f"LLM Response: {response}")

            # Validate response structure
            if (
                "choices" in response
                and len(response["choices"]) > 0
                and "message" in response["choices"][0]
                and "content" in response["choices"][0]["message"]
            ):
                content = response["choices"][0]["message"]["content"].strip()
                self.log.debug(f"Generated Content Before Cleanup: {content}")
                cleaned_content = self.clean_message(content)
                self.log.debug(f"Generated Content After Cleanup: {cleaned_content}")
                return cleaned_content
            else:
                self.log.error("Invalid response structure from LLM.")
                self.log.debug(f"Full LLM Response: {json.dumps(response, indent=4)}")
                return None
        except Exception as e:
            self.log.error(f"Error during LLM call: {e}")
            return None

    async def handle_internal_query(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
        message: str,
    ):
        """Handle internal queries like tag generation or title generation."""
        self.log.debug(f"Handling internal query for user {user_id}: {message}")

        # Determine the type of internal query
        if self.valves.tag_prompt_detection in message:
            # Tag Generation
            self.log.debug(f"Detected tag generation request for user {user_id}.")
            llm_response = await self.call_llm(
                model={"id": self.valves.base_model_id},
                messages=[{"role": "user", "content": message}],
            )
        elif self.valves.title_prompt_detection in message:
            # Title Generation
            self.log.debug(f"Detected title generation request for user {user_id}.")
            llm_response = await self.call_llm(
                model={"id": self.valves.base_model_id},
                messages=[{"role": "user", "content": message}],
            )
        else:
            # Unknown internal query
            self.log.debug(f"Unknown internal request for user {user_id}.")
            llm_response = "Invalid internal request."

        if llm_response:
            # Emit the response with a newline
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": f"{llm_response}\n"},
                }
            )
            self.log.debug(f"Emitted internal response to user {user_id}: {llm_response}")
        else:
            fallback_message = "Unable to process your internal request at the moment.\n"
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": fallback_message},
                }
            )
            self.log.debug(f"Emitted fallback internal response to user {user_id}.")

    async def emit_chat_response(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
        message: str,
    ):
        """Generate and emit a chat completion response based on user input."""
        base_model_id = self.valves.base_model_id

        # Retrieve and validate the model
        model = await self.get_model_by_id(base_model_id)
        if model is None:
            error_message = (
                f"Your base model is configured as {base_model_id} but your open-webui does not have any matching model by that ID."
            )
            self.log.error(error_message)
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": error_message},
                }
            )
            return

        # Update conversation history with the new user message
        async with self.lock:
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = {
                    "messages": [],
                    "timer_task": None,
                    "follow_up_count": 0,
                    "system_prompt": "",
                    "timer_version": 0,
                }
            self.conversation_history[user_id]["messages"].append(
                {"role": "user", "content": message}
            )

            # Retrieve and store system prompt using pop_system_message
            stored_system_prompt, filtered_messages = pop_system_message(
                self.conversation_history[user_id]["messages"]
            )
            if stored_system_prompt:
                self.conversation_history[user_id]["system_prompt"] = stored_system_prompt["content"]
                self.conversation_history[user_id]["messages"] = filtered_messages
                self.log.debug(
                    f"Stored system prompt for user {user_id}: {self.conversation_history[user_id]['system_prompt']}"
                )
            else:
                # If no system prompt in messages, determine based on model and Pipe configuration
                if "system_prompt" in model and model["system_prompt"]:
                    self.conversation_history[user_id]["system_prompt"] = model["system_prompt"]
                    self.log.debug(
                        f"Using system prompt from model '{base_model_id}': {model['system_prompt']}"
                    )
                elif self.valves.enable_follow_up_system_message:
                    self.conversation_history[user_id]["system_prompt"] = self.valves.follow_up_system_message
                    self.log.debug(
                        f"Using Pipe's follow-up system prompt for user {user_id}: {self.valves.follow_up_system_message}"
                    )
                else:
                    self.conversation_history[user_id]["system_prompt"] = "You are a helpful assistant."
                    self.log.debug(
                        f"No system prompt found for model '{base_model_id}'. Using default system prompt."
                    )

            # Increment timer_version to invalidate old timers
            self.conversation_history[user_id]["timer_version"] += 1
            current_timer_version = self.conversation_history[user_id]["timer_version"]

            # Prepare the conversation with the stored system prompt
            conversation = [
                {
                    "role": "system",
                    "content": self.conversation_history[user_id]["system_prompt"],
                }
            ] + [
                msg for msg in self.conversation_history[user_id]["messages"]
                if msg["role"] in ["user", "assistant"]
            ]

        # Generate LLM response with full conversation history
        self.log.debug(f"Generating chat completion for user {user_id}.")
        llm_response = await self.call_llm(
            model=model,
            messages=conversation,
        )

        if llm_response:
            # Update conversation history with assistant's response
            async with self.lock:
                self.conversation_history[user_id]["messages"].append(
                    {"role": "assistant", "content": llm_response}
                )
                # Reset follow_up_count on new user interaction
                self.conversation_history[user_id]["follow_up_count"] = 0

            # Emit the response with a newline
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": f"{llm_response}\n"},
                }
            )
            self.log.debug(f"Emitted chat response to user {user_id}: {llm_response}")
        else:
            error_message = (
                "I'm sorry, I couldn't process your request at the moment.\n"
            )
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": error_message},
                }
            )
            self.log.debug(f"Emitted fallback error message to user {user_id}.")

    async def emit_follow_up(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
    ):
        """Generate and emit a follow-up message based on conversation history."""
        if not self.valves.enable_follow_up:
            self.log.debug("Follow-up messages are disabled via valves.")
            return

        base_model_id = self.valves.base_model_id

        # Retrieve conversation history
        async with self.lock:
            user_data = self.conversation_history.get(user_id, {})
            conversation = user_data.get("messages", [])
            follow_up_count = user_data.get("follow_up_count", 0)
            system_prompt = user_data.get(
                "system_prompt", "You are a helpful assistant."
            )
            timer_version = user_data.get("timer_version", 0)

        if follow_up_count >= self.valves.max_follow_ups:
            # Emit "Completed" status and reset follow_up_count
            completed_message = "Completed\n"
            if self.valves.enable_status_emits:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": completed_message, "done": True},
                    }
                )
            self.log.debug(
                f"Emitted completed status message to user {user_id}: {completed_message.strip()}"
            )

            # Reset follow_up_count
            async with self.lock:
                if user_id in self.conversation_history:
                    self.conversation_history[user_id]["follow_up_count"] = 0
            return

        if not conversation:
            self.log.debug(f"No conversation history found for user {user_id}.")
            return

        # Instruction for follow-up: Use the original user instruction and include existing assistant output
        # Extract original user instruction (assumed to be the first user message)
        original_user_instruction = None
        for msg in conversation:
            if msg["role"] == "user":
                original_user_instruction = msg["content"]
                break

        if not original_user_instruction:
            self.log.debug(
                f"No original user instruction found for user {user_id}. Using default follow-up instruction."
            )
            original_user_instruction = "Please provide a thoughtful and detailed follow-up message to encourage further discussion or offer additional assistance."

        # Extract assistant's last response
        assistant_last_response = None
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                assistant_last_response = msg["content"]
                break

        if not assistant_last_response:
            self.log.debug(
                f"No assistant response found for user {user_id}. Using default follow-up instruction."
            )
            assistant_last_response = ""

        # Combine original user instruction and assistant's last response
        follow_up_instruction = (
            f"{original_user_instruction}\nAssistant: {assistant_last_response}"
        )

        self.log.debug(
            f"Generating follow-up message for user {user_id} with instruction: {follow_up_instruction}"
        )

        # Prepare the conversation without injecting another system prompt
        # The system prompt is already included in the conversation history
        follow_up_messages = [
            msg for msg in conversation
            if msg["role"] in ["user", "assistant"]
        ] + [{"role": "user", "content": follow_up_instruction}]

        # Log the follow-up messages for debugging
        self.log.debug(
            f"Follow-up Messages for user {user_id}: {json.dumps(follow_up_messages, indent=4)}"
        )

        # Generate follow-up message with full conversation history
        follow_up_message = await self.call_llm(
            model={"id": base_model_id},
            messages=follow_up_messages,
        )

        if follow_up_message:
            # Update conversation history with assistant's response
            async with self.lock:
                self.conversation_history[user_id]["messages"].append(
                    {"role": "assistant", "content": follow_up_message}
                )
            # Emit the follow-up message with a newline
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": f"{follow_up_message}\n"},
                }
            )
            self.log.debug(
                f"Emitted follow-up message to user {user_id}: {follow_up_message}"
            )
        else:
            fallback_message = "Is there anything else you'd like to discuss?\n"
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": fallback_message},
                }
            )
            self.log.debug(f"Emitted fallback follow-up message to user {user_id}.")

        # Increment follow_up_count
        async with self.lock:
            if user_id in self.conversation_history:
                self.conversation_history[user_id]["follow_up_count"] += 1
                current_follow_up_count = self.conversation_history[user_id][
                    "follow_up_count"
                ]

        # Only schedule another follow-up if not exceeded max_follow_ups
        if current_follow_up_count < self.valves.max_follow_ups:
            await self.emit_follow_up_with_status(__event_emitter__, user_id)
        else:
            if self.valves.enable_status_emits:
                completed_message = "Completed\n"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": completed_message, "done": True},
                    }
                )
                self.log.debug(
                    f"Emitted completed status message to user {user_id}: {completed_message.strip()}"
                )

    async def emit_follow_up_with_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
    ):
        """Emit status events indicating the wait time before sending a follow-up message."""
        if not self.valves.enable_follow_up:
            self.log.debug("Follow-up messages are disabled via valves.")
            return

        # Check if follow_up_count < max_follow_ups before emitting status
        async with self.lock:
            follow_up_count = self.conversation_history[user_id]["follow_up_count"]
            if follow_up_count >= self.valves.max_follow_ups:
                self.log.debug(
                    f"User {user_id} has reached maximum follow-ups ({self.valves.max_follow_ups}). No further follow-ups will be scheduled."
                )
                return

        # Calculate total delay
        total_delay = self.valves.fixed_delay + random.uniform(
            0, self.valves.random_delay
        )
        self.log.debug(
            f"Total follow-up delay for user {user_id}: {total_delay:.2f} seconds."
        )

        # Emit the initial status message indicating the wait time if enabled
        if self.valves.enable_status_emits:
            status_message = f"We will wait for {total_delay:.2f} seconds before sending a follow-up message.\n"
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": status_message, "done": False},
                }
            )
            self.log.debug(
                f"Emitted status event for user {user_id}: {status_message.strip()}"
            )

        # Start the follow-up timer coroutine
        async with self.lock:
            timer_version = self.conversation_history[user_id]["timer_version"]

        follow_up_task = asyncio.create_task(
            self.follow_up_timer_handler_with_status(
                __event_emitter__, user_id, total_delay, timer_version
            )
        )

        # Assign the task to timer_task for potential cancellation
        async with self.lock:
            # Cancel any existing follow_up_task
            existing_follow_up_task = self.conversation_history[user_id].get(
                "timer_task"
            )
            if existing_follow_up_task and not existing_follow_up_task.done():
                existing_follow_up_task.cancel()
                self.log.debug(
                    f"Cancelled existing follow-up timer for user {user_id}."
                )

            # Assign the new follow-up task
            self.conversation_history[user_id]["timer_task"] = follow_up_task

    async def follow_up_timer_handler_with_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
        delay: float,
        timer_version: int,
    ):
        """Handle the follow-up timer and emit the follow-up message."""
        try:
            # Wait for the total delay
            await asyncio.sleep(delay)

            # Before emitting follow-up, check if this timer is still valid
            async with self.lock:
                current_timer_version = self.conversation_history[user_id][
                    "timer_version"
                ]
                if timer_version != current_timer_version:
                    self.log.debug(
                        f"Timer version mismatch for user {user_id}. Expected {current_timer_version}, got {timer_version}. Aborting follow-up."
                    )
                    return

            self.log.debug(
                f"Follow-up timer expired for user {user_id}. Emitting follow-up message."
            )
            await self.emit_follow_up(__event_emitter__, user_id)

        except asyncio.CancelledError:
            self.log.debug(f"Follow-up timer was cancelled for user {user_id}.")
            # Clear the timer_task reference as the timer has been cancelled
            async with self.lock:
                if user_id in self.conversation_history:
                    self.conversation_history[user_id]["timer_task"] = None

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Main handler for chat completion requests."""
        # Extract user_id securely
        user_id = (
            __user__["id"]
            if __user__ and "id" in __user__
            else f"anonymous_{id(__user__)}"
        )

        try:
            # Extract system prompt and messages from the body using pop_system_message
            system_prompt, messages = pop_system_message(body.get("messages", []))
            user_message = get_last_user_message(messages)
            if user_message is None:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": "I didn't receive any message from you.\n"},
                    }
                )
                self.log.debug("No user message provided; emitting default response.")
                return {"error": "No message provided for chat completion."}

            if self.valves.enable_debug:
                self.log.debug(f"[chat_completion] User: {user_message}")
                self.log.debug(f"[chat_completion] User ID: {user_id}")
                self.log.debug(f"[chat_completion] System Prompt: {system_prompt}")

            # Check for internal query detection
            if self.valves.tag_prompt_detection in user_message:
                self.log.debug(f"Detected tag prompt in user message for user {user_id}.")
                # The user message is already appended to conversation_history
                await self.handle_internal_query(__event_emitter__, user_id, user_message)
                return {"status": "internal_request_handled"}

            elif self.valves.title_prompt_detection in user_message:
                self.log.debug(f"Detected title prompt in user message for user {user_id}.")
                # The user message is already appended to conversation_history
                await self.handle_internal_query(__event_emitter__, user_id, user_message)
                return {"status": "internal_request_handled"}

            else:
                # Emit chat response as a regular user message
                await self.emit_chat_response(__event_emitter__, user_id, user_message)

                # Start or reset the follow-up timer after the initial response is emitted
                if self.valves.enable_follow_up:
                    # Start or reset the follow-up timer
                    await self.emit_follow_up_with_status(__event_emitter__, user_id)

                return {"status": "success"}

        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n"
            if self.valves.enable_debug:
                self.log.error(f"Error during chat completion: {e}")
            # Handle error as a regular user message
            await self.emit_chat_response(__event_emitter__, user_id, error_message)
            return {"error": str(e)}
