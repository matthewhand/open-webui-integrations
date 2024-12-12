"""
title: Chatty Pipe
author: matthewh
version: 0.9.0
required_open_webui_version: 0.4.0

Instructions:
1. Configure the Chat Completion Engine:
    - Go to Admin Panel > Settings > Chat.
    - Set Chat Completion Engine to "Default (Open AI)".
    - Use model "llama3.2:latest"
2. Enable the Chatty Pipe:
    - Navigate to Workspace > Functions.
    - Ensure that the Pipe is enabled.
3. Start a new chat using the Pipe:
    - Click "New Chat".
    - Select "Chatty Pipe" from the model dropdown list.

- [x] Follow up feature.
- [ ] Status completion when interrupted
- [ ] Don't error when base model has a system prompt
"""

from typing import Optional, Callable, Awaitable, Dict, Any, List
from pydantic import BaseModel, Field
import asyncio
import logging
import random
import re
import json  # Ensure json is imported

from open_webui.main import generate_chat_completions
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

        # Valve for overriding the follow-up system message
        follow_up_system_message_override: str = Field(
            default="your-follow-up-system-prompt",
            description="Override for the follow-up system prompt. If set to 'your-follow-up-system-prompt', the system uses the existing system prompt. Otherwise, it uses the provided override prompt.",
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

        # Timeout after which to emit 'Interrupted' if no new follow-ups
        follow_up_timeout: float = Field(
            default=60.0,
            description="Timeout in seconds after which to emit 'Interrupted' if no new follow-up messages are detected.",
        )

        # Common valve
        enable_debug: bool = Field(
            default=False, description="Enable or disable debug logging."
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
        # - 'messages' (list)
        # - 'timer_task' (asyncio.Task or None)
        # - 'interruption_task' (asyncio.Task or None)
        # - 'follow_up_count' (int)
        # - 'system_prompt' (str)
        # - 'timer_version' (int)
        self.conversation_history: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

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

    async def call_llm(
        self,
        base_model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        """
        Helper method to handle LLM chat completions with full conversation history.

        Args:
            base_model_id (str): The model ID to use.
            messages (List[Dict[str, str]]): The list of messages representing the conversation.
            max_tokens (int): Maximum tokens for the response.
            temperature (float): Temperature setting for the response.

        Returns:
            Optional[str]: The generated content or None if failed.
        """
        payload = {
            "model": base_model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
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

    async def emit_chat_response(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
        message: str,
    ):
        """Generate and emit a chat completion response based on user input."""
        # No need to check for enable_llm_features or enable_chat_response as they are always enabled

        base_model_id = self.valves.base_model_id

        # Update conversation history
        async with self.lock:
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = {
                    "messages": [],
                    "timer_task": None,
                    "interruption_task": None,
                    "follow_up_count": 0,
                    "system_prompt": "You are a helpful assistant.",
                    "timer_version": 0,
                }
            self.conversation_history[user_id]["messages"].append(
                {"role": "user", "content": message}
            )

            # Retrieve and store system prompt using pop_system_message
            system_prompt, filtered_messages = pop_system_message(
                self.conversation_history[user_id]["messages"]
            )
            if system_prompt:
                self.conversation_history[user_id]["system_prompt"] = system_prompt
                self.conversation_history[user_id]["messages"] = filtered_messages

            # Increment timer_version to invalidate old timers
            self.conversation_history[user_id]["timer_version"] += 1
            current_timer_version = self.conversation_history[user_id]["timer_version"]

            # Prepare the conversation with the stored system prompt
            conversation = [
                {
                    "role": "system",
                    "content": self.conversation_history[user_id]["system_prompt"],
                }
            ] + self.conversation_history[user_id]["messages"]

            # Cancel any existing interruption_task as user has responded
            existing_interruption_task = self.conversation_history[user_id].get(
                "interruption_task"
            )
            if existing_interruption_task and not existing_interruption_task.done():
                existing_interruption_task.cancel()
                self.log.debug(
                    f"Cancelled existing interruption timer for user {user_id} due to new user message."
                )

        # Generate LLM response with full conversation history
        self.log.debug(f"Generating chat completion for user {user_id}.")
        llm_response = await self.call_llm(
            base_model_id=base_model_id,
            messages=conversation,
            max_tokens=200,  # Increased max_tokens for more detailed responses
            temperature=0.7,
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

        # Determine which system prompt to use based on the override valve
        if (
            self.valves.follow_up_system_message_override
            != "your-follow-up-system-prompt"
        ):
            follow_up_system_prompt = self.valves.follow_up_system_message_override
            self.log.debug(
                f"Using overridden follow-up system prompt for user {user_id}: {follow_up_system_prompt}"
            )
        else:
            follow_up_system_prompt = system_prompt
            self.log.debug(
                f"Using existing system prompt for user {user_id}: {follow_up_system_prompt}"
            )

        # Create a new list of messages including the follow-up instruction as a separate 'user' message
        follow_up_messages = (
            [{"role": "system", "content": follow_up_system_prompt}]
            + self.conversation_history[user_id]["messages"]
            + [{"role": "user", "content": follow_up_instruction}]
        )

        # Log the follow-up messages for debugging
        self.log.debug(
            f"Follow-up Messages for user {user_id}: {json.dumps(follow_up_messages, indent=4)}"
        )

        # Generate follow-up message with full conversation history
        follow_up_message = await self.call_llm(
            base_model_id=base_model_id,
            messages=follow_up_messages,
            max_tokens=150,  # Increased max_tokens for more detailed follow-ups
            temperature=0.6,  # Slightly higher temperature for variability
        )

        if follow_up_message:
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

        # After emitting a follow-up, set up a new follow-up timer
        if self.valves.enable_follow_up:
            await self.emit_follow_up_with_status(__event_emitter__, user_id)

    async def emit_follow_up_with_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
    ):
        """Emit status events indicating the wait time before sending a follow-up message and handle 'Interrupted'."""
        if not self.valves.enable_follow_up:
            self.log.debug("Follow-up messages are disabled via valves.")
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

        # Start the follow-up timer coroutine with versioning
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

            # Start the interruption timer
            # Cancel any existing interruption_task
            existing_interruption_task = self.conversation_history[user_id].get(
                "interruption_task"
            )
            if existing_interruption_task and not existing_interruption_task.done():
                existing_interruption_task.cancel()
                self.log.debug(
                    f"Cancelled existing interruption timer for user {user_id}."
                )

            # Start a new interruption_task
            interruption_task = asyncio.create_task(
                self.emit_follow_up_timeout(
                    __event_emitter__,
                    user_id,
                    self.valves.follow_up_timeout,
                    timer_version,
                )
            )
            self.conversation_history[user_id]["interruption_task"] = interruption_task

    async def follow_up_timer_handler_with_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
        delay: float,
        timer_version: int,
    ):
        """Handle the follow-up timer and emit the follow-up message followed by 'Completed' status."""
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

    async def emit_follow_up_timeout(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
        timeout: float,
        timer_version: int,
    ):
        """Handle the follow-up timeout and emit 'Interrupted' status."""
        try:
            # Wait for the follow-up timeout
            await asyncio.sleep(timeout)

            # Before emitting 'Interrupted', check if this timeout is still valid
            async with self.lock:
                current_timer_version = self.conversation_history[user_id][
                    "timer_version"
                ]
                if timer_version != current_timer_version:
                    self.log.debug(
                        f"Interruption timer version mismatch for user {user_id}. Expected {current_timer_version}, got {timer_version}. Aborting 'Interrupted' emission."
                    )
                    return

            self.log.debug(
                f"Follow-up timeout expired for user {user_id}. Emitting 'Interrupted' status."
            )
            await self.emit_interrupted(__event_emitter__, user_id)

        except asyncio.CancelledError:
            self.log.debug(f"Interruption timer was cancelled for user {user_id}.")
            # Clear the interruption_task reference as the timer has been cancelled
            async with self.lock:
                if user_id in self.conversation_history:
                    self.conversation_history[user_id]["interruption_task"] = None

    async def emit_interrupted(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        user_id: str,
    ):
        """Emit 'Interrupted' status if no new follow-up messages are detected within the timeout."""
        interrupted_message = "Interrupted\n"
        if self.valves.enable_status_emits:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": interrupted_message, "done": True},
                }
            )
        self.log.debug(
            f"Emitted interrupted status message to user {user_id}: {interrupted_message.strip()}"
        )

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
                await self.emit_chat_response(
                    __event_emitter__,
                    user_id,
                    "I didn't receive any message from you.\n",
                )
                self.log.debug("No user message provided; emitting default response.")
                return {"error": "No message provided for chat completion."}

            if self.valves.enable_debug:
                self.log.debug(f"[chat_completion] User: {user_message}")
                self.log.debug(f"[chat_completion] User ID: {user_id}")
                self.log.debug(f"[chat_completion] System Prompt: {system_prompt}")

            # Emit chat response
            await self.emit_chat_response(__event_emitter__, user_id, user_message)

            # Start or reset the follow-up timer after the initial response is emitted
            if self.valves.enable_follow_up:
                # Calculate total delay
                total_delay = self.valves.fixed_delay + random.uniform(
                    0, self.valves.random_delay
                )
                self.log.debug(
                    f"Total follow-up delay for user {user_id}: {total_delay:.2f} seconds."
                )

                # Emit status event indicating the wait time if enabled
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

                # Start or reset the follow-up timer
                await self.emit_follow_up_with_status(__event_emitter__, user_id)

            return {"status": "success"}

        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n"
            if self.valves.enable_debug:
                self.log.error(f"Error during chat completion: {e}")
            await self.emit_chat_response(__event_emitter__, user_id, error_message)
            return {"error": str(e)}
