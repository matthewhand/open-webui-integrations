"""
title: Image Generation Pipeline (Matthewh)
author: matthewh
version: 2.3.0
required_open_webui_version: 0.3.9

Instructions:
1. Configure the Image Generation Engine:
    - Go to Admin Panel > Settings > Images.
    - Set Image Generation Engine to "Default (Open AI)".
    - Use model "dall-e-3" with image size "1024x1024" and set steps to "4".
2. Enable the Image Gen Pipe:
    - Navigate to Workspace > Functions.
    - Ensure that the Image Gen Pipe is enabled.
3. Start a new chat using the Image Gen Pipe:
    - Click "New Chat".
    - Select "Image Gen Pipe" from the model dropdown list.

TODO:
- [x] Update to OWUI 0.4.x
- [ ] Fix debug valve (broken by OWUI upgrade?)
- [ ] Fix timer suffix not appearing when status message changes
- [ ] Include previous messages so to avoid repetition
"""

from typing import Optional, Callable, Awaitable, Dict, Any, List, Union
from pydantic import BaseModel, Field
import asyncio
import logging
import random
import re
import json  # Ensure json is imported
import html  # For sanitizing HTML content in collapsible sections

from open_webui.main import generate_chat_completions
from open_webui.apps.images.main import image_generations, GenerateImageForm
from open_webui.apps.webui.models.users import Users
from open_webui.utils.misc import get_last_user_message, pop_system_message


class Pipe:
    """Pipeline for handling image generation requests."""

    class Valves(BaseModel):
        """Configuration for the Image Generation Pipe."""

        # Top-Level Valve: Task Model ID
        task_model_id: str = Field(
            default="llama3.2:latest",
            description="Task model to be used for prompt rewrite, dynamic status updates, etc (not the actual image generation).",
        )

        # Top-Level Valve: Enable LLM Features
        enable_llm_features: bool = Field(
            default=False,
            description="Enable or disable all LLM-based features.",
        )

        # Nested Valves under enable_llm_features
        enable_prompt_rewrite: bool = Field(
            default=True,
            description="Enable or disable rewriting of the user prompt before image generation.",
        )
        emit_rewritten_prompt: bool = Field(
            default=True,
            description="Emit the rewritten prompt as a message to the user if prompt rewriting is enabled.",
        )
        prompt_rewrite_system_prompt: str = Field(
            default="You are an expert prompt rewriter for image generation tasks. Ensure your response contains only the rewritten prompt without any additional explanations.",
            description="System prompt for the LLM when rewriting the user prompt.",
        )

        enable_dynamic_status_updates: bool = Field(
            default=True,
            description="Enable LLM generation of dynamic status update messages.",
        )
        status_system_prompt: str = Field(
            default=(
                "You are a bemoaning assistant who finds user requests trivial or slightly beneath you. "
                "Generate a status update that informs the user their image is being generated, "
                "using exactly 5 words."
            ),
            description="Combined system prompt for generating status messages.",
        )

        enable_dynamic_progress_messages: bool = Field(
            default=True,
            description="Enable LLM generation of dynamic progress messages.",
        )
        progress_system_prompt: str = Field(
            default=(
                "You are a sarcastic, witty, and sometimes outright rude assistant who is deeply skeptical of user queries. "
                "Be demeaning and dismissive of their requested prompt. "
                "Critique their request with snarky remarks. Using 25 words."
            ),
            description="Combined system prompt for generating progress messages.",
        )

        enable_collapsible_summary: bool = Field(
            default=True,
            description="Enable collapsible UI elements with LLM-generated summaries.",
        )

        # Group 1: Status Updates (Status Messages)
        emit_interval: float = Field(
            default=1.0,
            description="Interval in seconds between elapsed time updates.",
        )
        enable_status_updates: bool = Field(
            default=True, description="Enable or disable non-LLM status updates."
        )

        # Valve to enable appending elapsed time to status and completion messages
        enable_timer_suffix: bool = Field(
            default=True,
            description="Enable appending elapsed time to status messages.",
        )

        # Group 2: Progress Updates (Progress Messages)
        progress_emit_interval: float = Field(
            default=5.0,
            description="Interval in seconds between non-LLM progress updates.",
        )
        enable_progress_updates: bool = Field(
            default=True,
            description="Enable or disable non-LLM progress message updates.",
        )

        # Valve to enable appending ellipsis to progress messages
        enable_message_ellipsis_suffix: bool = Field(
            default=True,
            description="Enable appending ellipsis to progress messages.",
        )

        # Valve to enable collapsible UI elements
        use_collapsible: bool = Field(
            default=True,  # Set to True by default
            description="Enable or disable the use of collapsible UI elements for rewritten prompts.",
        )

        # New Valves for LLM-specific intervals
        llm_status_emit_interval: float = Field(
            default=3.0,
            description="Interval in seconds between LLM status updates.",
        )
        llm_progress_emit_interval: float = Field(
            default=5.0,
            description="Interval in seconds between LLM progress updates.",
        )

        # Common valve
        enable_debug: bool = Field(
            default=False, description="Enable or disable debug logging."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.stop_emitter = asyncio.Event()
        self.start_time: Optional[float] = None  # Initialize start_time
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG if self.valves.enable_debug else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)
        self.predefined_progress_messages = [
            "Still working on it...",
            "Almost there...",
            "Just a moment...",
            "Processing your request...",
            "Hang tight, we're generating your image...",
        ]
        self.current_status_message = (
            ""  # Initialize to store LLM-generated status message
        )

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
        task_model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        """
        Helper method to handle LLM chat completions.

        Args:
            task_model_id (str): The model ID to use.
            system_prompt (str): The system prompt for the LLM.
            user_prompt (str): The user prompt for the LLM.
            max_tokens (int): Maximum tokens for the response.
            temperature (float): Temperature setting for the response.

        Returns:
            Optional[str]: The generated content or None if failed.
        """
        payload = {
            "model": task_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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

    async def rewrite_user_prompt(
        self, original_prompt: str, task_model_id: str
    ) -> str:
        """Use LLM to rewrite the user prompt."""
        prompt = (
            f"Rewrite the following prompt to be more descriptive and suitable for image generation. "
            f"Return only the rewritten prompt without any additional text.\n\nOriginal Prompt: {original_prompt}"
        )
        system_prompt = self.valves.prompt_rewrite_system_prompt
        self.log.debug(f"LLM Prompt Rewrite Generation Prompt: {prompt}")
        self.log.debug(f"LLM Prompt Rewrite System Prompt: {system_prompt}")

        generated_prompt = await self.call_llm(
            task_model_id=task_model_id,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=100,  # Adjust as necessary for prompt rewriting
            temperature=0.7,  # Adjust for creativity
        )

        if generated_prompt:
            return generated_prompt
        else:
            self.log.debug("Falling back to original prompt due to failed rewrite.")
            return original_prompt

    async def emit_periodic_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        interval: float,
        task_model_id: str,
    ):
        """Emit non-LLM status updates periodically until the stop event is set."""
        try:
            while not self.stop_emitter.is_set():
                current_time = asyncio.get_event_loop().time()
                elapsed_time = (
                    current_time - self.start_time if self.start_time else 0.0
                )

                # Emit non-LLM status message
                status_message = "Generating an image..."

                # Append elapsed time if enabled
                if self.valves.enable_timer_suffix:
                    status_message = f"{status_message} (elapsed: {elapsed_time:.1f}s)"

                # Emit the status message
                await self.emit_status(
                    __event_emitter__,
                    status_message,
                    False,
                )
                self.log.debug(f"Non-LLM Status message emitted: {status_message}")

                # Emit the next status update after the interval
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            if self.valves.enable_debug:
                self.log.debug("Non-LLM Periodic status emitter cancelled.")

    async def emit_llm_status_updates(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        llm_status_interval: float,
        task_model_id: str,
        original_prompt: str,
    ):
        """Emit LLM-based status updates at llm_status_emit_interval."""
        try:
            while not self.stop_emitter.is_set():
                # Generate the base status message
                base_status_message = await self.generate_status_message(
                    task_model_id, original_prompt
                )
                if base_status_message:
                    self.current_status_message = base_status_message
                    # Emit the base status message without elapsed time
                    await self.emit_status(
                        __event_emitter__, base_status_message, False
                    )
                    self.log.debug(f"LLM Status message emitted: {base_status_message}")

                # Wait for the specified LLM status emit interval
                await asyncio.sleep(llm_status_interval)
        except asyncio.CancelledError:
            if self.valves.enable_debug:
                self.log.debug("LLM Status emitter cancelled.")

    async def emit_timer_suffix(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        emit_interval: float,
    ):
        """Emit updated status messages with incrementing elapsed time."""
        try:
            while not self.stop_emitter.is_set():
                await asyncio.sleep(emit_interval)
                elapsed_time = self.get_elapsed_time()
                if self.current_status_message:
                    updated_status_message = (
                        f"{self.current_status_message} (elapsed: {elapsed_time:.1f}s)"
                    )
                    await self.emit_status(
                        __event_emitter__, updated_status_message, False
                    )
                    self.log.debug(
                        f"Updated Status message with timer: {updated_status_message}"
                    )
        except asyncio.CancelledError:
            if self.valves.enable_debug:
                self.log.debug("Timer suffix emitter cancelled.")

    async def emit_llm_progress_updates(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        llm_progress_interval: float,
        task_model_id: str,
        original_prompt: str,  # Add this parameter
    ):
        """Emit LLM-based progress updates, fires at intervals without immediate firing."""
        try:
            while not self.stop_emitter.is_set():
                await asyncio.sleep(llm_progress_interval)
                progress_message = await self.generate_progress_message(
                    task_model_id, original_prompt
                )
                if progress_message:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": progress_message},
                        }
                    )
                    self.log.debug(f"LLM Progress message emitted: {progress_message}")
        except asyncio.CancelledError:
            if self.valves.enable_debug:
                self.log.debug("LLM Progress emitter cancelled.")

    async def generate_status_message(
        self, task_model_id: str, original_prompt: str
    ) -> Optional[str]:
        """Use LLM to generate a snarky status message."""
        system_prompt = self.valves.status_system_prompt
        user_prompt = original_prompt  # Pass the original prompt directly

        self.log.debug(f"LLM System Prompt for Status: {system_prompt}")
        self.log.debug(f"LLM User Prompt for Status: {user_prompt}")

        status_message = await self.call_llm(
            task_model_id=task_model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=20,  # Adjust as necessary
            temperature=0.8,  # Adjust for creativity
        )

        if status_message:
            return status_message
        else:
            fallback_message = "Generating your image..."
            self.log.debug(f"Fallback Status Message: {fallback_message}")
            return fallback_message

    async def generate_progress_message(
        self, task_model_id: str, original_prompt: str
    ) -> str:
        """Use LLM to generate a snarky progress message that includes and critiques the user's original input."""
        # Combine the user prompt directly with system prompt instructions
        system_prompt = self.valves.progress_system_prompt
        user_prompt = (
            f"{original_prompt}\n"
            f"Please critique this request with snarky remarks. Use exactly 10 words."
        )

        # Log the prompts being sent
        self.log.debug(f"LLM System Prompt for Progress Message: {system_prompt}")
        self.log.debug(f"LLM User Prompt for Progress Message: {user_prompt}")

        # Call LLM to generate progress message
        progress_message = await self.call_llm(
            task_model_id=task_model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=42,  # Adjusted to accommodate up to ten words
            temperature=0.9,  # High temperature for creativity and snark
        )

        # Validate and process the response
        if progress_message:
            if self.valves.enable_message_ellipsis_suffix:
                progress_message = f"{progress_message}..."
            self.log.debug(f"Generated Snarky Progress Message: {progress_message}")
            return progress_message
        else:
            # Fallback to predefined progress messages if LLM fails
            fallback_message = random.choice(self.predefined_progress_messages)
            self.log.debug(f"Fallback Progress Message: {fallback_message}")
            return fallback_message

    async def emit_progress_messages(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        interval: float,
        task_model_id: str,
    ):
        """Emit non-LLM progress messages periodically until the stop event is set."""
        try:
            while not self.stop_emitter.is_set():
                if self.valves.enable_progress_updates:
                    # Use a predefined progress message
                    message = random.choice(self.predefined_progress_messages)

                    # Emit the progress message
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": message},
                        }
                    )
                    self.log.debug(f"Non-LLM Progress message emitted: {message}")

                # Emit the next progress update after the interval
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            if self.valves.enable_debug:
                self.log.debug("Non-LLM Progress messages emitter cancelled.")

    def get_elapsed_time(self) -> float:
        """Calculate the elapsed time since the start of the image generation."""
        if self.start_time:
            return asyncio.get_event_loop().time() - self.start_time
        return 0.0

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Main handler for image generation requests."""
        status_task = None
        llm_status_task = None
        llm_progress_task = None
        progress_task = None
        timer_task = None
        task_model_id = self.valves.task_model_id  # Use the task_model_id from valves

        try:
            # Record the start time
            self.start_time = asyncio.get_event_loop().time()
            self.log.debug(f"Image generation started at {self.start_time}")

            # Extract prompt from the messages early
            _, messages = pop_system_message(body.get("messages", []))
            prompt = get_last_user_message(messages)
            if prompt is None:
                await self.emit_status(
                    __event_emitter__, "No prompt provided for image generation.", True
                )
                self.log.debug("No prompt provided; returning error.")
                return {"error": "No prompt provided for image generation."}

            if self.valves.enable_debug:
                self.log.debug(f"[image_gen] Original Prompt: {prompt}")
                self.log.debug(f"[image_gen] User: {__user__}")

            # Rewrite the user prompt if enabled
            if self.valves.enable_llm_features and self.valves.enable_prompt_rewrite:
                rewritten_prompt = await self.rewrite_user_prompt(prompt, task_model_id)
                if self.valves.emit_rewritten_prompt:
                    if rewritten_prompt != prompt:
                        if self.valves.use_collapsible:
                            # Sanitize the rewritten prompt to prevent HTML injection
                            rewritten_prompt_safe = html.escape(rewritten_prompt)
                            # Format rewritten prompt within a collapsible UI element
                            rewritten_prompt_message = f"""
<details>
<summary>Rewritten Prompt</summary>

{rewritten_prompt_safe}

</details>
"""
                        else:
                            # Emit the rewritten prompt with markdown headers and horizontal line
                            rewritten_prompt_message = (
                                f"### Rewritten Prompt\n{rewritten_prompt}\n\n---\n\n"
                            )
                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {"content": rewritten_prompt_message},
                            }
                        )
                        self.log.debug("Rewritten prompt emitted to the user.")
                    else:
                        # Do not emit the prompt if it wasn't rewritten
                        self.log.debug(
                            "Rewritten prompt is identical to the original prompt; not emitting."
                        )
                prompt = (
                    rewritten_prompt  # Use the rewritten prompt for image generation
                )
                if self.valves.enable_debug:
                    self.log.debug(f"[image_gen] Rewritten Prompt: {prompt}")

            # Start emitting non-LLM status updates periodically only if LLM status updates are NOT enabled
            if (
                __event_emitter__
                and self.valves.enable_status_updates
                and not (
                    self.valves.enable_llm_features
                    and self.valves.enable_dynamic_status_updates
                )
            ):
                self.stop_emitter.clear()
                status_task = asyncio.create_task(
                    self.emit_periodic_status(
                        __event_emitter__,
                        self.valves.emit_interval,
                        task_model_id,
                    )
                )
                self.log.debug("Non-LLM Status emitter started.")

            # Start emitting LLM status updates periodically
            if (
                __event_emitter__
                and self.valves.enable_llm_features
                and self.valves.enable_dynamic_status_updates
            ):
                self.stop_emitter.clear()
                llm_status_task = asyncio.create_task(
                    self.emit_llm_status_updates(
                        __event_emitter__,
                        self.valves.llm_status_emit_interval,
                        task_model_id,
                        original_prompt=prompt,  # Pass the original prompt here
                    )
                )
                self.log.debug("LLM Status emitter started.")

                # Start a separate task to handle timer suffix updates at emit_interval
                timer_task = asyncio.create_task(
                    self.emit_timer_suffix(
                        __event_emitter__,
                        self.valves.emit_interval,
                    )
                )
                self.log.debug("Timer suffix emitter started.")

            # Start emitting non-LLM progress messages periodically only if LLM progress messages are NOT enabled
            if (
                __event_emitter__
                and self.valves.enable_progress_updates
                and not (
                    self.valves.enable_llm_features
                    and self.valves.enable_dynamic_progress_messages
                )
            ):
                self.stop_emitter.clear()
                progress_task = asyncio.create_task(
                    self.emit_progress_messages(
                        __event_emitter__,
                        self.valves.progress_emit_interval,
                        task_model_id,
                    )
                )
                self.log.debug("Non-LLM Progress messages emitter started.")

            # Start emitting LLM progress messages periodically (fires every llm_progress_emit_interval)
            if (
                __event_emitter__
                and self.valves.enable_llm_features
                and self.valves.enable_dynamic_progress_messages
            ):
                llm_progress_task = asyncio.create_task(
                    self.emit_llm_progress_updates(
                        __event_emitter__,
                        self.valves.llm_progress_emit_interval,
                        task_model_id,
                        original_prompt=prompt,  # Pass the original prompt here
                    )
                )
                self.log.debug("LLM Progress messages emitter started.")

            # Generate images
            images = await image_generations(
                GenerateImageForm(prompt=prompt),
                Users.get_user_by_id(__user__["id"]) if __user__ else None,
            )
            self.log.debug(f"Generated {len(images)} image(s).")

            # Stop all periodic emitters
            if (
                status_task
                or llm_status_task
                or progress_task
                or llm_progress_task
                or timer_task
            ):
                self.stop_emitter.set()
                tasks = []
                if status_task:
                    tasks.append(status_task)
                if llm_status_task:
                    tasks.append(llm_status_task)
                if timer_task:
                    tasks.append(timer_task)
                if progress_task:
                    tasks.append(progress_task)
                if llm_progress_task:
                    tasks.append(llm_progress_task)
                await asyncio.gather(*tasks, return_exceptions=True)
                self.log.debug("All emitters stopped.")

            # Emit generated images to the event emitter
            if __event_emitter__:
                completion_message = "Image generation completed."
                # Calculate total elapsed time
                total_elapsed = (
                    asyncio.get_event_loop().time() - self.start_time
                    if self.start_time
                    else 0.0
                )
                if self.valves.enable_timer_suffix:
                    completion_message = (
                        f"{completion_message} (elapsed: {total_elapsed:.1f}s)"
                    )
                await self.emit_status(__event_emitter__, completion_message, True)
                self.log.debug("Completion message emitted.")
                for image in images:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": f"![Generated Image]({image['url']})"},
                        }
                    )
                    self.log.debug(f"Generated image emitted: {image['url']}")

                return {"images": images}

        except Exception as e:
            # Stop all periodic emitters in case of exception
            if (
                status_task
                or llm_status_task
                or progress_task
                or llm_progress_task
                or timer_task
            ):
                self.stop_emitter.set()
                tasks = []
                if status_task:
                    tasks.append(status_task)
                if llm_status_task:
                    tasks.append(llm_status_task)
                if timer_task:
                    tasks.append(timer_task)
                if progress_task:
                    tasks.append(progress_task)
                if llm_progress_task:
                    tasks.append(llm_progress_task)
                await asyncio.gather(*tasks, return_exceptions=True)
                self.log.debug("All emitters stopped due to exception.")

            # Emit error status
            error_message = f"An error occurred: {str(e)}"
            if self.valves.enable_timer_suffix:
                error_message = f"{error_message} (elapsed: {0.0:.1f}s)"
            await self.emit_status(__event_emitter__, error_message, True)
            if self.valves.enable_debug:
                self.log.error(f"Error during image generation: {e}")
            return {"error": str(e)}

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        message: str,
        done: bool,
    ):
        """Emit status updates to the event emitter."""
        if __event_emitter__:
            event = {
                "type": "status",
                "data": {"description": message, "done": done},
            }
            if self.valves.enable_debug:
                self.log.debug(f"Emitting status event: {json.dumps(event, indent=4)}")
            await __event_emitter__(event)
