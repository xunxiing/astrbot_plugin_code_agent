from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


logger = getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    step_number: int
    timing: Timing
    model_input_messages: list[ChatMessage] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": self.model_output_message.dict() if self.model_output_message else None,
            "model_output": self.model_output,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any


class AgentMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)


__all__ = ["AgentMemory"]
