from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """Sliding-window conversation buffer.

    Replaces LangChain's ``ConversationBufferWindowMemory(k=N)`` with a simple
    list that keeps the last *window_size* exchanges.
    """

    window_size: int = 50
    _messages: list[dict[str, str]] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Append a message.  *role* is typically ``"user"`` or ``"assistant"``."""
        self._messages.append({"role": role, "content": content})

    def get_window(self) -> list[dict[str, str]]:
        """Return the most recent *window_size* messages."""
        return self._messages[-self.window_size :]

    def get_all(self) -> list[dict[str, str]]:
        """Return every message regardless of window."""
        return list(self._messages)

    def clear(self) -> None:
        """Wipe all messages."""
        self._messages.clear()

    def to_prompt_text(self) -> str:
        """Render the window as a plain-text conversation transcript.

        Suitable for injecting into a prompt's ``{history}`` placeholder.
        """
        lines: list[str] = []
        for msg in self.get_window():
            role_label = msg["role"].capitalize()
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    def to_message_list(self) -> list[dict[str, str]]:
        """Return the window as a list of ``{"role": ..., "content": ...}``
        dicts ready for a chat-completion API call."""
        return self.get_window()

    def __len__(self) -> int:
        return len(self._messages)
