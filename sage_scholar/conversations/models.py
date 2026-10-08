"""Conversation owns dialogue; Run owns one execution and its durable state."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class Conversation(BaseModel):
    conversation_id: str
    workspace_id: str
    title: str
    archived: bool = False
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    status: str = 'completed'


class ConversationMessage(BaseModel):
    message_id: str
    conversation_id: str
    run_id: str
    role: Literal['user', 'assistant']
    content: str = ''
    kind: str = 'message'
    created_at: datetime
    data: dict = Field(default_factory=dict)
