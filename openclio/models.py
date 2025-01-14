import random
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class MessageDict(TypedDict):
    """Represents a message in the LLM API format"""
    role: Literal["user", "assistant"]
    content: str

class Projection(TypedDict):
    x: float
    y: float

class MetadataDict(TypedDict):
    """Represents metadata for conversations and clusters"""
    task: str
    concerning_score: float
    request: str
    num_turns: int
    country: str | None
    session_date: str | None
    user_id: str | None
    embedding: list[float]
    projection: Projection | None

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    role: Literal["user", "assistant"]
    content: str
    timestamp: float | None = None
    metadata: MetadataDict = field(default_factory=dict)

@dataclass 
class Conversation:
    """Enhanced conversation class with structured turn data"""
    id: str
    turns: list[ConversationTurn]
    metadata: MetadataDict = field(default_factory=dict)
    
    @property
    def content(self) -> str:
        """Get full conversation content"""
        return "\n".join(f"{turn.role}: {turn.content}" for turn in self.turns)
    
    @property
    def num_turns(self) -> int:
        """Get number of conversation turns"""
        return len(self.turns)
    
    @property
    def human_messages(self) -> list[str]:
        """Get list of human messages"""
        return [turn.content for turn in self.turns if turn.role.lower() == "user"]

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary format for JSON serialization"""
        return {
            'id': self.id,
            'turns': [{'role': t.role, 'content': t.content} for t in self.turns],
            'metadata': {k:v for k,v in self.metadata.items() if k != 'embedding'}
        }

@dataclass 
class Cluster:
    """Represents a cluster of similar conversations"""
    id: str
    conversations: list[Conversation]
    name: str
    description: str
    level: int = 0
    parent: 'Cluster | None' = None
    children: list['Cluster'] = None
    projection: Projection | None = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

    def to_dict(self) -> dict[str, Any]:
        """Convert cluster to dictionary format for JSON serialization"""
        convs = self.conversations
        concern_scores = [c.metadata['concerning_score'] for c in convs]
        
        out = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'level': self.level,
            'parent_id': self.parent.id if self.parent else None,
            'conversation_ids': [c.id for c in convs],
            'concerning_score_avg': sum(concern_scores) / len(concern_scores) if concern_scores else 0,
            'sample_conversations': [c.id for c in random.sample(convs, min(3, len(convs)))],
            'projection': self.projection,
            'size': len(convs)
        }
        
        if self.children:
            out['child_clusters'] = [child.to_dict() for child in self.children]
            out['conversation_ids'] = [c.id for child in self.children for c in child.conversations]
            out['size'] = len(out['conversation_ids'])
            
        return out
