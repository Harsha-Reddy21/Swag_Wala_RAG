from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional

from models import Conversation, Message
import schemas

def create_conversation(db: Session, conversation: schemas.ConversationCreate) -> Conversation:
    db_conversation = Conversation(**conversation.model_dump())
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    return db_conversation

def get_conversation(db: Session, conversation_id: int) -> Optional[Conversation]:
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()

def get_conversations(db: Session, skip: int = 0, limit: int = 10) -> List[Conversation]:
    return db.query(Conversation).order_by(desc(Conversation.created_at)).offset(skip).limit(limit).all()

def create_message(db: Session, message: schemas.MessageCreate, conversation_id: int) -> Message:
    db_message = Message(**message.model_dump(), conversation_id=conversation_id)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message