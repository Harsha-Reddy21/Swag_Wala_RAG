from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db
import crud
import schemas
from rag import process_query

router = APIRouter()

@router.post("/conversations/", response_model=schemas.Conversation)
def create_conversation(conversation: schemas.ConversationCreate, db: Session = Depends(get_db)):
    return crud.create_conversation(db=db, conversation=conversation)

@router.get("/conversations/", response_model=List[schemas.Conversation])
def list_conversations(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    conversations = crud.get_conversations(db, skip=skip, limit=limit)
    return conversations

@router.get("/conversations/{conversation_id}", response_model=schemas.Conversation)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    conversation = crud.get_conversation(db, conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@router.post("/chat/{conversation_id}", response_model=schemas.ChatResponse)
async def chat(conversation_id: int, message: schemas.MessageCreate, db: Session = Depends(get_db)):
    conversation = crud.get_conversation(db, conversation_id=conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Save user message
    crud.create_message(db=db, message=message, conversation_id=conversation_id)
    
    # Process the message using RAG
    response = await process_query(message.content)
    
    # Save assistant response
    crud.create_message(
        db=db,
        message=schemas.MessageCreate(content=response, role="assistant"),
        conversation_id=conversation_id
    )
    
    return schemas.ChatResponse(response=response)