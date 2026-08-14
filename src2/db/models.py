"""
SQLAlchemy ORM models for KYC Verification System.
Includes User authentication and KYCSession tables for Supabase PostgreSQL.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_new_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    # Relationship to KYC Sessions
    sessions = relationship("KYCSession", back_populates="user")


class KYCSession(Base):
    __tablename__ = "kyc_sessions"

    id = Column(String, primary_key=True, default=_new_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)

    # Session Status: pending | processing | done | error
    status = Column(String, nullable=False, default="pending")

    # Liveness challenge instructions generated at session start
    liveness_instructions = Column(JSON, nullable=True)

    # Terminal state outcomes
    decision = Column(String, nullable=True)  # approved | manual_review | rejected | mobile_handoff_required
    rejection_reason = Column(String, nullable=True)
    extracted_fields = Column(JSON, nullable=True)
    face_similarity = Column(Float, nullable=True)
    requires_mobile_handoff = Column(Boolean, nullable=False, default=False)

    # National ID extracted from OCR for duplicate detection
    national_id = Column(String, nullable=True, index=True)

    # Mobile Handoff fields
    email = Column(String, nullable=True)
    handoff_token = Column(String, nullable=True)
    handoff_token_expires_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Relationship to User model
    user = relationship("User", back_populates="sessions")


import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class ConsolidationAgent:
    def __init__(self):
  
        api_key = os.environ.get("GROQ_API_KEY")
        
    
        self.llm = ChatGroq(
            temperature=0.0, 
            model_name="llama-3.1-70b-versatile", 
            api_key=api_key
        )
        

        self.prompt = PromptTemplate(
            input_variables=["raw_data"],
            template="""You are an expert Egyptian KYC data entry agent.
            I will provide you with raw OCR extracted text from an Egyptian National ID card.
            The OCR might have spaces in the middle of words (e.g., "م ح م د"), misspellings, or noise characters.
            Your job is to read the raw strings, fix any Arabic spelling/spacing issues, and consolidate them into a clean JSON object.
            
            Raw OCR Data:
            {raw_data}
            
            Rules:
            1. Fix fragmented Arabic words (e.g., "م ح م د" -> "محمد").
            2. Remove any random punctuation or noise.
            3. If a field is completely unreadable or missing, set its value to null.
            4. Do NOT invent or hallucinate data. Only clean what is provided.
            5. Output ONLY a valid JSON object matching the keys from the input. Do not include markdown tags like ```json.
            """
        )

    def consolidate_all(self, enhanced_reads: dict) -> dict:

        if not enhanced_reads:
            return {}
            
        try:

            raw_data_str = json.dumps(enhanced_reads, ensure_ascii=False)
            

            chain = self.prompt | self.llm
            response = chain.invoke({"raw_data": raw_data_str})
            

            cleaned_text = response.content.replace('```json', '').replace('```', '').strip()
            

            cleaned_data = json.loads(cleaned_text)
            return cleaned_data
            
        except Exception as e:
            print(f"[ConsolidationAgent] Error: {e}")

            return {}