import json
from datetime import datetime
from typing import Dict, Optional
from langchain_ollama.chat_models import ChatOllama

# SEAL Semantics: Edit Proposal Schema
# Tracks not just what to change, but why, what evidence supports it, and its risk.

class EditProposal:
    def __init__(
        self,
        intent: str,
        knowledge_gap: str,
        evidence: str,
        proposed_q: str,
        proposed_a: str,
        source: str,
        risk_score: float = 0.0,
        confidence: float = 0.0
    ):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.intent = intent
        self.knowledge_gap = knowledge_gap
        self.evidence = evidence
        self.proposed_q = proposed_q
        self.proposed_a = proposed_a
        self.source = source
        self.risk_score = risk_score
        self.confidence = confidence
        self.simulation_passed = False
        self.applied = False

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "intent": self.intent,
            "knowledge_gap": self.knowledge_gap,
            "evidence": self.evidence,
            "proposed_q": self.proposed_q,
            "proposed_a": self.proposed_a,
            "source": self.source,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "simulation_passed": self.simulation_passed,
            "applied": self.applied
        }

def propose_edit(topic: str, context: str, llm: ChatOllama) -> Optional[EditProposal]:
    """
    Generates a structured Edit Proposal based on a research topic and context.
    """
    prompt = f"""
    You are a SEAL Research Architect. Analyze the following context and propose a specific model edit.
    
    Topic: {topic}
    Context: {context}
    
    Your goal is to identify a specific fact or relationship that is missing or could be clarified.
    Propose a clear Question/Answer pair that encapsulates this knowledge.
    
    Respond strictly in JSON format:
    {{
      "intent": "Why is this edit needed? (e.g., clarify X, add fact about Y)",
      "knowledge_gap": "What exactly was missing or unclear before?",
      "evidence": "Briefly summarize the supporting evidence from the context",
      "proposed_q": "The factual question",
      "proposed_a": "The accurate answer",
      "risk_score": 0.0 to 1.0 (Low risk = fact, High risk = opinion/changing info),
      "confidence": 0.0 to 1.0
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        
        # Robust JSON extraction
        start = text.find("{")
        end = text.rfind("}") + 1
        json_text = text[start:end]
        data = json.loads(json_text)
        
        return EditProposal(
            intent=data.get("intent", ""),
            knowledge_gap=data.get("knowledge_gap", ""),
            evidence=data.get("evidence", ""),
            proposed_q=data.get("proposed_q", ""),
            proposed_a=data.get("proposed_a", ""),
            source=topic, # Using topic as source for now
            risk_score=float(data.get("risk_score", 0.0)),
            confidence=float(data.get("confidence", 0.0))
        )
    except Exception as e:
        print(f"[Self-Editor] Edit proposal generation failed: {e}")
        return None
