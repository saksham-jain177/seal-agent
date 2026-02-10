import json
import uuid
from datetime import datetime
from typing import Dict, Optional, List
from langchain_ollama.chat_models import ChatOllama

# SEAL Semantics: Edit Proposal Schema (v2 Alignment)
# Focuses on behavior change, auditability, and clear delta expectations.

class EditProposal:
    def __init__(
        self,
        intent: str,
        evidence: List[str],
        expected_delta: str,
        proposed_q: str,
        proposed_a: str,
        confidence: float,
        risk: str,  # low|medium|high
        knowledge_gap: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.created_at = datetime.utcnow().isoformat() + "Z"
        self.intent = intent
        self.evidence = evidence
        self.expected_delta = expected_delta
        self.proposed_q = proposed_q
        self.proposed_a = proposed_a
        self.confidence = confidence
        self.risk = risk
        self.knowledge_gap = knowledge_gap
        self.simulation_passed = False
        self.applied = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "intent": self.intent,
            "evidence": self.evidence,
            "expected_delta": self.expected_delta,
            "proposed_q": self.proposed_q,
            "proposed_a": self.proposed_a,
            "confidence": self.confidence,
            "risk": self.risk,
            "knowledge_gap": self.knowledge_gap,
            "simulation_passed": self.simulation_passed,
            "applied": self.applied
        }

def propose_edit(topic: str, context: str, llm: ChatOllama) -> Optional[EditProposal]:
    """
    Generates a structured Edit Proposal based on a research topic and context.
    Strictly adheres to SEAL-semantic contract.
    """
    prompt = f"""
    You are a SEAL Research Architect. Analyze the following context and propose a specific model edit.
    
    Topic: {topic}
    Context: {context}
    
    Your goal is to identify a specific fact or relationship that is missing or could be clarified.
    You must propose an edit that results in an explicit behavior change.
    
    Respond strictly in JSON format:
    {{
      "intent": "Explicit behavior change goal (e.g., 'Correct attribution of X to Y')",
      "evidence": ["url1", "Snippet summary"],
      "expected_delta": "Exactly how the model's output should change after this edit.",
      "proposed_q": "The factual question to train on",
      "proposed_a": "The accurate answer to train on",
      "knowledge_gap": "Analysis of what the base model currently lacks",
      "confidence": 0.0 to 1.0,
      "risk": "low|medium|high"
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
            evidence=data.get("evidence", []),
            expected_delta=data.get("expected_delta", ""),
            proposed_q=data.get("proposed_q", ""),
            proposed_a=data.get("proposed_a", ""),
            confidence=float(data.get("confidence", 0.0)),
            risk=data.get("risk", "medium"),
            knowledge_gap=data.get("knowledge_gap", "")
        )
    except Exception as e:
        print(f"[Self-Editor] Edit proposal generation failed: {e}")
        return None
