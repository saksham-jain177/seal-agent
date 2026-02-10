import json
import hashlib
from typing import Tuple, Optional, Dict
from langchain_ollama.chat_models import ChatOllama
from self_editor.propose_edit import EditProposal
from inference_adapter import generate_with_adapter

class SimulationResult:
    def __init__(
        self,
        proposal_id: str,
        simulation_passed: bool,
        delta_score: float,
        confidence_at_time: float,
        base_output_hash: str,
        simulated_output_hash: str,
        notes: str
    ):
        self.proposal_id = proposal_id
        self.simulation_passed = simulation_passed
        self.delta_score = delta_score
        self.confidence_at_time = confidence_at_time
        self.base_output_hash = base_output_hash
        self.simulated_output_hash = simulated_output_hash
        self.notes = notes

    def to_dict(self) -> Dict:
        return {
            "proposal_id": self.proposal_id,
            "simulation_passed": self.simulation_passed,
            "delta_score": self.delta_score,
            "confidence_at_time": self.confidence_at_time,
            "base_output_hash": self.base_output_hash,
            "simulated_output_hash": self.simulated_output_hash,
            "notes": self.notes
        }

def simulate_edit(proposal: EditProposal, llm: ChatOllama) -> SimulationResult:
    """
    Performs 'Shadow Inference' with Delta Tracing.
    Verifies if the proposed edit causes a verifiable behavior change.
    """
    # 1. Shadow Inference: Query the student model
    print(f"[Simulator] Shadow Inference for: {proposal.proposed_q}")
    
    base_response = generate_with_adapter(
        f"Question: {proposal.proposed_q}\nAnswer:", 
        max_length=200, 
        temperature=0.1
    )
    
    if not base_response:
        base_response = "I don't know."
        
    base_hash = hashlib.sha256(base_response.encode()).hexdigest()
    sim_hash = hashlib.sha256(proposal.proposed_a.encode()).hexdigest()

    # 2. Comparison & Auditing
    comparison_prompt = f"""
    You are a SEAL Simulation Auditor. 
    Compare a Student's (3B) response against a Proposers (8B) knowledge.
    
    STUDENT OUTPUT: {base_response}
    PROPOSED KNOWLEDGE: {proposal.proposed_a}
    EVIDENCE PROVIDED: {proposal.evidence}
    INTENT: {proposal.intent}
    
    REJECT_IF:
    - No Output Delta: Student already knows the fact precisely.
    - Hallucinated Evidence: The Proposed Knowledge is NOT supported by the Evidence provided.
    - High Risk / Low Confidence: The edit feels speculative or poorly grounded.
    
    Respond strictly in JSON:
    {{
      "simulation_passed": bool,
      "delta_score": 0.0 to 1.0 (How much new valid knowledge is gained?),
      "confidence": 0.0 to 1.0,
      "notes": "Reasoning for pass/fail"
    }}
    """
    
    try:
        eval_resp = llm.invoke(comparison_prompt).content.strip()
        start = eval_resp.find("{")
        end = eval_resp.rfind("}") + 1
        json_text = eval_resp[start:end]
        eval_data = json.loads(json_text)
        
        passed = eval_data.get("simulation_passed", False)
        
        # Enforce global contract rules
        if proposal.risk == "high" or proposal.confidence < 0.4:
            passed = False
            eval_data["notes"] = f"Global Gate: Rejected due to {proposal.risk} risk / low confidence."

        # If hashes are identical, no behavior change is possible
        if base_hash == sim_hash:
            passed = False
            eval_data["notes"] = "Trivial: Proposal is identical to current model behavior."

        return SimulationResult(
            proposal_id=proposal.id,
            simulation_passed=passed,
            delta_score=float(eval_data.get("delta_score", 0.0)),
            confidence_at_time=float(eval_data.get("confidence", 0.0)),
            base_output_hash=base_hash,
            simulated_output_hash=sim_hash,
            notes=eval_data.get("notes", "Unknown reasoning")
        )
            
    except Exception as e:
        print(f"[Simulator] Error: {e}")
        return SimulationResult(
            proposal_id=proposal.id,
            simulation_passed=False,
            delta_score=0.0,
            confidence_at_time=0.0,
            base_output_hash=base_hash,
            simulated_output_hash=sim_hash,
            notes=f"Internal Error: {e}"
        )
