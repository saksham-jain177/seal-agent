from typing import Tuple
from langchain_ollama.chat_models import ChatOllama
from self_editor.propose_edit import EditProposal

def simulate_edit(proposal: EditProposal, llm: ChatOllama) -> Tuple[bool, str]:
    """
    Performs 'Shadow Inference' to verify if the proposed edit is impactful.
    Compares the base model's current knowledge with the proposed change.
    
    Returns (simulation_passed, reason)
    """
    # 1. Query the model without the edit
    print(f"[Simulator] Querying base model for: {proposal.proposed_q}")
    base_response = llm.invoke(proposal.proposed_q).content.strip()
    print(f"[Simulator] Base model response: \"{base_response[:100]}...\"")
    
    # 2. Use the LLM to compare current knowledge vs proposed knowledge
    comparison_prompt = f"""
    You are a SEAL Simulation Sceptic. Your job is to prevent the model from ignoring NEW or GROUNDED knowledge.
    
    Question: {proposal.proposed_q}
    Model's Current Response (Pre-Edit): {base_response}
    Proposed Learning (Edit): {proposal.proposed_a}
    
    Task:
    Evaluate if the 'Proposed Learning' actually adds NEW, SPECIFIC, or GROUNDED information.
    
    CRITICAL GROUNDING RULE:
    If the 'Pre-Edit' response mentions the correct facts but sounds like a 'high-probability guess' or is generic, and the 'Edit' contains precise details/citations from the paper, mark it as IMPACTFUL. We want to ground the model in real data even if its 'hunch' was close.
    
    Guidelines:
    - Mark TRIVIAL ONLY if the 'Pre-Edit' response is already perfectly accurate, detailed, and contains the same specific metrics/names in a grounded way.
    - If there is ANY doubt about the model's certainty, mark it IMPACTFUL.
    
    Respond strictly in JSON:
    {{
      "assessment": "TRIVIAL" | "IMPACTFUL" | "INVALID",
      "reason": "Explain exactly why this is new knowledge or a grounded correction."
    }}
    """
    
    try:
        import json
        eval_resp = llm.invoke(comparison_prompt).content.strip()
        start = eval_resp.find("{")
        end = eval_resp.rfind("}") + 1
        json_text = eval_resp[start:end]
        eval_data = json.loads(json_text)
        
        assessment = eval_data.get("assessment", "INVALID")
        reason = eval_data.get("reason", "Unknown")
        
        if assessment == "IMPACTFUL":
            print(f"[Simulator] SUCCESS: Impactful knowledge detected. Gain: {reason}")
            return True, reason
        elif assessment == "TRIVIAL":
            print(f"[Simulator] REJECTED: Information already present in base weights.")
            return False, f"Trivial: No information gain ({reason})"
        else:
            print(f"[Simulator] REJECTED: Edit deemed invalid. Reason: {reason}")
            return False, f"Invalid: {reason}"
            
    except Exception as e:
        print(f"[Simulator] Error during simulation: {e}")
        return False, f"Simulation error: {e}"
