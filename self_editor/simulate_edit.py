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
    
    # 2. Use the LLM to compare current knowledge vs proposed knowledge
    comparison_prompt = f"""
    You are a SEAL Simulation Engine. Compare the Model's current response with the Proposed Answer.
    
    Question: {proposal.proposed_q}
    Model's Current Response: {base_response}
    Proposed Correct Answer: {proposal.proposed_a}
    
    Task:
    Check if the Model already knows this fact correctly.
    - If the Model's response is already factual and matches the Proposed Answer, the edit is 'TRIVIAL'.
    - If the Model's response is incorrect, outdated, or it admits ignorance, the edit is 'IMPACTFUL'.
    - If the Proposed Answer is contradicted by well-known facts (hallucination), the edit is 'INVALID'.
    
    Respond strictly in JSON:
    {{
      "assessment": "TRIVIAL" | "IMPACTFUL" | "INVALID",
      "reason": "Brief explanation"
    }}
    """
    
    try:
        eval_resp = llm.invoke(comparison_prompt).content.strip()
        start = eval_resp.find("{")
        end = eval_resp.rfind("}") + 1
        data = json_text = eval_resp[start:end]
        import json
        eval_data = json.loads(json_text)
        
        assessment = eval_data.get("assessment", "INVALID")
        reason = eval_data.get("reason", "Unknown")
        
        if assessment == "IMPACTFUL":
            print(f"[Simulator] SUCCESS: Edit is impactful. Reason: {reason}")
            return True, reason
        elif assessment == "TRIVIAL":
            print(f"[Simulator] REJECTED: Edit is trivial. Model already knows this.")
            return False, "Trivial: Knowledge already exists."
        else:
            print(f"[Simulator] REJECTED: Edit is invalid or hallucinated. Reason: {reason}")
            return False, f"Invalid: {reason}"
            
    except Exception as e:
        print(f"[Simulator] Error during simulation: {e}")
        return False, f"Simulation error: {e}"
