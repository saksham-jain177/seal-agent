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
    You are a SEAL Simulation Sceptic. Your job is to prevent the model from ignoring NEW knowledge.
    
    Question: {proposal.proposed_q}
    Model's Current Response (Pre-Edit): {base_response}
    Proposed Learning (Edit): {proposal.proposed_a}
    
    Task:
    Evaluate if the 'Proposed Learning' actually adds NEW, SPECIFIC, or CORRECTED information that is missing or wrong in the 'Pre-Edit' response.
    
    Guidelines:
    - If the 'Pre-Edit' response contains generic fluff or hallucinations and the 'Edit' contains specific metrics/names (like Claude 4.6, GPT-5.3), it is IMPACTFUL.
    - If the 'Pre-Edit' response says "I don't know" or has no knowledge of the specific versions/facts, it is IMPACTFUL.
    - Only mark TRIVIAL if the 'Pre-Edit' response ALREADY contains the EXACT same specific facts, versions, and metrics.
    
    Respond strictly in JSON:
    {{
      "assessment": "TRIVIAL" | "IMPACTFUL" | "INVALID",
      "reason": "Explain exactly what NEW knowledge is gained or why it's a perfect match."
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
