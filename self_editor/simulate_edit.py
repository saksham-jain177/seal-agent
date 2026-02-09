from typing import Tuple, Optional
from langchain_ollama.chat_models import ChatOllama
from self_editor.propose_edit import EditProposal
from inference_adapter import generate_with_adapter

def simulate_edit(proposal: EditProposal, llm: ChatOllama) -> Tuple[bool, str]:
    """
    Performs 'Shadow Inference' to verify if the proposed edit is impactful.
    Queries the 3B Learner model (via adapter) to check for current knowledge.
    
    Returns (simulation_passed, reason)
    """
    # 1. Query the specialized 3B model (Learner) without the new edit
    print(f"[Simulator] Querying 3B Learner model for: {proposal.proposed_q}")
    
    # We use high temperature for 'diversity check' or low for 'fact check'
    base_response = generate_with_adapter(
        f"Question: {proposal.proposed_q}\nAnswer:", 
        max_length=200, 
        temperature=0.1
    )
    
    if not base_response:
        # Fallback/Error handling if adapter fails to load
        base_response = "I don't know."
        
    print(f"[Simulator] 3B Learner response: \"{base_response[:100]}...\"")
    
    # 2. Use the LLM to compare current knowledge vs proposed knowledge
    comparison_prompt = f"""
    You are a SEAL Simulation Sceptic (8B Model). 
    Your job is to evaluate whether a "Student" (3B Model) needs to learn a new fact.
    
    STUDENT'S HALLUCINATION/RESPONSE: {base_response}
    PROPOSED CORRECT KNOWLEDGE: {proposal.proposed_a}
    CONTEXTUAL QUESTION: {proposal.proposed_q}
    
    CRITICAL RULE:
    1. FACTUAL CORRECTION = IMPACTFUL. If the Student's response contains WRONG facts, wrong companies (e.g. saying Google instead of Moonshot), or outdated metrics, the edit MUST be marked as IMPACTFUL.
    2. ABSTENTION = IMPACTFUL. If the Student says "I don't know" or gives a generic filler, the edit is IMPACTFUL.
    3. TRIVIAL ONLY IF: The Student's current response is 100% factually identical to the Proposed Knowledge, including specific metrics, dates, and entities.
    
    DANGER: Do NOT use your own internal knowledge. Only compare the "STUDENT'S HALLUCINATION" string provided above with the "PROPOSED CORRECT KNOWLEDGE". 
    If the Student said "Google" and the Edit says "Moonshot", DO NOT say it's trivial just because YOU know it's Moonshot. The Student needs the update!

    Respond strictly in JSON:
    {{
      "assessment": "TRIVIAL" | "IMPACTFUL" | "INVALID",
      "reason": "Identify if this is a CORRECTION, an EXTENSION, or a REPEATED FACT."
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
