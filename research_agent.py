import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
from self_editor.propose_edit import propose_edit
from self_editor.simulate_edit import simulate_edit
from self_editor.save import append_edit_proposal
from inference_adapter import is_adapter_available, generate_with_adapter
from langchain_community.tools import DuckDuckGoSearchRun


def main():
    # 1. Load Environment
    load_dotenv()
    # 2. Initialize Tools
    truth_source = os.getenv("TRUTH_SOURCE", "ddg").lower()
    
    if truth_source == "tavily":
        if not os.getenv("TAVILY_API_KEY"):
            print("[WARNING] TAVILY_API_KEY not found. Falling back to DuckDuckGo.")
            search_tool = DuckDuckGoSearchRun()
            truth_source = "ddg"
        else:
            search_tool = TavilySearch(max_results=3)
    else:
        search_tool = DuckDuckGoSearchRun()
        truth_source = "ddg"

    print(f"[INFO] Using {truth_source.upper()} as the truth source.")
    
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)
    
    # Check if fine-tuned adapter is available
    use_adapter = is_adapter_available()
    if use_adapter:
        print("[INFO] Fine-tuned adapter detected! Will use it for enhanced self-edit generation.")
    else:
        print("[INFO] No fine-tuned adapter found. Using base model for self-edit generation.")

    # 3. Ask User Question
    question = input("\n> Enter your research question: ")
    print(f"Searching web via {truth_source.upper()}...")
    
    if truth_source == "tavily":
        search_results = search_tool.invoke({"query": question})
    else:
        # DDG returns a string directly
        search_results = search_tool.invoke(question)

    # 4. Prepare Prompt
    prompt = ChatPromptTemplate.from_template("""
You are an intelligent research assistant.
You have access to the following search results:
{context}

Using these results, write a clear, accurate, and concise answer to the user's question:
{question}
""")

    chain = prompt | llm

    # 5. Generate Answer
    response = chain.invoke({"context": search_results, "question": question})
    print("\n--- FINAL RESPONSE ---")
    print(response.content if hasattr(response, "content") else response)
    
    # SEAL Phase: Edit Proposal & Simulation
    print("\n[SEAL] Initiating Edit Proposal phase...")
    
    # 1. Check Learning Budget
    ledger_path = "data/edit_ledger.json"
    if os.path.exists(ledger_path):
        with open(ledger_path, "r") as f:
            ledger = json.load(f)
        if ledger.get("total_edits_proposed", 0) >= ledger.get("current_budget", 50):
            print("[SEAL] Learning budget exceeded. No new edits will be proposed.")
            return

    try:
        search_snapshot = json.dumps(search_results, ensure_ascii=False)
    except Exception:
        search_snapshot = str(search_results)
    
    # 2. Propose Edit
    proposal = propose_edit(question, search_snapshot, llm)
    
    if proposal:
        print(f"[SEAL] Edit Proposed: {proposal.intent}")
        
        # 3. Simulate Edit (Shadow Inference)
        passed, reason = simulate_edit(proposal, llm)
        proposal.simulation_passed = passed
        
        if passed:
            print(f"[SEAL] Simulation PASSED. Fact is impactful.")
            # 4. Record to Ledger
            out_path, appended = append_edit_proposal(proposal.to_dict())
            if appended:
                print(f"[SEAL] Edit committed to proposal ledger: {out_path}")
            else:
                print("[SEAL] Duplicate edit detected. Not committed.")
        else:
            print(f"[SEAL] Simulation FAILED: {reason}")
            # Track failures in ledger
            if os.path.exists(ledger_path):
                with open(ledger_path, "r") as f:
                    ledger = json.load(f)
                ledger["total_edits_rejected"] = ledger.get("total_edits_rejected", 0) + 1
                with open(ledger_path, "w") as f:
                    json.dump(ledger, f, indent=2)
    else:
        print("[SEAL] Failed to generate edit proposal.")

if __name__ == "__main__":
    main()
