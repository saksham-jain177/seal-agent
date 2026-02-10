import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

from self_editor.propose_edit import propose_edit
from self_editor.simulate_edit import simulate_edit
from self_editor.save import append_edit_proposal, init_ledger_if_needed
from inference_adapter import is_adapter_available

CACHE_PATH = "data/research_cache.json"

def get_cached_results(query: str) -> Optional[str]:
    """Retrieves cached results if they exist and are not older than 24h."""
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        
        q_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()
        entry = cache.get(q_hash)
        if entry:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            if datetime.utcnow() - timestamp < timedelta(hours=24):
                return entry["results"]
    except:
        pass
    return None

def save_to_cache(query: str, results: str):
    """Saves results to the query cache."""
    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except:
            pass
    
    q_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()
    cache[q_hash] = {
        "query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results
    }
    
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def perform_research(query: str, truth_source: str, llm: ChatOllama) -> str:
    """Handles the retrieval and summarization of truth data."""
    cached = get_cached_results(query)
    if cached:
        print("[INFO] Loading from Research Cache...")
        return cached

    print(f"Searching {truth_source.upper()}...")
    if truth_source == "arxiv":
        import arxiv
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        results = [f"Title: {r.title}\nAbstract: {r.summary}\nURL: {r.entry_id}" for r in search.results()]
        search_results = "\n\n".join(results)
    elif truth_source == "tavily" and os.getenv("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch
        search_tool = TavilySearch(max_results=3)
        search_results = str(search_tool.invoke(query))
    else:
        from langchain_community.tools import DuckDuckGoSearchRun
        search_tool = DuckDuckGoSearchRun()
        search_results = str(search_tool.invoke(query))
    
    save_to_cache(query, search_results)
    return search_results

def run_audit_loop(query: str, search_results: str, llm: ChatOllama, truth_source: str, depth: int = 0):
    """Orchestrates the SEAL audit pipeline, supporting recursive identity anchoring."""
    if depth > 1: # Prevent infinite recursion
        print("[SEAL] Max recursion depth reached. Skipping.")
        return

    # 1. Decision (Consensus Generation)
    prompt = ChatPromptTemplate.from_template("""
    Context: {context}
    Query: {query}
    Provide a concise, factual answer based ONLY on the context.
    """)
    chain = prompt | llm
    response = chain.invoke({"context": search_results, "query": query})
    
    ans = response.content if hasattr(response, "content") else response
    print("\n--- RESEARCH SUMMARY ---")
    print(ans)
    
    # 2. Audit Initiation
    print("\n[SEAL] Starting Audit Pipeline...")
    
    with open("data/edit_ledger.json", "r") as f:
        ledger = json.load(f)
    if ledger.get("applied_edits", 0) >= ledger.get("budget", 50):
        print("[SEAL] Budget Exceeded. No capacity for new edits.")
        return

    proposal = propose_edit(query, search_results, llm)
    if not proposal: return

    print(f"[SEAL] Proposing Edit: {proposal.intent}")
    sim_result = simulate_edit(proposal, llm)
    
    if sim_result.simulation_passed:
        print(f"✅ PASSED Audit: Delta Score {sim_result.delta_score}")
        path, appended = append_edit_proposal(proposal.to_dict(), sim_result.to_dict())
        if appended: print(f"[SEAL] Audit committed to: {path}")
    else:
        print(f"❌ REJECTED Audit: {sim_result.notes}")
        
        # Phase 8: Automated Identity Recovery
        if "Identity Guard" in sim_result.notes:
            # Extract subject from notes or use proposer reasoning
            subject_query = f"What is the primary subject of the rejected edit '{proposal.intent}'? Respond with just the name."
            subject = llm.invoke(subject_query).content.strip()
            
            print(f"\n[SEAL] 🔄 IDENTITY LOOP: Piercing identity hallucination for '{subject}'...")
            anchor_query = f"What is {subject} and what are its key technical features?"
            anchor_research = perform_research(anchor_query, truth_source, llm)
            run_audit_loop(anchor_query, anchor_research, llm, truth_source, depth + 1)

def main():
    load_dotenv()
    init_ledger_if_needed(budget=100)
    
    truth_source = os.getenv("TRUTH_SOURCE", "ddg").lower()
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)

    print(f"[INFO] SEAL Agent active. Mode: Research & Audit (Systematic Grounding v8).")

    query = input("\n> Research Query: ")
    search_results = perform_research(query, truth_source, llm)
    run_audit_loop(query, search_results, llm, truth_source)

if __name__ == "__main__":
    main()
