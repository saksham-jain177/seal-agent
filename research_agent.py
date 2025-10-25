import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
from self_editor.generate_selfedit import generate_self_edit
from self_editor.validate import validate_self_edit
from self_editor.save import append_self_edit
from self_editor.review_selfedits import review_entry_with_llm


def main():
    # 1. Load Environment
    load_dotenv()
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("TAVILY_API_KEY not found in .env file.")
    print("Environment loaded.")

    # 2. Initialize Tools
    search = TavilySearch(max_results=3)
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)

    # 3. Ask User Question
    question = input("\n> Enter your research question: ")
    print("Searching web...")
    search_results = search.invoke({"query": question})

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
    print("\n[Phase 2] Generating self-edit from the answer...")
    try:
        search_snapshot = json.dumps(search_results, ensure_ascii=False)
    except Exception:
        try:
            search_snapshot = str(search_results)
        except Exception:
            search_snapshot = ""
    raw_edit = generate_self_edit(topic=question, context=search_snapshot)
    if not raw_edit:
        print("[Self-Editor] No structured output from generator. Self-edit not created.")
    else:
        try:
            cleaned = validate_self_edit(raw_edit)
        except Exception as e:
            print(f"[Self-Editor] Validation failed: {e}")
            print("Raw output for debugging:", raw_edit)
        else:
            out_path, appended = append_self_edit(cleaned)
            if appended:
                print(f"[Self-Editor] Saved new self-edit to: {out_path}")
                try:
                    review_result = review_entry_with_llm(llm, cleaned)
                    print("\n[Self-Editor Review] ------------------------")
                    print(json.dumps(review_result, indent=2))
                    print("[Self-Editor Review] ------------------------\n")
                except Exception as e:
                    print(f"[Self-Editor Review] Failed during review: {e}")
            else:
                print("[Self-Editor] Duplicate detected — not appended.")

if __name__ == "__main__":
    main()
