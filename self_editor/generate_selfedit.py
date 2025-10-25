import json
from langchain_ollama.chat_models import ChatOllama

def generate_self_edit(topic: str, context: str = "") -> dict:
    """
    Given a topic and optional web context, ask the local LLM to produce a self-edit (Q/A pair).
    """
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)

    prompt = f"""
You are a research assistant designed to generate a structured 'self-edit' for model improvement.

Topic: {topic}

Relevant context (if any): {context}

Create a concise, factual Q&A pair that represents new knowledge learned from this information.
Respond ONLY in strict JSON format like this:
{{
  "question": "<clear, factual question>",
  "answer": "<accurate answer>",
  "source": "<URL or 'summary' if no URL>"
}}
    """

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()

        # Try parsing JSON; fall back if the model adds extra text
        start = text.find("{")
        end = text.rfind("}") + 1
        json_text = text[start:end]

        data = json.loads(json_text)
        assert "question" in data and "answer" in data, "Missing keys in generated JSON."
        return data

    except Exception as e:
        print(f"[Error] Failed to generate or parse self-edit: {e}")
        print("Raw output:\n", text)
        return None

if __name__ == "__main__":
    topic = input("Enter topic for self-edit generation: ")
    result = generate_self_edit(topic)
    print("\nGenerated Self-Edit:")
    print(json.dumps(result, indent=2))
