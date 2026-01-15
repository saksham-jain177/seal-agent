from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)
print(llm.invoke("Give me the latest news about tariffs by Donald Trump"))
