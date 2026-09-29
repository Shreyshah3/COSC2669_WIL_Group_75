print("STEP 1: Python script started")

from langchain_ollama import ChatOllama

print("STEP 2: LangChain imported")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("STEP 3: Ollama connection created")

response = llm.invoke(
    "What is a bank FAQ? Explain in one short sentence."
)

print("STEP 4: Response received")
print(response.content)

print("STEP 5: Test finished")