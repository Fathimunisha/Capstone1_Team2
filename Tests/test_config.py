from regulatory_compliance.retrievers.hybrid_retrievers import HybridRetriever
from regulatory_compliance.agents.rag_agents import RAGAgent

query = "RBI?"


# Retrieve context

retriever = HybridRetriever(top_k=5)


documents = retriever.search(query)


print("Retrieved Documents:", len(documents))


# Generate answer

agent = RAGAgent()


answer = agent.generate_answer(query, documents)


print("\n")
print("=" * 60)

print(answer)

print("=" * 60)
