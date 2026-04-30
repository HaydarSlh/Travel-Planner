# Tool: rag_retrieve
# Input validated by schemas.tools.RAGRetrieveInput before execution.
# Rewrites the user query for vector search (cheap model), then calls rag.store.similarity_search.
# Returns ranked destination chunks with source metadata for the agent to reason over.
# On failure, returns a structured error dict — never raises into the agent loop.
