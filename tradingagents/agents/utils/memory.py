import chromadb
from chromadb.config import Settings
from openai import OpenAI


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.llm_provider = config["llm_provider"]
        if config["llm_provider"] == "ollama":
            self.embedding = "nomic-embed-text"
        elif config["llm_provider"] == "llamacpp":
            try:
                from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
            except Exception:
                LlamaCppEmbeddings = None
            # path to your local gguf/ggml model file (absolute recommended)
            self.embedding = config.get(
                "embedding_model_path", "models/embeddinggemma-300M-F32.gguf"
            )
            if LlamaCppEmbeddings is None:
                raise RuntimeError(
                    "LlamaCppEmbeddings not available; install llama-cpp-python and langchain_community."
                )
            # instantiate the LlamaCpp embeddings wrapper
            self.embeddings_model = LlamaCppEmbeddings(
                model_path=self.embedding,
                n_ctx=512,
            )
            self.client = None
        else:
            self.embedding = "text-embedding-3-small"

        if config["llm_provider"] != "llamacpp":
            # For ollama and others that use the OpenAI compatibility layer
            base_url = config.get("embedding_backend_url", config.get("backend_url", "http://localhost:11434/v1"))
            # ensure /v1 for ollama OpenAI compatibility
            if "localhost:11434" in base_url and not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
                base_url = base_url.rstrip("/") + "/v1"
            self.client = OpenAI(base_url=base_url)

        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)

    def get_embedding(self, text):
        """Get embedding for a text (provider-dependent)"""
        if self.llm_provider == "ollama" or self.llm_provider == "llamacpp":
            # LlamaCppEmbeddings implements embed_documents(list[str]) -> list[list[float]]
            emb = self.embeddings_model.embed_documents([text])
            return emb[0]
        else:
            # OpenAI/ollama path using OpenAI client
            response = self.client.embeddings.create(model=self.embedding, input=text)
            return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
