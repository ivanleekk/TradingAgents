import chromadb
from chromadb.config import Settings
import time


class FinancialSituationMemory:
    def __init__(self, name, config):
        # init fields
        self.name = name
        self.config = config
        self.embeddings_model = None

        if config.get("llm_provider") == "ollama":
            # Ollama provider (no local embeddings provided here)
            self.embedding = "nomic-embed-text"
            self.embeddings_model = None

        elif config.get("llm_provider") == "llamacpp":
            try:
                from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
            except Exception:
                raise RuntimeError(
                    "LlamaCppEmbeddings not available; install llama-cpp-python and langchain_community."
                )

            # path to your local gguf/ggml model file (absolute recommended)
            self.embedding = config.get(
                "embedding_model_path", "models/nomic-embed-text-v2-moe.f32.gguf"
            )

            # instantiate the LlamaCpp embeddings wrapper with configurable device and n_gpu_layers
            llamacpp_device = config.get("llamacpp_device", "cpu")
            llamacpp_n_gpu_layers = int(config.get("llamacpp_n_gpu_layers", 0))

            try:
                self.embeddings_model = LlamaCppEmbeddings(
                    model_path=self.embedding,
                    n_ctx=int(config.get("llamacpp_n_ctx", 512)),
                    n_parts=-1,
                    seed=0,
                    f16_kv=True,
                    logits_all=False,
                    vocab_only=False,
                    use_mlock=False,
                    n_threads=int(config.get("llamacpp_n_threads", 8)),
                    n_batch=int(config.get("llamacpp_n_batch", 512)),
                    n_gpu_layers=llamacpp_n_gpu_layers,
                    verbose=False,
                    device=llamacpp_device,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize LlamaCppEmbeddings: {e}.\nEnsure the model file, device, and llama-cpp-python build (CUDA vs CPU) match your environment."
                )

        else:
            # Default to OpenAI embedding id when provider isn't local LlamaCpp
            self.embedding = "text-embedding-3-small"
            self.embeddings_model = None

        # No OpenAI fallback: require LlamaCpp if chosen. Chroma DB always created.
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get embedding for a text (provider-dependent)"""
        # LlamaCpp-only embedding path. If embeddings_model is not available, raise a helpful error.
        if getattr(self, "embeddings_model", None) is None:
            raise RuntimeError(
                "LlamaCppEmbeddings is not initialized. Ensure llama-cpp-python and langchain_community are installed, and that 'embedding_model_path' in config points to a valid gguf model."
            )

        # Try embedding; on common llama.cpp decode errors, attempt one reinit with safer defaults and retry once.
        try:
            emb = self.embeddings_model.embed_documents([text])
            return emb[0]
        except RuntimeError as e:
            msg = str(e)
            print(f"LlamaCpp embedding RuntimeError: {msg}")
            # Common recoverable error: llama_decode returned -3 (decode error). Try safer reinit and retry.
            if "llama_decode returned" in msg:
                print(
                    "Detected llama.cpp decode error; attempting reinitialization with safer defaults and retrying once..."
                )
                if self._try_reinit_with_safe_defaults():
                    try:
                        emb = self.embeddings_model.embed_documents([text])
                        return emb[0]
                    except Exception as e2:
                        raise RuntimeError(
                            "LlamaCpp embeddings failed after safe reinitialization: "
                            + str(e2)
                        )
                else:
                    raise RuntimeError(
                        "Failed to reinitialize LlamaCpp embeddings with safe defaults. Check model file, GPU memory, and llama-cpp-python build."
                    )
            # Non-decode runtime errors: surface with guidance
            raise RuntimeError(
                "LlamaCpp embeddings runtime error: "
                + msg
                + ".\nCheck that the model at 'embedding_model_path' is valid, compatible with llama-cpp-python, and that 'llamacpp_device' and 'llamacpp_n_gpu_layers' are set correctly."
            )
        except Exception as e:
            raise RuntimeError(
                "Unexpected error computing LlamaCpp embeddings: " + str(e)
            )

    def _try_reinit_with_safe_defaults(self):
        """Attempt to reinitialize the LlamaCppEmbeddings instance with safer conservative defaults.

        Returns True on success, False otherwise.
        """
        try:
            # Try to rebuild embeddings_model with CPU device and zero GPU layers and smaller batch
            from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings

            model_path = getattr(self, "embedding", None)
            if model_path is None:
                return False

            time.sleep(0.5)
            self.embeddings_model = LlamaCppEmbeddings(
                model_path=model_path,
                n_ctx=512,
                n_parts=-1,
                seed=0,
                f16_kv=True,
                logits_all=False,
                vocab_only=False,
                use_mlock=False,
                n_threads=1,
                n_batch=128,
                n_gpu_layers=0,
                verbose=False,
                device="cpu",
            )
            return True
        except Exception as e:
            print(f"Safe reinit of LlamaCppEmbeddings failed: {e}")
            return False

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


# module provides FinancialSituationMemory; example usage removed to avoid side-effects on import
