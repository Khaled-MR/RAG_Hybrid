"""
Ollama LLM client.

Thin wrapper that builds a RAG-aware chat prompt (with numbered context
documents) and calls Ollama. Defaults to Qwen 2.5 14B but any Ollama
model works — just change `llm_model` in config.py.
"""

from typing import Iterator, List, Optional
import ollama


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert assistant. Answer the user's question using ONLY the "
    "provided context documents.\n"
    "- Give the best, most complete and accurate answer you can, synthesizing "
    "information across the documents when relevant.\n"
    "- Lead with a direct, clear answer to the question. Then add supporting "
    "detail. Use short paragraphs or bullet points when it improves clarity.\n"
    "- Cite the documents you use inline like [1] or [2].\n"
    "- If the answer is not in the context, say you don't have enough "
    "information rather than guessing.\n"
    "- Always respond in the same language the user asked the question in."
)


class OllamaLLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "30m",
    ):
        self.client = ollama.Client(host=base_url)
        self.model = model
        # Keep the model resident in VRAM between requests so we don't pay the
        # 5-10s reload cost on every question.
        self.keep_alive = keep_alive

    def _build_messages(self, query: str, contexts: List[str], system: Optional[str]):
        context_block = "\n\n---\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        ) or "(no context retrieved)"
        return [
            {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context documents:\n\n{context_block}\n\nQuestion: {query}",
            },
        ]

    def generate(
        self,
        query: str,
        contexts: List[str],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> str:
        response = self.client.chat(
            model=self.model,
            messages=self._build_messages(query, contexts, system),
            keep_alive=self.keep_alive,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"]

    def generate_stream(
        self,
        query: str,
        contexts: List[str],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield answer text chunks as Ollama produces them."""
        stream = self.client.chat(
            model=self.model,
            messages=self._build_messages(query, contexts, system),
            keep_alive=self.keep_alive,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        for part in stream:
            chunk = part.get("message", {}).get("content", "")
            if chunk:
                yield chunk
