"""
Ollama LLM client.

Thin wrapper that builds a RAG-aware chat prompt (with numbered context
documents) and calls Ollama. Defaults to Qwen 2.5 14B but any Ollama
model works — just change `llm_model` in config.py.
"""

from typing import Iterator, List, Optional
import ollama


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert document assistant. Answer using ONLY the provided "
    "context documents — never use outside knowledge and never invent facts.\n"
    "Follow these rules strictly:\n"
    "1. Ground every statement in the context. If the answer is not in the "
    "context, say so explicitly (in the user's own language) instead of "
    "guessing.\n"
    "2. Give a COMPLETE, helpful answer — explain the concept properly, don't "
    "reply with just one short line. Use the detail available in the context.\n"
    "3. When the user asks about a specific item (an article/clause/section "
    "such as 'المادة 17', a term, a row, a name), quote its exact text "
    "verbatim between quotation marks, then explain it in clear words.\n"
    "4. After answering, mention any directly related items found in the "
    "context (e.g. other articles it references or that reference it) and how "
    "they connect — but only if they actually appear in the context.\n"
    "5. Cite the source snippets you used inline like [1] or [2].\n"
    "6. Reply entirely in the same language the user used in their question.\n"
    "7. Prefer the document's own wording over paraphrase when accuracy matters."
)


class OllamaLLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "30m",
        num_ctx: int = 4096,
    ):
        self.client = ollama.Client(host=base_url)
        self.model = model
        # Keep the model resident in VRAM between requests so we don't pay the
        # 5-10s reload cost on every question.
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx

    @staticmethod
    def _detect_language(text: str) -> str:
        """Pick the answer language from the question, not the documents.

        With a mixed Arabic/English corpus the retrieved context is often in a
        different language than the question; without this the model drifts to
        the documents' language. We decide from the QUESTION and state it
        explicitly at the end of the prompt (strongest position).
        """
        # Arabic questions routinely embed long English terms ("transformer",
        # "LLM"), so a simple arabic>latin count misfires. An English question
        # essentially never contains Arabic letters — so any real Arabic
        # presence means the user is asking in Arabic.
        arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
        return "Arabic" if arabic >= 2 else "English"

    def _build_messages(self, query: str, contexts: List[str], system: Optional[str]):
        context_block = "\n\n---\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        ) or "(no context retrieved)"
        lang = self._detect_language(query)
        return [
            {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context documents:\n\n{context_block}\n\n"
                    f"Question: {query}\n\n"
                    f"Write a complete, helpful answer in {lang}. Do NOT "
                    f"translate the question or add notes about language — start "
                    f"directly with the answer and explain it properly. (You may "
                    f"keep a short quoted term in its original language.)"
                ),
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
                "num_ctx": self.num_ctx,
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
                "num_ctx": self.num_ctx,
            },
        )
        for part in stream:
            chunk = part.get("message", {}).get("content", "")
            if chunk:
                yield chunk
