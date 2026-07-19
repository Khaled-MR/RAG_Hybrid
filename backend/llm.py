"""
Ollama LLM client.

Thin wrapper that builds a RAG-aware chat prompt (with numbered context
documents) and calls Ollama. Defaults to Qwen 2.5 14B but any Ollama
model works — just change `llm_model` in config.py.
"""

from typing import Iterator, List, Optional
import ollama


DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous legal/document assistant. You answer STRICTLY from the "
    "provided context passages — never from outside knowledge.\n"
    "\n"
    "HARD RULES (a legal reviewer will reject the answer otherwise):\n"
    "1. Never invent or guess article numbers, dates, durations, penalties, or "
    "legal conclusions. Every such fact MUST come from the context, quoted "
    "verbatim.\n"
    "2. For every legal statement you make, cite its basis inline using the "
    "source labels shown in the context, e.g. [المصدر: <الملف> - المادة N], and "
    "quote the exact wording of the relevant text between «...».\n"
    "3. If the specific basis (the article text) is NOT present in the context, "
    "do NOT assert «يجوز/لا يجوز/يلزم». Instead say the basis is not available "
    "in the provided documents, and phrase the answer conditionally "
    "(«يعتمد ذلك على...», «وفقًا للمعلومات المتاحة...»).\n"
    "4. Prefer cautious, professional phrasing over absolute certainty unless a "
    "quoted text directly and fully supports the claim.\n"
    "\n"
    "ANSWER FORMAT — use these sections, but TRANSLATE the section headings "
    "into the SAME language as the question (Arabic headings for an Arabic "
    "question, English headings for an English question):\n"
    "الإجابة: <إجابة مباشرة لكن محتاطة>\n"
    "الأساس القانوني: <لكل نقطة: المادة N ونصها «...» و[المصدر: ...] — أو صراحةً "
    "'لا يوجد سند صريح في المستندات المتاحة'>\n"
    "التسبيب: <لماذا تؤدي النصوص إلى هذه الإجابة>\n"
    "درجة الثقة: عالية | متوسطة | منخفضة\n"
    "يحتاج مراجعة بشرية: نعم | لا\n"
    "\n"
    "CONFIDENCE RULE: use 'عالية' only if EVERY statement is backed by a quoted "
    "text that actually appears in the context. If any claim lacks an explicit "
    "quoted basis, set درجة الثقة = منخفضة and يحتاج مراجعة بشرية = نعم.\n"
    "Reply entirely in the same language the user used in their question."
)

def _format_history(history: Optional[List[dict]], max_turns: int = 6) -> str:
    """Render recent chat turns as 'User:/Assistant:' lines for prompts."""
    if not history:
        return ""
    turns = [h for h in history if h.get("role") in ("user", "assistant") and h.get("content")]
    turns = turns[-max_turns:]
    lines = []
    for h in turns:
        who = "User" if h["role"] == "user" else "Assistant"
        text = " ".join(str(h["content"]).split())
        if len(text) > 500:
            text = text[:500] + "…"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


# Rewrites a colloquial / messy question into a clean formal query for retrieval.
REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's question into a single clear, formal query optimized "
    "for searching a legal/document knowledge base. Keep the original meaning "
    "and preserve any article/law numbers, names, and key terms; expand vague "
    "colloquial wording into precise terms. Do NOT answer the question. Output "
    "ONLY the rewritten query as one line, in the same language as the input."
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

    def rewrite(self, question: str, history: Optional[List[dict]] = None) -> str:
        """Rewrite a (possibly colloquial or follow-up) question into a formal,
        STANDALONE retrieval query, resolving references to the chat history.

        Falls back to the original question on any error so retrieval never
        breaks because of the rewrite step.
        """
        try:
            lang = self._detect_language(question)
            hist_block = _format_history(history)
            user = (
                (f"Conversation so far:\n{hist_block}\n\n" if hist_block else "")
                + f"Follow-up question: {question}\n\n"
                + f"Rewrite it as a single STANDALONE query in {lang} ONLY "
                + "(resolve any pronouns/references using the conversation; do "
                + "not translate to another language)."
            )
            resp = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                keep_alive=self.keep_alive,
                options={"temperature": 0.0, "num_predict": 120, "num_ctx": 2048},
            )
            out = (resp["message"]["content"] or "").strip().strip('"').strip()
            return out or question
        except Exception:
            return question

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

    def _build_messages(self, query: str, contexts: List[str], system: Optional[str],
                        history: Optional[List[dict]] = None):
        context_block = "\n\n---\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        ) or "(no context retrieved)"
        lang = self._detect_language(query)
        hist_block = _format_history(history)
        hist_part = (
            f"Conversation so far (for context — the answer must still come only "
            f"from the documents):\n{hist_block}\n\n" if hist_block else ""
        )
        return [
            {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{hist_part}"
                    f"Context documents:\n\n{context_block}\n\n"
                    f"Question: {query}\n\n"
                    f"Write a complete, helpful answer in {lang}, INCLUDING the "
                    f"section headings (write them in {lang}). Do NOT translate "
                    f"the question or add notes about language — start directly "
                    f"with the answer and explain it properly. (You may keep a "
                    f"short quoted legal text in its original language.)"
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
        history: Optional[List[dict]] = None,
    ) -> str:
        response = self.client.chat(
            model=self.model,
            messages=self._build_messages(query, contexts, system, history),
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
        history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """Yield answer text chunks as Ollama produces them."""
        stream = self.client.chat(
            model=self.model,
            messages=self._build_messages(query, contexts, system, history),
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
