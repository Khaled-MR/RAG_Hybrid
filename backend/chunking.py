
import re
from typing import List

# Strong structural boundaries. Each match starts a new segment so a whole
# article / clause / section / heading stays together in its own chunk — which
# is what lets the model quote it verbatim and answer "what is article 17"
# precisely. Generic enough to leave plain prose untouched.
_STRUCTURE_RE = re.compile(
    r"(?m)^(?=\s*(?:"
    r"#{1,6}\s+"                                                    # markdown heading
    r"|(?:المادة|مادة|الفصل|الباب|الجزء|القسم|البند|بند)\s+[\d٠-٩]"  # Arabic article/section
    r"|(?:Article|Section|Clause|Chapter|Part|Rule)\s+\d"            # English article/section
    r"))",
    re.IGNORECASE,
)


class RecursiveChunker:
    DEFAULT_SEPARATORS = [
        "\n\n\n",
        "\n\n",
        "\n",
        "۔ ",  
        ". ",
        "؟ ",   
        "? ",
        "! ",
        "؛ ",  
        "; ",
        "، ",   
        ", ",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        separators: List[str] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        # First split on structural boundaries (articles/sections/headings) so
        # each one becomes its own chunk; then sub-split only the long ones.
        result: List[str] = []
        for segment in self._structural_split(text):
            raw: List[str] = []
            self._split(segment, self.separators, raw)
            result.extend(self._apply_overlap(raw))
        return [c for c in result if c.strip()]

    def _structural_split(self, text: str) -> List[str]:
        """Split at article/section/heading starts, keeping the heading text."""
        starts = [m.start() for m in _STRUCTURE_RE.finditer(text)]
        if len(starts) < 2:
            return [text]
        if starts[0] != 0:
            starts = [0] + starts
        segments = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            seg = text[start:end].strip()
            if seg:
                segments.append(seg)
        return segments

    def _split(self, text: str, separators: List[str], out: List[str]) -> None:
        if len(text) <= self.chunk_size:
            if text.strip():
                out.append(text)
            return

        separator = separators[-1]
        remaining = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining = []
                break
            if sep in text:
                separator = sep
                remaining = separators[i + 1:]
                break

        if separator == "":
            for i in range(0, len(text), self.chunk_size):
                piece = text[i:i + self.chunk_size]
                if piece.strip():
                    out.append(piece)
            return

        splits = text.split(separator)
        pieces = [
            s + separator if i < len(splits) - 1 else s
            for i, s in enumerate(splits)
        ]

        buffer = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if buffer.strip():
                    out.append(buffer)
                    buffer = ""
                self._split(piece, remaining, out)
            elif len(buffer) + len(piece) <= self.chunk_size:
                buffer += piece
            else:
                if buffer.strip():
                    out.append(buffer)
                buffer = piece

        if buffer.strip():
            out.append(buffer)

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return [c.strip() for c in chunks if c.strip()]

        result = [chunks[0].strip()]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            result.append((overlap + chunks[i]).strip())
        return [c for c in result if c]
