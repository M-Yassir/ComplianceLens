import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import tiktoken

    ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(ENC.encode(text))

except ImportError:

    def count_tokens(text: str) -> int:
        return round(len(text.split()) * 1.3)


from src.schemas.chunks import ChildChunk, GlossaryEntry, ParentChunk


class LegalChunker:

    def __init__(
        self,
        max_parent_tokens: int = 1200,
        target_child_tokens: int = 250,
        max_child_tokens: int = 400,
    ):
        self.max_parent_tokens = max_parent_tokens
        self.target_child_tokens = target_child_tokens
        self.max_child_tokens = max_child_tokens

    def process_file(
        self, file_path: Path, version_tag: str
    ) -> Tuple[List[ChildChunk], List[ParentChunk], List[GlossaryEntry]]:
        """Parses a version text file and returns ChildChunks, ParentChunks, and GlossaryEntries."""
        raw_text = file_path.read_text(encoding="utf-8")
        sections = self._split_into_sections(raw_text)

        all_children: List[ChildChunk] = []
        all_parents: List[ParentChunk] = []
        all_glossary: List[GlossaryEntry] = []

        for sec_num, sec_title, sec_body in sections:
            # RULE A: Definition Section (§ 7001)
            if sec_num == "7001" or "definition" in sec_title.lower():
                children, parents, glossary = self._process_definitions(
                    sec_num, sec_title, sec_body, version_tag
                )
                all_children.extend(children)
                all_parents.extend(parents)  # Will be [] under Option 2
                all_glossary.extend(glossary)

            # RULE C: Micro Section (< 500 tokens)
            elif count_tokens(sec_body) < 500:
                child, parent = self._process_micro_section(
                    sec_num, sec_title, sec_body, version_tag
                )
                all_children.append(child)
                all_parents.append(parent)

            # RULE B: Standard Sections (Parent-Child Hierarchy with Fallback)
            else:
                children, parents = self._process_standard_section(
                    sec_num, sec_title, sec_body, version_tag
                )
                all_children.extend(children)
                all_parents.extend(parents)

        # Safety Guardrail: Guarantee absolute uniqueness on child chunk IDs
        seen_ids = set()
        unique_children = []
        for c in all_children:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                unique_children.append(c)
            else:
                print(
                    f"⚠️ Warning: Suppressed duplicate chunk_id in parsing: {c.chunk_id}"
                )

        return unique_children, all_parents, all_glossary

    def _split_into_sections(self, text: str) -> List[Tuple[str, str, str]]:
        """Splits raw text by section headers (§ XXXX. Title)."""
        pattern = r"\n(?=§\s*\d+)"
        raw_sections = [s.strip() for s in re.split(pattern, text) if s.strip()]

        parsed = []
        header_regex = r"^§\s*(\d+)\.\s*([^\n]+)"

        for sec in raw_sections:
            match = re.search(header_regex, sec)
            if match:
                sec_num = match.group(1).strip()
                sec_title = match.group(2).strip()
                parsed.append((sec_num, sec_title, sec))
            else:
                # Top level text or header preamble
                parsed.append(("0000", "Preamble", sec))

        return parsed

    def _process_definitions(
        self, sec_num: str, sec_title: str, body: str, version: str
    ) -> Tuple[List[ChildChunk], List[ParentChunk], List[GlossaryEntry]]:
        """RULE A: Treats each definition as an atomic ChildChunk."""
        children, glossary = [], []

        QUOTES = r"[\"'\u201c\u201d\u2018\u2019]"
        def_pattern = rf"(?=\n\([a-z]{{1,3}}\)\s*{QUOTES})"
        defs = [d.strip() for d in re.split(def_pattern, body) if d.strip()]

        for idx, def_text in enumerate(defs):
            # SKIP header preamble line e.g. "§ 7001. Definitions."
            if not re.search(r"^\([a-z]{1,3}\)", def_text, re.IGNORECASE):
                continue

            sub_match = re.search(r"^\(([a-z]{1,3})\)", def_text, re.IGNORECASE)
            sub_item = f"({sub_match.group(1)})" if sub_match else f"({idx + 1})"

            term_match = re.search(
                rf"^\([a-z]{{1,3}}\)\s*{QUOTES}([^\"'\u201c\u201d\u2018\u2019]+){QUOTES}",
                def_text,
                re.IGNORECASE,
            )

            term_name = term_match.group(1).strip().lower() if term_match else ""
            sub_tag = sub_item.strip("()")

            chunk_id = f"v{version}_sec{sec_num}_child_{sub_tag}"
            parent_id = chunk_id
            prov_lineage = f"v{version}_sec{sec_num}_sub_{sub_tag}"

            formatted_text = f"§ {sec_num}. {sec_title}\n{def_text}"
            tokens = count_tokens(formatted_text)

            child = ChildChunk(
                chunk_id=chunk_id,
                parent_id=parent_id,
                version=version,
                section=sec_num,
                sub_item=sub_item,
                section_title=sec_title,
                provisional_lineage_id=prov_lineage,
                text=formatted_text,
                token_count=tokens,
                is_fallback_split=False,
            )
            children.append(child)

            if term_name:
                is_distinctive = (
                    len(term_name.split()) > 1
                    or term_name in ["admt", "coppa", "ccpa"]
                )
                glossary.append(
                    GlossaryEntry(
                        version=version,
                        term=term_name,
                        aliases=[term_name.upper()] if len(term_name) <= 5 else [],
                        section_citation=f"§ {sec_num}{sub_item}",
                        definition_text=def_text,
                        is_distinctive=is_distinctive,
                        source_chunk_id=chunk_id,
                    )
                )

        return children, [], glossary

    def _process_micro_section(
        self, sec_num: str, sec_title: str, body: str, version: str
    ) -> Tuple[ChildChunk, ParentChunk]:
        """RULE C: Direct passthrough for sections under 500 tokens."""
        chunk_id = f"v{version}_sec{sec_num}_child_all"
        parent_id = f"v{version}_sec{sec_num}_parent_all"
        prov_lineage = f"v{version}_sec{sec_num}_full"
        tokens = count_tokens(body)

        child = ChildChunk(
            chunk_id=chunk_id,
            parent_id=parent_id,
            version=version,
            section=sec_num,
            sub_item=None,
            section_title=sec_title,
            provisional_lineage_id=prov_lineage,
            text=body,
            token_count=tokens,
            is_fallback_split=False,
        )

        parent = ParentChunk(
            parent_id=parent_id,
            version=version,
            section=sec_num,
            sub_item=None,
            section_title=sec_title,
            provisional_lineage_id=prov_lineage,
            text=body,
            token_count=tokens,
            child_ids=[chunk_id],
        )

        return child, parent

    def _process_standard_section(
        self, sec_num: str, sec_title: str, body: str, version: str
    ) -> Tuple[List[ChildChunk], List[ParentChunk]]:
        """RULE B: Splits into parent groups (max 1200 tokens) and child units (max 400 tokens) with fallback."""
        children, parents = [], []

        # Split by top-level subsections: (a), (b), (c)
        sub_pattern = r"(?=\n\([a-z]\)\s*)"
        subsections = [s.strip() for s in re.split(sub_pattern, body) if s.strip()]

        # FALLBACK TRIGGER: Dense block without subsection structures
        if len(subsections) <= 1 and count_tokens(body) > self.max_child_tokens:
            return self._fallback_character_split(sec_num, sec_title, body, version)

        current_parent_text = ""
        current_parent_subs = []
        parent_idx = 1

        for sub in subsections:
            sub_tokens = count_tokens(sub)
            if (
                count_tokens(current_parent_text) + sub_tokens > self.max_parent_tokens
                and current_parent_text
            ):
                # Flush parent window
                p_childs, p_obj = self._build_parent_and_children(
                    sec_num,
                    sec_title,
                    current_parent_text,
                    current_parent_subs,
                    parent_idx,
                    version,
                )
                children.extend(p_childs)
                parents.append(p_obj)

                parent_idx += 1
                current_parent_text = sub
                current_parent_subs = [sub]
            else:
                current_parent_text += (
                    "\n\n" + sub if current_parent_text else sub
                )
                current_parent_subs.append(sub)

        if current_parent_text:
            p_childs, p_obj = self._build_parent_and_children(
                sec_num,
                sec_title,
                current_parent_text,
                current_parent_subs,
                parent_idx,
                version,
            )
            children.extend(p_childs)
            parents.append(p_obj)

        return children, parents

    def _build_parent_and_children(
        self,
        sec_num: str,
        sec_title: str,
        parent_text: str,
        subs: List[str],
        parent_idx: int,
        version: str,
    ) -> Tuple[List[ChildChunk], ParentChunk]:
        """Sub-divides a Parent window into leaf ChildChunks."""
        parent_id = f"v{version}_sec{sec_num}_p{parent_idx}"
        prov_lineage = f"v{version}_sec{sec_num}_p{parent_idx}"

        child_objs = []
        child_ids = []

        child_idx = 1
        for sub in subs:
            match = re.search(r"^\(([a-z0-9]+)\)", sub, re.IGNORECASE)

            # Skip creating a ChildChunk for the standalone header preamble line
            if not match and sub.startswith("§") and len(subs) > 1:
                continue

            sub_item = (
                f"({match.group(1)})" if match else f"p{parent_idx}_c{child_idx}"
            )

            child_id = f"v{version}_sec{sec_num}_p{parent_idx}_c{child_idx}"
            formatted_child_text = f"§ {sec_num}. {sec_title}\n{sub}"

            child = ChildChunk(
                chunk_id=child_id,
                parent_id=parent_id,
                version=version,
                section=sec_num,
                sub_item=sub_item,
                section_title=sec_title,
                provisional_lineage_id=f"v{version}_sec{sec_num}_sub_{sub_item.strip('()')}",
                text=formatted_child_text,
                token_count=count_tokens(formatted_child_text),
                is_fallback_split=False,
            )
            child_objs.append(child)
            child_ids.append(child_id)
            child_idx += 1

        parent = ParentChunk(
            parent_id=parent_id,
            version=version,
            section=sec_num,
            sub_item=None,
            section_title=sec_title,
            provisional_lineage_id=prov_lineage,
            text=f"§ {sec_num}. {sec_title}\n{parent_text}",
            token_count=count_tokens(parent_text),
            child_ids=child_ids,
        )

        return child_objs, parent

    def _fallback_character_split(
        self, sec_num: str, sec_title: str, body: str, version: str
    ) -> Tuple[List[ChildChunk], List[ParentChunk]]:
        """Fallback for unstructured text blocks using standard overlapping character windows."""
        words = body.split()
        chunk_words = 200
        overlap_words = 30

        children, parents = [], []
        parent_id = f"v{version}_sec{sec_num}_p_fallback"
        child_ids = []

        step = chunk_words - overlap_words
        for i in range(0, len(words), step):
            segment = " ".join(words[i : i + chunk_words])
            child_id = f"v{version}_sec{sec_num}_c_fallback_{i // step + 1}"
            formatted_text = f"§ {sec_num}. {sec_title}\n{segment}"

            child = ChildChunk(
                chunk_id=child_id,
                parent_id=parent_id,
                version=version,
                section=sec_num,
                sub_item=None,
                section_title=sec_title,
                provisional_lineage_id=f"v{version}_sec{sec_num}_fb_{i // step + 1}",
                text=formatted_text,
                token_count=count_tokens(formatted_text),
                is_fallback_split=True,
            )
            children.append(child)
            child_ids.append(child_id)

        parent = ParentChunk(
            parent_id=parent_id,
            version=version,
            section=sec_num,
            sub_item=None,
            section_title=sec_title,
            provisional_lineage_id=f"v{version}_sec{sec_num}_fb_parent",
            text=f"§ {sec_num}. {sec_title}\n{body}",
            token_count=count_tokens(body),
            child_ids=child_ids,
        )
        parents.append(parent)

        return children, parents


# =====================================================================
# CLI Persistence Execution
# =====================================================================
def run_chunker_pipeline(input_file: Path, output_dir: Path, version_tag: str):
    """Executes chunking and persists separated JSON files."""
    chunker = LegalChunker()
    children, parents, glossary = chunker.process_file(input_file, version_tag)

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"chunks_{version_tag}.json").write_text(
        json.dumps([c.model_dump() for c in children], indent=2), encoding="utf-8"
    )
    (output_dir / f"parents_{version_tag}.json").write_text(
        json.dumps([p.model_dump() for p in parents], indent=2), encoding="utf-8"
    )
    (output_dir / f"glossary_{version_tag}.json").write_text(
        json.dumps([g.model_dump() for g in glossary], indent=2), encoding="utf-8"
    )

    print(f" Chunking Complete for Version [{version_tag}]:")
    print(
        f"   • Child Chunks    : {len(children):>4} -> {output_dir / f'chunks_{version_tag}.json'}"
    )
    print(
        f"   • Parent Chunks   : {len(parents):>4} -> {output_dir / f'parents_{version_tag}.json'}"
    )
    print(
        f"   • Glossary Terms  : {len(glossary):>4} -> {output_dir / f'glossary_{version_tag}.json'}\n"
    )


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    # Execution on processed version files
    file_2026 = PROCESSED_DIR / "version_2026.txt"

    if file_2026.exists():
        run_chunker_pipeline(file_2026, PROCESSED_DIR, version_tag="2026")