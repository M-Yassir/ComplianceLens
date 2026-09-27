import json
from collections import Counter
from pathlib import Path
import pytest

from src.chunkers.legal_chunker import LegalChunker


# =====================================================================
# Fixtures
# =====================================================================
@pytest.fixture
def sample_definition_text():
    return """§ 7001. Definitions.
(a) “Agency” means the California Privacy Protection Agency.
(b) “Alternative Opt-out Link” means the link required by Civil Code section 1798.100, subdivisions (a) and
(b), and specified in these regulations.
(c) “CCPA” means the California Consumer Privacy Act.
"""


@pytest.fixture
def sample_standard_text():
    return """§ 7002. Restrictions on the Collection and Use of Personal Information.
(a) A business's collection, use, retention, and/or sharing of a consumer's personal information shall be reasonably necessary and proportionate to achieve the purpose(s) for which the personal information was collected.
(b) A business shall obtain explicit consent before collecting sensitive personal information.
"""


@pytest.fixture
def chunker():
    return LegalChunker()


# =====================================================================
# Unit Tests
# =====================================================================
def test_definition_chunking_option_2(chunker, tmp_path):
    """Rule A: Definitions should serve as their own parents (no duplicate ParentChunk output)."""
    text_file = tmp_path / "version_test.txt"
    text_file.write_text(
        """§ 7001. Definitions.
(a) “Agency” means the California Privacy Protection Agency.
(b) “CCPA” means the California Consumer Privacy Act.
""",
        encoding="utf-8",
    )

    children, parents, glossary = chunker.process_file(text_file, version_tag="2026")

    # Option 2 assertions
    assert len(children) == 2
    assert len(parents) == 0  # Parents list is kept empty for definitions
    assert len(glossary) == 2

    # Verify self-referential parent_id contract
    for child in children:
        assert child.parent_id == child.chunk_id
        assert child.is_fallback_split is False


def test_quote_boundary_parsing(chunker, tmp_path):
    """Verifies that mid-paragraph citations (e.g. 'subdivisions (a) and (b)') are NOT misparsed as sub-items."""
    text_file = tmp_path / "version_test.txt"
    text_file.write_text(
        """§ 7001. Definitions.
(a) “Agency” means the agency under section 1798.100, subdivisions (a) and
(b), and specified in these regulations.
(b) “CCPA” means the California Consumer Privacy Act.
""",
        encoding="utf-8",
    )

    children, _, _ = chunker.process_file(text_file, version_tag="2026")
    
    # Should yield exactly 2 chunks, not 3
    assert len(children) == 2
    assert children[0].sub_item == "(a)"
    assert children[1].sub_item == "(b)"


def test_standard_section_hierarchy(chunker, sample_standard_text):
    """Rule B: Standard sections generate distinct Parent and Child chunks."""
    sec_num = "7002"
    sec_title = "Restrictions on the Collection and Use of Personal Information."

    # Call _process_standard_section directly to isolate Rule B testing from Rule C (Micro Section)
    children, parents = chunker._process_standard_section(
        sec_num, sec_title, sample_standard_text, version="2026"
    )

    assert len(children) == 2
    assert len(parents) == 1

    parent = parents[0]
    assert parent.parent_id == "v2026_sec7002_p1"
    assert len(parent.child_ids) == 2
    assert children[0].parent_id == parent.parent_id

def test_micro_section_routing(chunker, tmp_path, sample_standard_text):
    """Rule C: Sections under 500 tokens route to _process_micro_section as a single passthrough."""
    text_file = tmp_path / "version_test.txt"
    text_file.write_text(sample_standard_text, encoding="utf-8")

    children, parents, glossary = chunker.process_file(text_file, version_tag="2026")

    # Under Rule C (< 500 tokens), process_file returns 1 passthrough child and 1 parent
    assert len(children) == 1
    assert len(parents) == 1
    assert len(glossary) == 0

    child = children[0]
    parent = parents[0]

    assert child.chunk_id == "v2026_sec7002_child_all"
    assert parent.parent_id == "v2026_sec7002_parent_all"
    assert child.sub_item is None
    assert child.parent_id == parent.parent_id
    assert child.token_count < 500
    assert child.is_fallback_split is False

def test_fallback_split_flag(chunker, tmp_path):
    """Verifies that unstructured dense blocks set is_fallback_split = True."""
    dense_text = "§ 7099. Dense Unstructured Rules.\n" + ("Lorem ipsum dolor sit amet. " * 100)
    text_file = tmp_path / "version_test.txt"
    text_file.write_text(dense_text, encoding="utf-8")

    children, parents, _ = chunker.process_file(text_file, version_tag="2026")

    assert len(children) > 0
    assert children[0].is_fallback_split is True


# =====================================================================
# Disk Artifact Integration Test (Converted Duplicate Diagnostic)
# =====================================================================
def test_chunks_2026_json_uniqueness():
    """Reads data/processed/chunks_2026.json and asserts zero duplicate chunk_ids."""
    chunks_file = Path("data/processed/chunks_2026.json")

    if not chunks_file.exists():
        pytest.skip(f"Skipping artifact test: {chunks_file} does not exist.")

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    ids = [c["chunk_id"] for c in chunks]

    counts = Counter(ids)
    duplicates = {chunk_id: count for chunk_id, count in counts.items() if count > 1}

    # Failure report detailing duplicates if any exist
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate chunk IDs: {duplicates}"