from pathlib import Path
import pymupdf
import pytest

from src.parsers.pdf_vector_parser import extract_horizontal_lines, parse_and_split_pdf


# =====================================================================
# Synthetic PDF Fixtures
# =====================================================================
@pytest.fixture
def create_synthetic_pdf(tmp_path):
    """Factory fixture to generate programmatic PDFs with custom text and vector lines."""

    def _generate_pdf(file_name: str = "test_redline.pdf") -> Path:
        pdf_path = tmp_path / file_name
        doc = pymupdf.open()

        # Page 1: Standard Redlined Document
        page1 = doc.new_page(width=612, height=792)  # Standard Letter size

        # 1. Running Header (Inside header margin y < 70)
        page1.insert_text((50, 40), "CONFIDENTIAL - HEADER TEXT", fontsize=10)

        # 2. Normal Body Text (Middle of page)
        page1.insert_text((50, 150), "Section 7000. Core rules apply.", fontsize=12)

        # 3. Deleted Text + Strikethrough Line
        # Text baseline at y=200, rect approx y0=188, y1=202 (height 14)
        page1.insert_text((50, 200), "OldDeletedWord", fontsize=12)
        shape1 = page1.new_shape()
        shape1.draw_line(pymupdf.Point(48, 195), pymupdf.Point(150, 195))  # Mid-height strikethrough line
        shape1.finish(color=(1, 0, 0), width=1)
        shape1.commit()

        # 4. Added Text + Underline Line
        page1.insert_text((200, 200), "NewInsertedWord", fontsize=12)
        shape2 = page1.new_shape()
        shape2.draw_line(pymupdf.Point(198, 202), pymupdf.Point(310, 202))  # Bottom-edge underline line
        shape2.finish(color=(0, 0, 1), width=1)
        shape2.commit()

        # 5. Running Footer (Inside footer margin y > 792 - 80 = 712)
        page1.insert_text((50, 750), "Page 1 of 2 - FOOTER TEXT", fontsize=10)

        # Page 2: Second Page for max_pages testing
        page2 = doc.new_page(width=612, height=792)
        page2.insert_text((50, 150), "Section 7001. Second page content.", fontsize=12)

        doc.save(pdf_path)
        doc.close()
        return pdf_path

    return _generate_pdf


# =====================================================================
# Unit Tests: extract_horizontal_lines
# =====================================================================
def test_extract_horizontal_lines_margin_filtering(create_synthetic_pdf):
    """Verifies vector line extraction filters out header/footer lines and isolates body lines."""
    pdf_path = create_synthetic_pdf("lines_test.pdf")
    doc = pymupdf.open(pdf_path)
    page = doc[0]

    # Add a line in top header margin (y=30) and body area (y=200)
    shape = page.new_shape()
    shape.draw_line(pymupdf.Point(10, 30), pymupdf.Point(100, 30))  # Top margin
    shape.draw_line(pymupdf.Point(10, 200), pymupdf.Point(100, 200))  # Body
    shape.finish()
    shape.commit()

    lines = extract_horizontal_lines(page, header_margin=70.0, footer_margin=80.0)

    # Should only capture lines outside the margins
    y_positions = [line[0] for line in lines]
    assert all(y >= 70.0 for y in y_positions)
    assert any(abs(y - 200) < 5.0 for y in y_positions)
    assert not any(abs(y - 30) < 5.0 for y in y_positions)
    doc.close()


# =====================================================================
# Unit Tests: parse_and_split_pdf
# =====================================================================
def test_parse_pdf_raises_file_not_found():
    """Asserts FileNotFoundError is raised when PDF path does not exist."""
    with pytest.raises(FileNotFoundError):
        parse_and_split_pdf("non_existent_file.pdf")


def test_redline_text_separation(create_synthetic_pdf):
    """Verifies Old Law (N-1) gets Strikethrough and New Law (N) gets Underlines."""
    pdf_path = create_synthetic_pdf("redline_sep.pdf")

    result = parse_and_split_pdf(pdf_path, max_pages=1)

    text_old = result["text_n_minus_1"]
    text_new = result["text_n"]

    # Shared normal text should appear in both
    assert "Section 7000. Core rules apply." in text_old
    assert "Section 7000. Core rules apply." in text_new

    # Strikethrough text should ONLY be in Old Law (N-1)
    assert "OldDeletedWord" in text_old
    assert "OldDeletedWord" not in text_new

    # Underlined text should ONLY be in New Law (N)
    assert "NewInsertedWord" in text_new
    assert "NewInsertedWord" not in text_old


def test_margin_header_footer_stripping(create_synthetic_pdf):
    """Verifies running headers and footers are omitted from extracted texts."""
    pdf_path = create_synthetic_pdf("margin_strip.pdf")

    result = parse_and_split_pdf(pdf_path, max_pages=1)

    text_old = result["text_n_minus_1"]
    text_new = result["text_n"]

    # Header and footer content should be completely stripped
    assert "HEADER TEXT" not in text_old
    assert "HEADER TEXT" not in text_new
    assert "FOOTER TEXT" not in text_old
    assert "FOOTER TEXT" not in text_new

    # Stripped word metric should be tracked
    assert result["metrics"]["margin_words_stripped"] > 0


def test_max_pages_limit(create_synthetic_pdf):
    """Verifies processing is constrained when max_pages parameter is set."""
    pdf_path = create_synthetic_pdf("multi_page.pdf")

    result_page1 = parse_and_split_pdf(pdf_path, max_pages=1)
    result_all = parse_and_split_pdf(pdf_path, max_pages=None)

    assert result_page1["metrics"]["pages_processed"] == 1
    assert result_all["metrics"]["pages_processed"] == 2

    assert "Second page content" not in result_page1["text_n"]
    assert "Second page content" in result_all["text_n"]


def test_output_file_persistence(create_synthetic_pdf, tmp_path):
    """Verifies output text files are written to output_dir when provided."""
    pdf_path = create_synthetic_pdf("persistence_test.pdf")
    out_dir = tmp_path / "processed_output"

    parse_and_split_pdf(pdf_path, output_dir=out_dir)

    file_2023 = out_dir / "version_2023.txt"
    file_2026 = out_dir / "version_2026.txt"

    assert file_2023.exists()
    assert file_2026.exists()
    assert "OldDeletedWord" in file_2023.read_text(encoding="utf-8")
    assert "NewInsertedWord" in file_2026.read_text(encoding="utf-8")