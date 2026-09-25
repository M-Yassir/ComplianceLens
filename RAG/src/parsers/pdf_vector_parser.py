import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pymupdf


def extract_horizontal_lines(
    page: pymupdf.Page,
    header_margin: float = 50.0,
    footer_margin: float = 50.0,
) -> list[Tuple[float, float, float]]:
    """Extracts horizontal line vector coordinates (strikethroughs/underlines) from a PDF page,

    ignoring vector graphics in the page margins.
    """
    horizontal_lines = []
    page_height = page.rect.height
    drawings = page.get_drawings()

    for path in drawings:
        for item in path.get("items", []):
            # Standard line vector: ('l', point1, point2)
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) <= 3.0:  # Horizontal line tolerance
                    y_pos = (p1.y + p2.y) / 2
                    # Skip lines located inside header or footer margins
                    if y_pos < header_margin or y_pos > (page_height - footer_margin):
                        continue
                    x_min, x_max = min(p1.x, p2.x), max(p1.x, p2.x)
                    horizontal_lines.append((y_pos, x_min, x_max))

            # Thin rectangle rule: ('re', rect)
            elif item[0] == "re":
                rect = item[1]
                if rect.height <= 4.0:  # Horizontal rule thickness
                    y_pos = (rect.y0 + rect.y1) / 2
                    if y_pos < header_margin or y_pos > (page_height - footer_margin):
                        continue
                    horizontal_lines.append((y_pos, rect.x0, rect.x1))

    return horizontal_lines


def parse_and_split_pdf(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
    max_pages: Optional[int] = None,
    coverage_threshold: float = 0.40,
    header_margin: float = 70.0,
    footer_margin: float = 80.0,
) -> Dict[str, Any]:
    """Parses a redlined statutory PDF into Version N-1 (Old) and Version N (New) texts using

    vector-coordinate overlap checks and margin-based header/footer removal.

    Args:
        pdf_path: Path to the raw input PDF file.
        output_dir: Optional directory to save output .txt files.
        max_pages: Maximum number of pages to process (None processes full document).
        coverage_threshold: Ratio of word width covered by line to count as
          strikethrough/underline.
        header_margin: Vertical top margin (in points) to strip headers.
        footer_margin: Vertical bottom margin (in points) to strip footers.

    Returns:
        Dict containing extracted text strings and execution metrics.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages_to_process = (
        min(max_pages, len(doc)) if max_pages is not None else len(doc)
    )

    n_minus_1_words_by_line = []
    n_words_by_line = []

    total_strikethrough = 0
    total_underline = 0
    total_normal = 0
    total_stripped_margin_words = 0

    current_line_key = None
    line_n_minus_1 = []
    line_n = []

    for page_num in range(pages_to_process):
        page = doc[page_num]
        page_height = page.rect.height

        horizontal_lines = extract_horizontal_lines(
            page, header_margin=header_margin, footer_margin=footer_margin
        )
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)

        for w in words:
            w_rect = pymupdf.Rect(w[:4])
            word_str = w[4]
            block_no, line_no = w[5], w[6]

            # Ignore running headers and footers in vertical margins
            if w_rect.y0 < header_margin or w_rect.y1 > (page_height - footer_margin):
                total_stripped_margin_words += 1
                continue

            line_key = (page_num, block_no, line_no)

            # Flush buffered words when moving to a new visual line
            if current_line_key is not None and line_key != current_line_key:
                if line_n_minus_1:
                    n_minus_1_words_by_line.append(" ".join(line_n_minus_1))
                    line_n_minus_1 = []
                if line_n:
                    n_words_by_line.append(" ".join(line_n))
                    line_n = []

            current_line_key = line_key
            is_strikethrough = False
            is_underline = False

            # Check word bounding box against horizontal vector lines
            for line_y, x_min, x_max in horizontal_lines:
                overlap_x = max(0, min(w_rect.x1, x_max) - max(w_rect.x0, x_min))
                word_width = w_rect.width
                horizontal_coverage = (overlap_x / word_width) if word_width > 0 else 0

                if horizontal_coverage >= coverage_threshold:
                    word_height = w_rect.height

                    # Strikethrough check: Line passes through middle y-bounds
                    if (
                        w_rect.y0 + (0.15 * word_height)
                        <= line_y
                        <= w_rect.y1 - (0.15 * word_height)
                    ):
                        is_strikethrough = True
                        break

                    # Underline check: Line runs along bottom y-edge
                    elif w_rect.y1 - 2.5 <= line_y <= w_rect.y1 + 4.0:
                        is_underline = True
                        break

            # Metrics collection
            if is_strikethrough:
                total_strikethrough += 1
            elif is_underline:
                total_underline += 1
            else:
                total_normal += 1

            # Version N-1 (Old Law): Keep Normal + Strikethrough
            if not is_underline:
                line_n_minus_1.append(word_str)

            # Version N (New Law): Keep Normal + Underline
            if not is_strikethrough:
                line_n.append(word_str)

    # Final buffer flush at document completion
    if line_n_minus_1:
        n_minus_1_words_by_line.append(" ".join(line_n_minus_1))
    if line_n:
        n_words_by_line.append(" ".join(line_n))

    text_n_minus_1 = "\n".join(n_minus_1_words_by_line)
    text_n = "\n".join(n_words_by_line)

    # Optional file output persistence
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        path_n_minus_1 = out_path / "version_2023.txt"
        path_n = out_path / "version_2026.txt"

        path_n_minus_1.write_text(text_n_minus_1, encoding="utf-8")
        path_n.write_text(text_n, encoding="utf-8")

    return {
        "text_n_minus_1": text_n_minus_1,
        "text_n": text_n,
        "metrics": {
            "pages_processed": pages_to_process,
            "normal_words": total_normal,
            "strikethrough_words": total_strikethrough,
            "underline_words": total_underline,
            "margin_words_stripped": total_stripped_margin_words,
        },
    }


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    INPUT_PDF = (
        PROJECT_ROOT / "data" / "raw" / "ccpa_updates_cyber_risk_admt_appr_text.pdf"
    )
    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

    if INPUT_PDF.exists():
        results = parse_and_split_pdf(
            pdf_path=INPUT_PDF, output_dir=OUTPUT_DIR, max_pages=None
        )
        print("=== Parser Run Successful (using pymupdf) ===")
        print(f"Pages Processed:        {results['metrics']['pages_processed']}")
        print(f"Normal Words:           {results['metrics']['normal_words']}")
        print(f"Strikethrough:          {results['metrics']['strikethrough_words']}")
        print(f"Underlines:             {results['metrics']['underline_words']}")
        print(f"Header/Footer Stripped: {results['metrics']['margin_words_stripped']}")
    else:
        print(f"Test PDF not found at: {INPUT_PDF}")