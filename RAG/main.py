import os
import fitz  # PyMuPDF


def parse_and_split_pdf(
    pdf_path: str, max_pages: int = 5, output_dir: str = "."
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages_to_process = min(max_pages, len(doc))

    n_minus_1_words_by_line = []
    n_words_by_line = []

    total_strikethrough_words = 0
    total_underline_words = 0
    total_normal_words = 0

    print(
        f"--- Processing first {pages_to_process} pages of '{pdf_path}' ---\n"
    )

    for page_num in range(pages_to_process):
        page = doc[page_num]

        # 1. Collect horizontal line vectors (strikethroughs and underlines)
        horizontal_lines = []
        drawings = page.get_drawings()

        for path in drawings:
            for item in path.get("items", []):
                # Line item: ('l', p1, p2)
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    # Check if line is roughly horizontal
                    if abs(p1.y - p2.y) <= 3.0:
                        y_pos = (p1.y + p2.y) / 2
                        x_min, x_max = min(p1.x, p2.x), max(p1.x, p2.x)
                        horizontal_lines.append((y_pos, x_min, x_max))

                # Thin rectangle item (often used for drawn rules): ('re', rect)
                elif item[0] == "re":
                    rect = item[1]
                    if rect.height <= 4.0:  # Thin horizontal bar
                        y_pos = (rect.y0 + rect.y1) / 2
                        horizontal_lines.append((y_pos, rect.x0, rect.x1))

        # 2. Extract words with coordinates: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        words = page.get_text("words")

        # Organise text into lines for readable output
        current_line_key = None
        line_n_minus_1 = []
        line_n = []

        for w in words:
            w_rect = fitz.Rect(w[:4])
            word_str = w[4]
            block_no, line_no = w[5], w[6]
            line_key = (page_num, block_no, line_no)

            # Insert newline when switching lines in PDF layout
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

            # --- CHECK OVERLAP AGAINST ALL HORIZONTAL LINES ---
            for line_y, x_min, x_max in horizontal_lines:
                # Calculate horizontal overlap length between word and vector line
                overlap_x = max(
                    0, min(w_rect.x1, x_max) - max(w_rect.x0, x_min)
                )
                word_width = w_rect.width

                horizontal_coverage = (
                    (overlap_x / word_width) if word_width > 0 else 0
                )

                # Only mark if the line covers at least 40% of the word's width
                if horizontal_coverage >= 0.40:
                    word_height = w_rect.height

                    # Strikethrough: Line crosses middle vertical section of word
                    if (
                        w_rect.y0 + (0.15 * word_height)
                        <= line_y
                        <= w_rect.y1 - (0.15 * word_height)
                    ):
                        is_strikethrough = True
                        break

                    # Underline: Line runs along bottom edge of word
                    elif w_rect.y1 - 2.5 <= line_y <= w_rect.y1 + 4.0:
                        is_underline = True
                        break

            # Metrics count
            if is_strikethrough:
                total_strikethrough_words += 1
            elif is_underline:
                total_underline_words += 1
            else:
                total_normal_words += 1

            # --- Extraction Rules ---
            # N-1 (Old Law): Keep Normal + Strikethrough (Ignore Underlines)
            if not is_underline:
                line_n_minus_1.append(word_str)

            # N (New Law): Keep Normal + Underline (Ignore Strikethroughs)
            if not is_strikethrough:
                line_n.append(word_str)

        # Flush remaining words on page end
        if line_n_minus_1:
            n_minus_1_words_by_line.append(" ".join(line_n_minus_1))
        if line_n:
            n_words_by_line.append(" ".join(line_n))

    # 3. Write outputs to text files
    os.makedirs(output_dir, exist_ok=True)
    path_n_minus_1 = os.path.join(output_dir, "version_N_minus_1.txt")
    path_n = os.path.join(output_dir, "version_N.txt")

    with open(path_n_minus_1, "w", encoding="utf-8") as f:
        f.write("\n".join(n_minus_1_words_by_line))

    with open(path_n, "w", encoding="utf-8") as f:
        f.write("\n".join(n_words_by_line))

    # Diagnostic output
    print("=== Extracted Word Diagnostics ===")
    print(f"Normal Words Extracted:        {total_normal_words}")
    print(f"Strikethrough Words (N-1 Only): {total_strikethrough_words}")
    print(f"Underlined Words (N Only):     {total_underline_words}\n")
    print(f"SUCCESS! Output files generated:")
    print(f" - N-1 Version (2023 Law): {os.path.abspath(path_n_minus_1)}")
    print(f" - N Version (2026 Law):   {os.path.abspath(path_n)}")


if __name__ == "__main__":
    PDF_FILE = "docs/ccpa_updates_cyber_risk_admt_appr_text.pdf"
    parse_and_split_pdf(pdf_path=PDF_FILE, max_pages=10)