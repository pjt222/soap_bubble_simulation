#!/bin/bash
# Usage: ./extract_pdf.sh input.pdf [pages_per_chunk]
# Splits PDF, extracts content via Claude CLI, merges results
#
# This script is idempotent - re-running skips already-processed chunks

set -e

INPUT="$1"
PAGES_PER_CHUNK="${2:-10}"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 input.pdf [pages_per_chunk]"
    echo "  pages_per_chunk: Number of pages per chunk (default: 10)"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: File not found: $INPUT"
    exit 1
fi

BASENAME=$(basename "$INPUT" .pdf)
CHUNKS_DIR="./chunks_${BASENAME}"
SUMMARIES_DIR="./summaries_${BASENAME}"
FINAL_OUTPUT="./${BASENAME}_summary.md"

# Step 1: Split PDF
echo "=== Splitting $INPUT into $PAGES_PER_CHUNK-page chunks ==="
./split_pdf.sh "$INPUT" "$PAGES_PER_CHUNK"

mkdir -p "$SUMMARIES_DIR"

# Step 2: Extract from each chunk via Claude CLI
echo "=== Extracting content from chunks ==="
PROCESSED=0
SKIPPED=0
FAILED=0

for chunk in "$CHUNKS_DIR"/*.pdf; do
    CHUNK_NAME=$(basename "$chunk" .pdf)
    OUT_FILE="$SUMMARIES_DIR/${CHUNK_NAME}.md"

    if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
        echo "Skipping $CHUNK_NAME (already exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Build prompt with file path included
    PROMPT="Read the PDF file $chunk and extract for a soap bubble simulation project:
1. **Key Concepts**: Main ideas, definitions, terminology
2. **Equations/Formulas**: Mathematical content (use LaTeX notation)
3. **Relevance**: Connection to thin-film interference, drainage, rendering
4. **Citations**: Notable references

Output as concise markdown."

    echo "Processing $CHUNK_NAME..."
    if claude -p "$PROMPT" > "$OUT_FILE" 2>/dev/null; then
        PROCESSED=$((PROCESSED + 1))
    else
        echo "Warning: Failed to process $CHUNK_NAME"
        rm -f "$OUT_FILE"  # Remove empty/failed output
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Extraction complete: $PROCESSED processed, $SKIPPED skipped, $FAILED failed"

# Step 3: Merge summaries
if ls "$SUMMARIES_DIR"/*.md 1>/dev/null 2>&1; then
    echo ""
    echo "=== Merging summaries ==="
    ./merge_summaries.sh "$FINAL_OUTPUT" "$SUMMARIES_DIR"/*.md
    echo ""
    echo "=== Done! Output: $FINAL_OUTPUT ==="
else
    echo ""
    echo "Warning: No summaries to merge"
fi
