#!/bin/bash
# Usage: ./merge_summaries.sh output.md summaries_dir/*.md
# Combines extracted markdown summaries into a single file

OUTPUT="$1"
shift

if [ -z "$OUTPUT" ]; then
    echo "Usage: $0 output.md summaries_dir/*.md"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Error: No input files provided"
    exit 1
fi

BASENAME=$(basename "$OUTPUT" _summary.md)

echo "# $BASENAME - Extracted Summary" > "$OUTPUT"
echo "" >> "$OUTPUT"
echo "_Generated: $(date)_" >> "$OUTPUT"
echo "" >> "$OUTPUT"
echo "---" >> "$OUTPUT"
echo "" >> "$OUTPUT"

CHUNK_NUM=1
for chunk in "$@"; do
    if [ -f "$chunk" ]; then
        echo "## Part $CHUNK_NUM" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        cat "$chunk" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        echo "---" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        CHUNK_NUM=$((CHUNK_NUM + 1))
    fi
done

echo "_Total parts: $((CHUNK_NUM - 1))_" >> "$OUTPUT"
echo "Merged $((CHUNK_NUM - 1)) parts into $OUTPUT"
