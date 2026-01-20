#!/bin/bash
# Usage: ./split_pdf.sh input.pdf [pages_per_chunk]
# Splits PDF into chunks of N pages each (default: 10)

INPUT="$1"
PAGES_PER_CHUNK="${2:-10}"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 input.pdf [pages_per_chunk]"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: File not found: $INPUT"
    exit 1
fi

BASENAME=$(basename "$INPUT" .pdf)
OUTDIR="./chunks_${BASENAME}"

mkdir -p "$OUTDIR"

# Get total pages
TOTAL=$(qpdf --show-npages "$INPUT")
echo "Total pages: $TOTAL"

# Split into chunks
START=1
CHUNK=1
while [ $START -le $TOTAL ]; do
    END=$((START + PAGES_PER_CHUNK - 1))
    [ $END -gt $TOTAL ] && END=$TOTAL

    qpdf "$INPUT" --pages . $START-$END -- "$OUTDIR/${BASENAME}_chunk_$(printf %03d $CHUNK).pdf"

    START=$((END + 1))
    CHUNK=$((CHUNK + 1))
done

echo "Split into $((CHUNK - 1)) chunks in $OUTDIR/"
