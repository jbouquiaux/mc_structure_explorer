#!/bin/bash

# Check if a directory argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

TARGET_DIR="$1"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

# Directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for json_file in "$TARGET_DIR"/dataset_*.json; do
    [ -e "$json_file" ] || continue
    html_file="${json_file%.json}.html"
    cat "$SCRIPT_DIR/chemiscope_standalone.html" "$json_file" > "$html_file"
    open "$html_file"
    sleep 1
done
