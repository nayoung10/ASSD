#!/bin/bash

# Define the directory to search in
search_dir="summaries/"

# Use find to locate all _metainfo files and then process each file
find "$search_dir" -type f -name "_metainfo" | while read -r file; do
  # Use sed to replace "AGN" with "ASSD" in each file
  sed -i 's/AGN/ASSD/g' "$file"
done
