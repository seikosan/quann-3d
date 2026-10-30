#!/bin/bash
# CloudLM Neural Atlas Viewer
# Usage: ./start_viewer.sh

cd "$(dirname "$0")"
python3 -m http.server 8080 --bind 127.0.0.1 --directory neural-atlas
echo "Open http://127.0.0.1:8080/cloudlm_viewer.html in your browser"
