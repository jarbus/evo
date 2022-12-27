#!/bin/bash
if [ -z "$1" ]; then
    file="$(ls runs/*/*.log | fzf --tac)"
else 
    file="$1"
fi

echo "$file"
python ts.py "$file"
