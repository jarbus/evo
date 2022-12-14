#!/bin/bash
file="$(ls runs/*.log | fzf --tac)"
echo "$file"
python ts.py "$file"
