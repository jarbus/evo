#!/bin/bash
python ts.py $(ls runs/*.log | fzf --tac)   
