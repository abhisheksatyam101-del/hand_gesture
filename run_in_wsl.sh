#!/bin/bash
export LD_LIBRARY_PATH=$(find /home/abhis/.local/lib/python3.11/site-packages/nvidia/ -name "lib" | paste -d: -s -)
python3.11 "$@"
