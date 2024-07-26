#!/bin/bash

nsys profile --stats=true -f true -o profile/mmha_after_1024_1_625 \
python3.8 01-many-mmha.py

ncu --nvtx --nvtx-exclude "beforeLoop" --set detailed \
    -f -o profile/mmha_1024_1_after_false_2128 python3.8 01-many-mmha.py
