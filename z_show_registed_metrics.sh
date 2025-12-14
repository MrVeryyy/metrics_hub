#!/bin/bash

python - << 'PY'
import main
from main.registry import list_metrics, get_metric

print("registered:", list_metrics())
m = get_metric("id")
print("id metric device:", m.device)
PY