"""M2.7 attack-harness module — see m2_7/README.md.

This package exists so that `from m2_7.m2_7_common import …` works
from the parent harness's `run_all.py`. The M2.7 drivers themselves
(`run_static_attacks`, `run_token_attacks`, `run_hidden_state_attacks`,
`run_ima_embedrow_attacks`) remain runnable as `python m2_7/<name>.py …`
per their docstrings.
"""
