# veRL visual-SLM plugin

This directory is the compatibility boundary for the CE-trained 5M Crafter
SLM. It is not a standalone RL implementation.

`materialize_ce_checkpoint.py` produces a new, immutable-input-derived HF
checkpoint with the CNN vision adapter registered in the model. The four
reserved visual markers replace the CE soft-token prefix without changing the
text-token positions. Missing pixel data raises an error.

The remaining launch contract is equally strict: the veRL async rollout server
and the vector CrafText agent loop must both pass `pixel_values` for every
response and its replay item. Do not use vLLM's text-only Qwen3 implementation
for this checkpoint. Until the custom veRL async server is connected, no SLM RL
run is valid to launch.
