"""Turn finished Harbor jobs into training data: distillation trajectories and RL task lists.

Reads Harbor job directories only (via `touchstone.harbor.jobs`). No GPU, no torch, no training —
this stops at the GPU line and produces the files a training stack consumes.
"""
