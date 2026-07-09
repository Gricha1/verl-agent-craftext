"""AlfWorld actor prompt: one token = one admissible command."""

ALFWORLD_SINGLE_TOKEN_ACTION_TEMPLATE = """
Your goal is to complete the following task:
**TASK:** {task_description}

This is what you currently see:
{current_observation}

Reply with exactly ONE token — your chosen action (no tags, no explanation):
{action_legend}
"""
