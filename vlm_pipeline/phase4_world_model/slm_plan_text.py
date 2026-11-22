"""Prompt text and answer parsing of the phase 4 SLMs over Dreamer 4 latents, one source for training
(train_slm_reason_act_d4.py), evaluation and the veRL agent loop (phase3_verl/craftext_integration/plan_manager.py).
No torch import: the veRL rollout worker loads it."""
import re

SYSTEM = ("You play Crafter from the game screen. Think about the best course of action step by step inside "
          "<reasoning></reasoning>, then give the action inside <action></action>")
SYSTEM_PLAN = ("You play Crafter from the game screen. Think about the best course of action step by step inside "
               "<reasoning></reasoning>, then give your next {} actions, comma-separated, inside <plan></plan>")
NAMES = ("Noop", "Move West", "Move East", "Move North", "Move South", "Do", "Sleep", "Place Stone", "Place Table",
         "Place Furnace", "Place Plant", "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
         "Make Wood Sword", "Make Stone Sword", "Make Iron Sword")   # ZeroHero ACTIONS, the Craftax-Classic order
EOS = 2


def parse_action(text):
    m = re.search(r"<action>(.*?)</action>", text)
    return NAMES.index(m.group(1).strip()) if m and m.group(1).strip() in NAMES else None


def parse_plan(text):
    """List of action indices inside <plan></plan> (None for a name that is not an action), or None without the tags."""
    m = re.search(r"<plan>(.*?)</plan>", text, re.S)
    if m is None:
        return None
    return [NAMES.index(s.strip()) if s.strip() in NAMES else None for s in m.group(1).split(",")]


def prompt_parts(task, plan):
    """(pre, post) text around the latent window: 'System:\\n<system>\\nUser:\\n' | latents + readouts | post."""
    return "System:\n" + (SYSTEM_PLAN.format(plan) if plan else SYSTEM) + "\nUser:\n", f"\nTask: {task}\nAssistant:\n"
