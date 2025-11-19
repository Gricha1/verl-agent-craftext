# Using Subtask-Based GiGPO with Craftext

This guide shows how to use the `CraftextSubtaskEnvironmentManager` to implement subtask-based GiGPO grouping.

## Overview

The key idea is simple:
1. The LLM agent predicts a subtask in its response (e.g., `<subtask>collect_wood</subtask>`)
2. The environment manager extracts this subtask from the agent's response
3. The subtask is used as the `anchor` state instead of the raw observation
4. GiGPO groups states by these subtask labels (no changes to GiGPO code needed!)

## Implementation

### Step 1: Use CraftextSubtaskEnvironmentManager

In your training script, replace `CraftextEnvironmentManager` with `CraftextSubtaskEnvironmentManager`:

```python
from agent_system.environments.craftext_subtask_manager import CraftextSubtaskEnvironmentManager

# Instead of:
# envs = CraftextEnvironmentManager(_envs, projection_f, config)

# Use:
envs = CraftextSubtaskEnvironmentManager(_envs, projection_f, config)
val_envs = CraftextSubtaskEnvironmentManager(_val_envs, projection_f, config)
```

### Step 2: Modify Agent Prompt to Predict Subtasks

Update your agent's prompt to instruct it to predict subtasks. For example:

```
You are an agent in a crafting environment. For each action, you should:
1. Predict which subtask you're trying to complete
2. Take the action

Format your response as:
<subtask>subtask_name</subtask>
<action>your_action_here</action>

Valid subtasks: collect_wood, collect_stone, collect_iron, craft_tool, build_structure, place_block, mine_resource, explore
```

### Step 3: Extract Subtask in Projection Function

You may need to modify your projection function to extract both the subtask and the action. The subtask will be automatically extracted by `CraftextSubtaskEnvironmentManager`, but you need to make sure the action extraction still works.

Example projection function:

```python
def craftext_projection_with_subtask(actions: List[str]) -> Tuple[List[int], List[int]]:
    """
    Extract actions from agent responses that may include subtask predictions.
    """
    action_ids = []
    valids = []
    
    for action in actions:
        # Remove subtask tags if present
        action_clean = re.sub(r'<subtask>.*?</subtask>', '', action, flags=re.IGNORECASE)
        action_clean = re.sub(r'Subtask:\s*\w+', '', action_clean, flags=re.IGNORECASE)
        
        # Then do your normal action extraction
        # ... rest of your projection logic ...
    
    return action_ids, valids
```

### Step 4: That's It!

The existing GiGPO code will automatically use subtasks for grouping because:
- `CraftextSubtaskEnvironmentManager.step()` sets `anchor` to subtasks instead of observations
- `anchor_obs` in the batch will contain subtask strings
- `build_step_group()` in `gigpo/core_gigpo.py` will group by these subtask strings

## How It Works

1. **Agent generates response**: `"<subtask>collect_wood</subtask> I will chop down a tree"`
2. **Environment manager extracts subtask**: `extract_subtask_from_action()` returns `"collect_wood"`
3. **Subtask becomes anchor**: `next_observations['anchor'] = ["collect_wood"]` (instead of text render)
4. **GiGPO groups by subtask**: States with the same subtask are grouped together for advantage computation

## Customizing Subtask Extraction

You can customize the `extract_subtask_from_action()` method in `CraftextSubtaskEnvironmentManager` to:
- Support different formats (XML, JSON, prefix, etc.)
- Use LLM-based prediction
- Add more sophisticated parsing logic

Example with LLM-based prediction:

```python
def extract_subtask_from_action(self, text_action: str) -> str:
    # Use a small LLM to predict subtask
    prompt = f"Given this action: '{text_action}', predict the subtask. Valid: {self.valid_subtasks}"
    # ... LLM prediction logic ...
    return predicted_subtask
```

## Benefits

1. **No GiGPO code changes**: Works with existing `compute_gigpo_outcome_advantage()` function
2. **Semantic grouping**: Groups by task semantics rather than exact state matching
3. **Simple implementation**: Just change the environment manager class
4. **Flexible**: Easy to customize subtask extraction logic

