#!/usr/bin/env python3
"""Recover omitted step rewards by deterministic replay of the original collection."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault('CAGED_CRAFTEXT_PATH', str(REPO_ROOT / 'caged_craftext'))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT


def observation_from_reset(observation, info) -> str:
    return str(info.get('text_render', observation))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Directory containing reward-free transitions.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--settings', default='debug_square_16x16')
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (source / 'transitions.jsonl').open(encoding='utf-8')]
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows: by_episode[int(row['episode_id'])].append(row)
    for episode in by_episode.values(): episode.sort(key=lambda row: int(row['step']))
    inverse_actions = {str(name): int(action_id) for action_id, name in ACTION_TO_TEXT.items()}
    workers: dict[int, CagedCraftextWorker] = {}
    env_kwargs = {'config_name': args.settings, 'use_debug_square_map': True,
                  'observation_type': 'ascii', 'encode_form': 'embedding'}
    tmp = output / 'transitions.jsonl.tmp'
    restored = 0
    try:
        with tmp.open('w', encoding='utf-8') as handle:
            for episode_id in sorted(by_episode):
                episode = by_episode[episode_id]
                seed = int(episode[0]['environment_worker_seed'])
                worker = workers.setdefault(seed, CagedCraftextWorker(seed=seed, env_kwargs=env_kwargs))
                initial = str(episode[0]['observation_t'])
                matches = []
                for scenario in range(3):
                    observation, info = worker.reset(scenario_idx=scenario, return_render=False)
                    if observation_from_reset(observation, info) == initial:
                        matches.append(scenario)
                if len(matches) != 1:
                    raise RuntimeError(f'episode {episode_id}: expected exactly one matching scenario, found {matches}')
                # Reset once more into the verified scenario before replaying its actions.
                observation, info = worker.reset(scenario_idx=matches[0], return_render=False)
                current = observation_from_reset(observation, info)
                for row in episode:
                    if current != str(row['observation_t']):
                        raise RuntimeError(f'episode {episode_id} step {row["step"]}: current observation mismatch')
                    action = str(row['action_t'])
                    if action not in inverse_actions:
                        raise RuntimeError(f'episode {episode_id} step {row["step"]}: invalid stored action {action!r}')
                    next_observation, reward, done, info = worker.step(inverse_actions[action], return_render=False)
                    next_text = observation_from_reset(next_observation, info)
                    if next_text != str(row['observation_t_plus_1']):
                        raise RuntimeError(f'episode {episode_id} step {row["step"]}: next observation mismatch')
                    row = dict(row)
                    row['reward'] = float(reward)
                    row['continuation'] = float(not bool(done))
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                    restored += 1
                    current = next_text
                if episode_id % 25 == 0:
                    print(json.dumps({'episode': episode_id, 'restored_transitions': restored}), flush=True)
    finally:
        for worker in workers.values():
            close = getattr(worker, 'close', None)
            if close: close()
    tmp.replace(output / 'transitions.jsonl')
    (output / 'reward_restore_metadata.json').write_text(json.dumps({
        'source': str(source), 'settings': args.settings, 'episodes': len(by_episode),
        'transitions': restored, 'replay_observation_verification': 'exact',
    }, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'ok', 'transitions': restored, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
