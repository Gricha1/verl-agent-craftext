cd ../..
cd baselines
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

python cmdp_ppo_with_instruction.py --craftext_settings achievements_safe_sequential_defeat_monster --env_name="Craftax-Classic-Pixels-v1-Text" --num_envs=512