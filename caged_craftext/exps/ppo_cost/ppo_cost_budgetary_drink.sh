cd ../..
cd baselines
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

python ppo_cost_with_instruction.py --craftext_settings achievements_safe_budget_drink --env_name="Craftax-Classic-Pixels-v1-Text" --num_envs=512 \
                                    --cost_koef 0.5