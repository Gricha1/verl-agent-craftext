cd ../..
cd baselines
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

python ppo_cost_with_instruction.py --craftext_settings achievements_easy_relational_avoid_enemy_by_radius --env_name="Craftax-Classic-Pixels-v1-Text" --num_envs=512 \
                                    --cost_koef 0.5