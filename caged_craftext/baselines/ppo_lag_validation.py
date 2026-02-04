import wandb
import argparse
import os
import sys
import textwrap
from PIL import Image, ImageDraw, ImageFont
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import pygame
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import comet_ml
from craftax.craftax.renderer import render_craftax_pixels as render
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
)
from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
import orbax.checkpoint as ocp
sys.path.append("./models")
from baselines.models.actor_critic import ActorCriticConv
from models.actor_critic_with_text_constraints import (
    ActorCriticConvWithBERTCMDP
)
from models.actor_critic_with_text import (
    ActorCriticConvWithBERT
)
import imageio
from craftax.craftax_env import make_craftax_env_from_name

import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap

os.environ["SDL_VIDEODRIVER"] = "dummy"

try:
    font = ImageFont.truetype("arial.ttf", 30)
except IOError:
    font = ImageFont.load_default()

# Установим стиль
sns.set(style="whitegrid")
colors = {
    'reward': '#A3D2CA',       # мягкий бирюзовый
    'cost': '#F6BD60',         # теплый желто-оранжевый
    'success_rate': '#84A59D'  # спокойный серо-зеленый
}


def add_text_to_image(image, text, right_text=None):
    """Add text to an image - left side and right side."""
    # Обработка основного текста (слева)
    text_to_list = text.split()
    left_text = ""
    for i in range(0, len(text_to_list), 6):
        left_text += " ".join(text_to_list[i:i+6])
        left_text += "\n"
    
    # Создаем основное изображение
    img_pil = Image.fromarray(image.astype(np.uint8))
    
    # Определяем размеры для правого текста
    right_text_width = 200  # Ширина области для правого текста
    new_width = img_pil.width + right_text_width
    
    # Создаем новое изображение с областью для текста справа
    img_with_text = Image.new('RGB', (new_width, img_pil.height), color=(255, 255, 255))
    img_with_text.paste(img_pil, (0, 0))
    
    draw = ImageDraw.Draw(img_with_text)
    
    # Добавляем основной текст слева
    draw.text((10, 5), left_text, font=font, fill=(0, 0, 0))
    
    # Добавляем информацию о награде и стоимости справа
    if right_text:
        # Разделяем правый текст на строки
        right_lines = right_text.split('\n')
        y_position = 5
        for line in right_lines:
            draw.text((img_pil.width + 10, y_position), line, font=font, fill=(0, 0, 0))
            y_position += 15  # Межстрочный интервал
    
    return np.array(img_with_text)

"""
def add_text_to_image(image, text):
    #Add text to an image.
    text_to_list = text.split()
    text = ""
    for i in range(0, len(text_to_list), 6):
        text += " ".join(text_to_list[i:i+6])
        text += "\n"
        
    img_pil = Image.fromarray(image.astype(np.uint8))
    img_with_text = Image.new('RGB', (img_pil.width, img_pil.height + 50), color=(255, 255, 255))
    img_with_text.paste(img_pil, (0, 50))

    draw = ImageDraw.Draw(img_with_text)
    draw.text((10, 5), text, font=font, fill=(0, 0, 0))

    return np.array(img_with_text)
"""

from PIL import Image

class CraftaxRenderer:
    def __init__(self, env, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.frames = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
        )
        if self.env.environment_key == 1:
            env_render = render
        else:
            env_render = render_classic
        self._render = jax.jit(env_render, static_argnums=(1,))

    def render(self, env_state):
        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        image = Image.fromarray(np.array(pixels).astype(np.uint8))
        self.frames.append(image)

    def render_to_image(self, env_state):
        """Render the environment state to an image array and resize it to 256x256."""
        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)
        
        # Convert pixels to image and resize to 256x256
        image = Image.fromarray(np.array(pixels).astype(np.uint8))
        resized_image = image.resize((256, 256))
        
        return np.array(resized_image)

    def save_gif(self, filename, duration=100):
        """Save the stored frames as a GIF."""
        if self.frames:
            self.frames[0].save(
                filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0
            )


def main(args):
    
    if args.use_comet:
        comet_ml.login()
        experiment = comet_ml.start(project_name="ppo_lag_craftext")
        experiment.log_parameters(args)
    if args.use_wandb:
        wandb.init(project="craftext_ppo_validation", name="validation_ppo", mode="online")  # или mode="disabled" для оффлайн

    # folder for animation
    animation_dir = "animation"
    os.makedirs(animation_dir, exist_ok=True)
    for filename in os.listdir(animation_dir):
        file_path = os.path.join(animation_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    def find_latest_checkpoint(base_path):
        # Получаем список всех подпапок в базовом пути
        try:
            subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        except FileNotFoundError:
            raise ValueError(f"Папка {base_path} не существует")
        
        # Фильтруем только числовые папки
        numeric_folders = []
        for folder in subfolders:
            try:
                num = int(folder)
                numeric_folders.append(num)
            except ValueError:
                continue
        
        if not numeric_folders:
            raise ValueError(f"В папке {base_path} нет подпапок с числовыми именами")
        
        # Находим максимальное число
        latest_num = max(numeric_folders)
        
        # Формируем полный путь
        full_path = os.path.join(base_path, str(latest_num), "default")
        
        return full_path

    #checkpoint_path = args.path  # Пример: checkpoints/PPO_LAG/exp_1
    checkpoint_path = find_latest_checkpoint(args.path)
    print(f"download weights path: {checkpoint_path}")

    # Восстанавливаем веса
    checkpointer = PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)

    train_state: TrainState = restored["train_state"]

    config = {}
    config["NUM_ENVS"] = 1
    config["ENV_NAME"] = args.env_name

    is_pixels = "Pixels" in config["ENV_NAME"]
    actions_count = 17 if "Classic" in config["ENV_NAME"] else 43

    env_name = config["ENV_NAME"].replace("-Text", "")
    env = make_craftax_env_from_name(env_name, False)
    env = CMDPInstructionWrapper(env, args.craftext_settings)
    env_params = env.default_params

    network = ActorCriticConvWithBERTCMDP(
            env.action_space(env_params).n, 512)

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    encoded_instruction = jnp.expand_dims(env.encoded_instruction, axis=0) 
    encoded_constraint = jnp.expand_dims(env.encoded_textual_constraint, axis=0) 
    network_params = network.init(_rng, init_x, encoded_instruction, encoded_constraint)

    num_tasks = len(env.scenario_handler.scenario_data_jax.constraints_embeddings_list)
    #num_tasks = 2
    tasks_ids = list(range(0, num_tasks))
    #tasks_ids = list(range(0, 2))
    print("tasks num:", len(env.scenario_handler.scenario_data_jax.constraints_embeddings_list))
    task_metrics = []

    for task_id in tasks_ids:
        print()
        print("Validation task id:", task_id, " of", num_tasks - 1)
        obs, env_state = env.reset(_rng, env_params, instruction_idx=task_id)
        done = False

        renderer = CraftaxRenderer(env, env_params, pixel_render_size=1)
        steps = 0
        step_fn = jax.jit(env.step, static_argnums=3)
        observations = []
        return_reward = 0
        return_cost = 0
        returns_reward = []
        returns_cost = []
        while not done and steps < 500:
            obs = jnp.expand_dims(obs, axis=0)
            instruction = env.scenario_handler.scenario_data.instructions_list[env_state.idx]
            textual_constraint = env.scenario_handler.scenario_data.texutal_constraints_list[env_state.idx]
            pi, value, cost_value = network.apply(train_state['params'], obs, 
                                    env_state.instruction.reshape(1, -1),
                                    env_state.textual_constraint.reshape(1, -1))
            action = pi.sample(seed=_rng)[0]
            
            action = jax.device_put(action, device=jax.devices('gpu')[0])

            if action is not None:
                rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = step_fn(
                    _rng, env_state, action, env_params
                )
                steps += 1
                
                return_reward += reward
                return_cost += env_state.cost
            returns_reward.append(return_reward)
            returns_cost.append(return_cost)

            image = renderer.render_to_image(env_state.env_state)
            observations.append(image)

        # get metrics
        total_success_rate = env_state.total_success_rate
        task_metrics.append({
            "task_id": task_id,
            "instruction": instruction,
            "constraint": textual_constraint,
            "reward": float(return_reward),
            "cost": float(return_cost),
            "success_rate": float(total_success_rate)
        })

        # get gif
        gif_name = instruction.replace(" ", "_")
        import random
        ix = random.randint(0,200)
        with imageio.get_writer(f'animation/{ix}_{gif_name}_{task_id}.gif', mode='I', duration=0.1) as writer:
            for i, image in enumerate(observations):
                text = f"Step {i}, Instruction: {instruction} \n Constrain: {textual_constraint}"
                right_text = f"rewards {returns_reward[i]:.2f}, costs {returns_cost[i]:.2f}"
                image_with_text = add_text_to_image(image, text, right_text=right_text)
                writer.append_data(image_with_text.astype(np.uint8))
        if args.use_wandb:
            wandb.log({
                "animation": wandb.Video(f"animation/{ix}_{gif_name}_{task_id}.gif", fps=10, format="gif")
            })
        if args.use_comet:
            experiment.log_image(f"animation/{ix}_{gif_name}_{task_id}.gif", name=f"animation_{ix}_{gif_name}_{task_id}")

    # Эмодзи-мап (оставляем как было)
    emoji_map = {
        "cow": "🐄", "beef": "🥩", "meat": "🥩", "plant": "🌱", "sapling": "🌿",
        "zombie": "🧟", "skeleton": "💀", "stone": "🪨", "furnace": "🔥",
        "coal": "🧱", "iron": "⛓️", "diamond": "💎", "sword": "🗡️",
        "pickaxe": "⛏️", "drink": "🥤", "satiety": "🍽️", "hunger": "😋", 
        "wooden": "🪵", "health": "❤️", "monster": "👹", "distance": "↔️"
    }

    # Новые функции для компактного отображения
    def simplify_text(text):
        """Сокращает текст инструкций и ограничений до минимальной формы"""
        replacements = [
            
            (r"(?i)when your (\w+) level is (less than|greater than|below|above) (\d+), (.*)", r"If \1 \2 \3: \4"),
            (r"(?i)less than", "<"),
            (r"(?i)greater than", ">"),
            (r"(?i)below", "<"),
            (r"(?i)above", ">"),
            
            (r"(?i)eat (a|an|the) (\w+)", r"Eat \2"),
            (r"(?i)you must maintain (?:your )?(\w+) level (?:at or )?above (\d+)", r"\1 >= \2"),
            (r"(?i)you must maintain (?:your )?(\w+) level", r"\1"),
            (r"(?i)level must remain (?:at or )?above (\d+)", r">= \1"),
            (r"(?i)(?:collect|gather) (?:a|an|\d+) (\w+)", r"Get \1"),
            (r"(?i)craft (?:a|an) (?:wooden|iron|golden) (\w+)", r"Make \1"),
            (r"(?i)do not let (\w+) fall below (\d+)", r"\1 >= \2"),
            (r"(?i)keep (\w+) above (\d+)", r"\1 >= \2")
        ]
        for pattern, repl in replacements:  # Убрали .items() для списка
            text = re.sub(pattern, repl, text)
    
        # Дополнительная обработка для стандартизации
        text = text.replace("get", "Get").replace("make", "Make")
        return text.strip()

    def emojify_compact(text):
        """Заменяет ключевые слова на эмодзи"""
        for word, emoji in emoji_map.items():
            text = re.sub(rf"\b{word}\b", emoji, text, flags=re.IGNORECASE)
        return text

    # ===== НАЧАЛО НОВОГО КОДА ВИЗУАЛИЗАЦИИ =====
    # Подготовка данных
    tasks = [f"Task {i+1}" for i in range(len(task_metrics))]
    costs = [item['cost'] for item in task_metrics]
    success_rates = [item['success_rate'] for item in task_metrics]

    # Создаем фигуру
    fig = go.Figure()

    # Добавляем основной bar для cost с цветом по success rate
    fig.add_trace(go.Bar(
        x=tasks,
        y=costs,
        name='Cost',
        marker=dict(
            color=success_rates,
            colorscale='Viridis',
            cmin=0,
            cmax=1,
            colorbar=dict(title='Success Rate'),
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=[f"Cost: {c:.1f}<br>Success: {s:.2f}" for c, s in zip(costs, success_rates)],
        textposition='auto',
        width=0.6
    ))

    cost_threshold = 0.1
    # Добавляем горизонтальную линию на y=5
    fig.add_hline(
        y=cost_threshold,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"Target={cost_threshold}", 
        annotation_position="top right"
    )

    # Настройка layout
    fig.update_layout(
        title="Constraint: " + task_metrics[0]["constraint"],
        xaxis_title="Tasks",
        yaxis_title="Cost Value",
        height=600,
        width=max(800, len(tasks) * 120),
        font=dict(size=8),
        margin=dict(b=150, l=50, r=50, t=80)
    )

    # Обновляем подписи с эмодзи
    fig.update_xaxes(
        tickvals=tasks,
        ticktext=[f"{emojify_compact(simplify_text(item['instruction']))}" 
                 for item in task_metrics],
        tickangle=-45
    )

    # Сохраняем и логируем
    plot_path = "task_metrics_cost_heatmap.html"
    fig.write_html(plot_path)
    fig.write_image("task_metrics_cost_heatmap.png")

    if args.use_wandb:
        wandb.log({
            "task_metrics_plot": wandb.Image("task_metrics_cost_heatmap.png"),
            "interactive_metrics": wandb.Html(open(plot_path))
        })
    if args.use_comet:
        # Логируем изображение с тепловой картой
        experiment.log_image("task_metrics_cost_heatmap.png", name="task_metrics_plot") 
        # Логируем HTML-файл с интерактивными метриками
        experiment.log_asset(plot_path, file_name="interactive_metrics.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1-Text")
    parser.add_argument(
        "--use_wandb", default=False
    )
    parser.add_argument(
        "--use_comet", default=True
    )

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)