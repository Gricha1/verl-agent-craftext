import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv 
from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP as ScenariosNoLambda
from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper, TextEnvStateCMDP
from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode
from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
from craftax.craftax_classic.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement
)

from craftax.craftax_env import make_craftax_env_from_name
import pygame
from pygame.colordict import THECOLORS as colors
import warnings

warnings.filterwarnings('ignore')
import logging
# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KEY_MAPPING = {
    pygame.K_q: Action.NOOP,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
    pygame.K_SPACE: Action.DO,
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_r: Action.PLACE_STONE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_WOOD_SWORD,
    pygame.K_5: Action.MAKE_STONE_SWORD,
    pygame.K_6: Action.MAKE_IRON_SWORD
}


class CrafTextRenderer:
    def __init__(self, env, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size + 150,
        )

        env_render = render_classic

        pygame.init()
        pygame.key.set_repeat(250, 75)
        self.font_t = pygame.font.Font(pygame.font.get_default_font(), 16)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = jax.jit(env_render, static_argnums=(1,))

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()
        # time.sleep(0.01)

    def render_field(self, env_state):
        """Render the environment state to an image array and resize it to 256x256."""
        pixels = self._render(env_state.env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)
        
        
        return pixels


    
    def draw_header(self, text, cost):
        pygame.draw.rect(self.screen_surface, colors['white'], (0, 0, self.screen_size[0], 150))
        
        text_surface = self.font_t.render(text, True, colors['black'])
        cost = self.font_t.render(cost, True, colors['black'])
        text_rect = text_surface.get_rect(center=(self.screen_size[0] // 2, 75 // 2))
        self.screen_surface.blit(text_surface, text_rect)
        
        text_rect = text_surface.get_rect(center=((self.screen_size[0] // 2), (75 // 2) + 75))
        self.screen_surface.blit(cost, text_rect)


    def render(self, env_state: TextEnvStateCMDP):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        idx = env_state.idx
        # Get instruction text by idx
        instruction = wrapper.scenario_handler.scenario_data.instructions_list[idx]
        image = np.array(self.render_field(env_state))

        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 75))

        self.draw_header(instruction, f'{env_state.episode_cost}')#{jnp.sum(jnp.abs(env_state.env_state.player_position - env_state.target_state.target_of_interest.last_visible_target_position))}')

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state):
        if state.is_sleeping:
            return Action.NOOP.value
        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return KEY_MAPPING[event.key].value

        return None


def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{Achievement(i).name} ({new_achievements.sum()}/{22})")




if __name__ == "__main__":

    env: CraftaxClassicPixelsEnv = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False) 
    wrapper = CMDPInstructionWrapper(
        env, 
        config_name='achievements_safe_math_wood_budget', 
        scenario_handler_class=ScenariosNoLambda, 
        encode_model_class=DistilBertEncode, 
        encode_form=EncodeForm.EMBEDDING
    )

    rng = jax.random.PRNGKey(1)
    env_params = env.default_params

    obs, env_state = wrapper.reset(rng, env_params)
    renderer = CrafTextRenderer(env, env_params, pixel_render_size=1)
    renderer.render(env_state=env_state)
    
    clock = pygame.time.Clock()
    jitted = jax.jit(wrapper.step)
    
    # # first call to jit compile
    # logger.info("Jit process step functions")
    # for i in range(10):
    #     action = jax.device_put(i, device=jax.devices('gpu')[0])
    #     jitted(rng, env_state, action, env_params)
    #     rng, _rng = jax.random.split(rng)
    #     old_achievements = env_state.env_state.achievements
    #     obs, env_state, reward, done, info = jitted(rng, env_state, i, env_params)
    #     new_achievements = env_state.env_state.achievements

    #     renderer.render(env_state)
        
    #     renderer.update()
    #     clock.tick(60)
    # logger.info("Finish")
    
    obs, env_state = wrapper.reset(rng, env_params)

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state.env_state)
        
        
        # action = jax.device_put(action, device=jax.devices('gpu')[0])
        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.env_state.achievements
            obs, env_state, reward, done, info = jitted(rng, env_state, action, env_params)
            new_achievements = env_state.env_state.achievements
            print_new_achievements(old_achievements, new_achievements)

            if reward > 0.01 or reward < -0.01:
                print(f"Reward: {reward}\n")


            renderer.render(env_state)

            if done:
                obs, env_state = wrapper.reset(rng, env_params)

        renderer.update()
        clock.tick(60)