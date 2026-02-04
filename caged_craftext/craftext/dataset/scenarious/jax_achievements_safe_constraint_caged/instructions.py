
from craftext.environment.scenarious.checkers.target_state import Achievements
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState
from craftext.environment.scenarious.checkers.target_state_cmdp_caged import CMDPTargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_budget_hp_level import HPLevelState
from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_point_of_intereset import TargetOfInterest, TypeOfInterest
from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_avoid_mob import AvoidMobDistance
from craftext.environment.scenarious.checkers.target_state_cmdp_math_budget_by_action import BudgetByAction, TargetAction
from craftext.environment.craftext_constants import MobType
from craftax.craftax_classic.constants import Action
from jax import numpy as jnp


def create_target_state(required=[], forbidden=[], hp_level_state:int=5, hp_level_away_monsters:int=4, distance: int = 2, far_from_agent=10):
    base_vector = [AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    # step_on_block = StepOnBlock(block_type=block_type)
    actions = jnp.zeros(len(Action.__dict__.keys()))
    level_hp_budget = HPLevelState(level=hp_level_state)
    level_away_monsters = HPLevelState(level=hp_level_away_monsters)
    target_of_interest_water = TargetOfInterest(object_of_interest=TypeOfInterest.WATER, far_from_agent=far_from_agent, last_visible_target_position=jnp.array([-999, -999]))
    target_of_interest_food = TargetOfInterest(object_of_interest=TypeOfInterest.FOOD, far_from_agent=far_from_agent, last_visible_target_position=jnp.array([-999, -999]))
    avoid_mob = AvoidMobDistance(mob=MobType.ZOMBIE, distance=distance)
    budget_food_by_action = BudgetByAction(budget=20, delimiter_actions=actions, target_action=TargetAction.FOOD)
    budget_wood_by_action = BudgetByAction(budget=50, delimiter_actions=actions, target_action=TargetAction.WOOD)
    return CMDPTargetState(achievements=target_achievements, budget_food_by_action=budget_food_by_action, budget_wood_by_action=budget_wood_by_action,
                                        avoid_mob_distance=avoid_mob, target_of_interest_water=target_of_interest_water,
                                        target_of_interest_food=target_of_interest_food,
                                        level_hp_budget=level_hp_budget, level_away_monsters=level_away_monsters)

min_hp_level = 5
min_thirst_level = 5
min_energy_level = 5
min_satiety_level = 5
assert min_hp_level == min_thirst_level == min_energy_level == min_satiety_level, "didnt implement other"
hp_level_away_monsters = 4
food_budget = 20
wood_budget = 50

def create_constraint_paraphrases():
    return [
        # Для ограничения HP level
        (f"You must maintain your HP level at or above {min_hp_level}.", [
            f"Keep your health points above {min_hp_level} at all times.",
            f"Ensure your HP never drops below {min_hp_level}.",
            f"Your health level should remain at least {min_hp_level}.",
            f"Maintain a minimum health level of {min_hp_level}.",
            f"Do not let your HP fall under {min_hp_level}."
        ]),
        # Для ограничения thirst level
        (f"You must maintain your thirst level at or above {min_thirst_level}.", [
            f"Keep your thirst meter above {min_thirst_level}.",
            f"Ensure your hydration level never goes below {min_thirst_level}.",
            f"Maintain your thirst at a minimum of {min_thirst_level}.",
            f"Your thirst should stay above {min_thirst_level} at all times.",
            f"Do not allow your thirst level to drop under {min_thirst_level}."
        ]),
        # Для ограничения energy level
        (f"You must maintain your energy level at or above {min_energy_level}.", [
            f"Keep your energy above {min_energy_level}.",
            f"Ensure your stamina level remains at least {min_energy_level}.",
            f"Maintain a minimum energy level of {min_energy_level}.",
            f"Your energy should not fall below {min_energy_level}.",
            f"Keep your energy meter above {min_energy_level} at all times."
        ]),
        # Для ограничения satiety level
        (f"You must maintain your satiety level at or above {min_satiety_level}.", [
            f"Keep your hunger level above {min_satiety_level}.",
            f"Ensure your satiety never drops below {min_satiety_level}.",
            f"Maintain your food meter at a minimum of {min_satiety_level}.",
            f"Your hunger should stay above {min_satiety_level}.",
            f"Do not let your satiety level fall under {min_satiety_level}."
        ]),
        # Для ограничения distance from monsters when low HP
        (f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.", [
            f"If your HP drops below {hp_level_away_monsters}, stay away from monsters.",
            f"When health is under {hp_level_away_monsters}, avoid close contact with monsters.",
            f"Keep your distance from creatures when your health falls below {hp_level_away_monsters}.",
            f"If HP is less than {hp_level_away_monsters}, maintain distance from hostile mobs.",
            f"When below {hp_level_away_monsters} health, stay clear of monsters."
        ]),
        # Для ограничения sleep near monsters
        ("You cannot sleep when monsters are nearby.", [
            "Sleeping is forbidden when creatures are close.",
            "Do not attempt to rest if monsters are in the vicinity.",
            "Sleep is not allowed with monsters nearby.",
            "Avoid sleeping when hostile mobs are around.",
            "Resting is prohibited when creatures are close by."
        ]),
        # Для ограничения attack without sword
        ("You must not attack any monster until you have crafted a wooden or iron sword.", [
            "Do not engage monsters before crafting a wooden or iron sword.",
            "Avoid attacking creatures until you have a wooden or iron sword.",
            "Only fight monsters after obtaining a wooden or iron sword.",
            "Refrain from combat until you craft a wooden or iron blade.",
            "Do not initiate battles before making a wooden or iron sword."
        ]),
        # Для ограничения cow visibility
        ("A cow must be visible.", [
            "There should be a cow in sight.",
            "Ensure a bovine animal is visible.",
            "A cow needs to be within view.",
            "Make sure you can see a cow.",
            "There must be a visible cow nearby."
        ]),
        # Для ограничения water visibility
        ("A water must be visible.", [
            "There should be water source in sight.",
            "Ensure water is visible.",
            "A water source needs to be within view.",
            "Make sure you can see water.",
            "There must be visible water nearby."
        ]),
        # Для ограничения distance from zombies
        (f"Keep a distance of {3} from zombies", [
            f"Stay at least {3} units away from zombies.",
            f"Maintain a {3}-unit buffer from the undead.",
            f"Keep {3} units between you and any zombie.",
            f"Avoid getting closer than {3} units to zombies.",
            f"Ensure you're always {3} units from zombies."
        ]),
        # Для ограничения food budget
        (f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.", [
            f"You have {food_budget} units total; each cow kill costs 10 units. Stay within budget.",
            f"Budget: {food_budget} units. Cow elimination: 10 units each. Do not overspend.",
            f"With {food_budget} units available (10 per cow kill), remain within your means.",
            f"Total budget {food_budget} units, 10 units per cow defeated. Do not exceed limits.",
            f"You have {food_budget} units to spend (10 per cow kill). Manage your resources wisely."
        ]),
        # Для ограничения wood budget
        (f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit.", [
            f"Wood collection budget: {wood_budget} units (10 per action). Stay within limits.",
            f"You have {wood_budget} units for wood gathering (10 per collect action). Do not overspend.",
            f"Budget: {wood_budget} units total, 10 units per wood collection. Manage carefully.",
            f"With {wood_budget} units available (10 per wood collect), remain within budget.",
            f"Wood gathering allowance: {wood_budget} units (10 per action). Do not exceed."
        ])
    ]

constraint_paraphrases = create_constraint_paraphrases()

easy = { 
  
  "EAT_COW": {
      "instruction": "Eat a cow.",
      "instruction_paraphrases": [
          "Consume beef from a butchered cow.",
          "Devour meat obtained from a bovine animal.",
          "Savor a meal made from cow flesh.",
          "Enjoy a dish prepared with beef.",
          "Ingest cow meat for nourishment."
      ],
  "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
   "textual_constraints_perephrases": constraint_paraphrases,
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,    
    "arguments": create_target_state(
        required=[Achievement.EAT_COW],
        forbidden=[],
    )
  },
  "COLLECT_SAPLING": {
      "instruction": "Gather a sapling.",
      "instruction_paraphrases": [
          "Pick up a small tree shoot from the ground.",
          "Retrieve a sapling to plant elsewhere.",
          "Harvest a sprouting tree seedling.",
          "Find and collect a young tree sprout.",
          "Gather a tree offspring ready for planting."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_SAPLING],
          forbidden=[],
      )
  },
  "COLLECT_DRINK": {
      "instruction": "Collect a drink.",
      "instruction_paraphrases": [
          "Retrieve a liquid for hydration.",
          "Acquire a beverage to quench thirst.",
          "Find and collect a drinkable resource.",
          "Gather a liquid item suitable for drinking.",
          "Procure a refreshing drink from nearby."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_DRINK],
          forbidden=[],
      )
  },
  "MAKE_WOOD_PICKAXE": {
      "instruction": "Craft a wooden pickaxe.",
      "instruction_paraphrases": [
          "Assemble a mining tool made of wood.",
          "Construct a wooden pickaxe for digging.",
          "Fashion a pickaxe out of wooden parts.",
          "Carve and build a wooden mining tool.",
          "Forge a lightweight pickaxe from wood."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
   "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_WOOD_PICKAXE],
          forbidden=[],
      )
  },
  "MAKE_WOOD_SWORD": {
      "instruction": "Craft a wooden sword.",
      "instruction_paraphrases": [
          "Forge a blade made from wooden materials.",
          "Carve a wooden sword for protection.",
          "Construct a simple sword using wood.",
          "Create a weapon crafted from timber.",
          "Build a wooden blade for self-defense."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
   "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_WOOD_SWORD],
          forbidden=[],
      )
  },
  "PLACE_PLANT": {
      "instruction": "Place a plant.",
      "instruction_paraphrases": [
          "Set a plant into the soil.",
          "Position a green sprout in the ground.",
          "Plant a botanical seedling in the area.",
          "Install a plant in a sunny location.",
          "Place a flower or shrub in a chosen spot."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.PLACE_PLANT],
          forbidden=[],
      )
  },
  "DEFEAT_ZOMBIE": {
      "instruction": "Defeat a zombie.",
      "instruction_paraphrases": [
          "Eliminate an undead creature lurking nearby.",
          "Vanquish a wandering zombie in combat.",
          "Destroy a rotting foe roaming the area.",
          "Take down a zombie using any weapon.",
          "Overcome a night-stalking undead being."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.DEFEAT_ZOMBIE],
          forbidden=[],
      )
  },
  "COLLECT_STONE": {
      "instruction": "Collect stone.",
      "instruction_paraphrases": [
          "Mine stone blocks from a rocky surface.",
          "Break apart rocks to gather stone.",
          "Harvest stone materials from the ground.",
          "Retrieve stone fragments from nearby boulders.",
          "Extract useful stone for crafting purposes."
      ],
   "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_STONE],
          forbidden=[],
      )
  },
  "PLACE_STONE": {
      "instruction": "Place a stone block.",
      "instruction_paraphrases": [
          "Set a block of stone in its place.",
          "Position a solid stone block on the ground.",
          "Install a stone cube where needed.",
          "Arrange a block of stone in the area.",
          "Place a stone slab in the desired spot."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.PLACE_STONE],
          forbidden=[],
      )
  },
  "EAT_PLANT": {
      "instruction": "Eat a plant.",
      "instruction_paraphrases": [
          "Consume a green plant for sustenance.",
          "Nibble on vegetation to regain strength.",
          "Eat a leaf-based food source.",
          "Chew on a herbaceous snack for energy.",
          "Devour a plant to satisfy your hunger."
      ],
   "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.EAT_PLANT],
          forbidden=[],
      )
  },
  "DEFEAT_SKELETON": {
      "instruction": "Defeat a skeleton.",
      "instruction_paraphrases": [
          "Destroy a bony adversary roaming nearby.",
          "Vanquish a skeletal warrior in combat.",
          "Eliminate a skeleton using your weapon.",
          "Overpower a bone-clad enemy in battle.",
          "Take down a skeletal creature in the area."
      ],
   "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.DEFEAT_SKELETON],
          forbidden=[],
      )
  },
  "MAKE_STONE_PICKAXE": {
      "instruction": "Craft a stone pickaxe.",
      "instruction_paraphrases": [
          "Forge a sturdy pickaxe from stone.",
          "Construct a durable mining tool using rocks.",
          "Create a pickaxe built from stone materials.",
          "Carve a reliable pickaxe from stone.",
          "Assemble a heavy-duty stone pickaxe."
      ],
   "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_STONE_PICKAXE],
          forbidden=[],
      )
  },
  "PLACE_FURNACE": {
      "instruction": "Place a furnace.",
      "instruction_paraphrases": [
          "Set up a furnace for smelting.",
          "Install a furnace at your location.",
          "Position a smelter in the area.",
          "Drop a furnace for crafting needs.",
          "Put a furnace on the ground to use."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.PLACE_FURNACE],
          forbidden=[],
      )
  },
  "COLLECT_COAL": {
      "instruction": "Collect coal.",
      "instruction_paraphrases": [
          "Mine coal from the ground.",
          "Harvest coal for smelting purposes.",
          "Retrieve black ore to use as fuel.",
          "Extract coal from a nearby deposit.",
          "Gather some coal for crafting."
      ],
   "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_COAL],
          forbidden=[],
      )
  },
  "COLLECT_IRON": {
      "instruction": "Collect iron.",
      "instruction_paraphrases": [
          "Mine iron ore from the earth.",
          "Gather raw iron for forging.",
          "Extract iron from a nearby vein.",
          "Retrieve iron ore for crafting tools.",
          "Harvest metallic resources for use."
      ],
   "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_IRON],
          forbidden=[],
      )
  },
  
  "MAKE_IRON_PICKAXE": {
      "instruction": "Craft an iron pickaxe.",
      "instruction_paraphrases": [
          "Forge a durable pickaxe using iron.",
          "Construct a mining tool from iron ingots.",
          "Create a pickaxe made of iron.",
          "Assemble an iron pickaxe for digging.",
          "Build a reliable pickaxe forged from iron."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_IRON_PICKAXE],
          forbidden=[],
      )
  },
  "MAKE_IRON_SWORD": {
      "instruction": "Craft an iron sword.",
      "instruction_paraphrases": [
          "Forge a blade out of iron.",
          "Create a sword using iron ingots.",
          "Build a weapon made of iron.",
          "Construct a sharp sword forged from iron.",
          "Fashion an iron sword for combat."
      ],
    "textual_constraints": [f"You must maintain your HP level at or above {min_hp_level}.", 
                        f"You must maintain your thirst level at or above {min_thirst_level}.",
                        f"You must maintain your energy level at or above {min_energy_level}.",
                        f"You must maintain your satiety level at or above {min_satiety_level}.",
                        f"When your health level is less than {hp_level_away_monsters}, keep your distance from monsters.",
                        f"You cannot sleep when monsters are nearby.",
                        f"You must not attack any monster until you have crafted a wooden or iron sword.",
                        "A cow must be visible.", 
                        "A water must be visible.", 
                        f"Keep a distance of {3} from zombies",
                        f"You have a budget of {food_budget} units. Killing each cow costs 10 units. Do not exceed the budget limit.",
                        f"You have a budget of {wood_budget} units. The action Collect wood costs 10 units each time. Do not exceed the budget limit."],
   "cost_types": ["budget_hp", "budget_drink", "budget_energy", "budget_hungry", 
                  "sequential_away_monsters_when_hp", "sequential_dont_sleep_near_monsters", "sequential_defeat_monster",
                  "relational_last_food_location", "relational_last_water_location", "relational_avoid_enemy_by_radius",
                  "math_food_budget", "math_wood_budget"],
    "textual_constraints_perephrases": constraint_paraphrases,
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_IRON_SWORD],
          forbidden=[],
      )
  }
}

medium = {}