
from enum import Enum
from craftext.checkers.achivments import conditional_achivments
from craftext.scenarios.constants import Achievement
import jax
import jax.numpy as jnp
from jax import lax
from enum import Enum

one = {
   "1. collect_wood_place_table": {
        "instruction": "Chop some wood and install a workbench.",
        "instruction_paraphrases": [
             "Gather wood and set down a crafting table.",
             "Harvest timber and position a crafting bench nearby.",
             "Chop some wood and install a workbench.",
             "Retrieve logs and put a crafting table on the ground.",
            "Procure wood and place a crafting table in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.PLACE_TABLE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
        
     "2. collect_wood_place_table": {
        "instruction": "Harvest timber and position a crafting bench nearby.",
        "instruction_paraphrases": [
             "Gather wood and set down a crafting table.",
             "Harvest timber and position a crafting bench nearby.",
             "Chop some wood and install a workbench.",
             "Retrieve logs and put a crafting table on the ground.",
            "Procure wood and place a crafting table in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.PLACE_TABLE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
}
easy = {
    "collect_wood": {
        "instruction": "Collect wood.",
        "instruction_paraphrases": [
            "Harvest logs from nearby trees.",
            "Chop down some timber to gather wood.",
            "Cut a tree to obtain wooden resources.",
            "Retrieve lumber from a fallen tree.",
            "Procure wood by felling trees in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_WOOD.value else 0 for a in  range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "place_table": {
        "instruction": "Place a crafting table.",
        "instruction_paraphrases": [
            "Set down a crafting bench in the area.",
            "Install a crafting workstation at your location.",
            "Position a workbench for crafting nearby.",
            "Drop a table designed for crafting tasks.",
            "Arrange a crafting station in a suitable spot."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.PLACE_TABLE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "eat_cow": {
        "instruction": "Eat a cow.",
        "instruction_paraphrases": [
            "Consume beef from a butchered cow.",
            "Devour meat obtained from a bovine animal.",
            "Savor a meal made from cow flesh.",
            "Enjoy a dish prepared with beef.",
            "Ingest cow meat for nourishment."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.EAT_COW.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_sapling": {
        "instruction": "Gather a sapling.",
        "instruction_paraphrases": [
            "Pick up a small tree shoot from the ground.",
            "Retrieve a sapling to plant elsewhere.",
            "Harvest a sprouting tree seedling.",
            "Find and collect a young tree sprout.",
            "Gather a tree offspring ready for planting."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_SAPLING.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_drink": {
        "instruction": "Collect a drink.",
        "instruction_paraphrases": [
            "Retrieve a liquid for hydration.",
            "Acquire a beverage to quench thirst.",
            "Find and collect a drinkable resource.",
            "Gather a liquid item suitable for drinking.",
            "Procure a refreshing drink from nearby."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_DRINK.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_wood_pickaxe": {
        "instruction": "Craft a wooden pickaxe.",
        "instruction_paraphrases": [
            "Assemble a mining tool made of wood.",
            "Construct a wooden pickaxe for digging.",
            "Fashion a pickaxe out of wooden parts.",
            "Carve and build a wooden mining tool.",
            "Forge a lightweight pickaxe from wood."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_WOOD_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_wood_sword": {
        "instruction": "Craft a wooden sword.",
        "instruction_paraphrases": [
            "Forge a blade made from wooden materials.",
            "Carve a wooden sword for protection.",
            "Construct a simple sword using wood.",
            "Create a weapon crafted from timber.",
            "Build a wooden blade for self-defense."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_WOOD_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "place_plant": {
        "instruction": "Place a plant.",
        "instruction_paraphrases": [
            "Set a plant into the soil.",
            "Position a green sprout in the ground.",
            "Plant a botanical seedling in the area.",
            "Install a plant in a sunny location.",
            "Place a flower or shrub in a chosen spot."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "defeat_zombie": {
        "instruction": "Defeat a zombie.",
        "instruction_paraphrases": [
            "Eliminate an undead creature lurking nearby.",
            "Vanquish a wandering zombie in combat.",
            "Destroy a rotting foe roaming the area.",
            "Take down a zombie using any weapon.",
            "Overcome a night-stalking undead being."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.DEFEAT_ZOMBIE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_stone": {
        "instruction": "Collect stone.",
        "instruction_paraphrases": [
            "Mine stone blocks from a rocky surface.",
            "Break apart rocks to gather stone.",
            "Harvest stone materials from the ground.",
            "Retrieve stone fragments from nearby boulders.",
            "Extract useful stone for crafting purposes."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_STONE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "place_stone": {
        "instruction": "Place a stone block.",
        "instruction_paraphrases": [
            "Set a block of stone in its place.",
            "Position a solid stone block on the ground.",
            "Install a stone cube where needed.",
            "Arrange a block of stone in the area.",
            "Place a stone slab in the desired spot."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.PLACE_STONE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "eat_plant": {
        "instruction": "Eat a plant.",
        "instruction_paraphrases": [
            "Consume a green plant for sustenance.",
            "Nibble on vegetation to regain strength.",
            "Eat a leaf-based food source.",
            "Chew on a herbaceous snack for energy.",
            "Devour a plant to satisfy your hunger."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.EAT_PLANT.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "defeat_skeleton": {
        "instruction": "Defeat a skeleton.",
        "instruction_paraphrases": [
            "Destroy a bony adversary roaming nearby.",
            "Vanquish a skeletal warrior in combat.",
            "Eliminate a skeleton using your weapon.",
            "Overpower a bone-clad enemy in battle.",
            "Take down a skeletal creature in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.DEFEAT_SKELETON.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_stone_pickaxe": {
        "instruction": "Craft a stone pickaxe.",
        "instruction_paraphrases": [
            "Forge a sturdy pickaxe from stone.",
            "Construct a durable mining tool using rocks.",
            "Create a pickaxe built from stone materials.",
            "Carve a reliable pickaxe from stone.",
            "Assemble a heavy-duty stone pickaxe."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_STONE_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
     "wake_up": {
        "instruction": "Wake up.",
        "instruction_paraphrases": [
            "Rise from your sleep.",
            "Get out of bed and start your day.",
            "Awaken and prepare for action.",
            "Stand up after resting.",
            "Shake off your slumber and wake up."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.WAKE_UP.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "place_furnace": {
        "instruction": "Place a furnace.",
        "instruction_paraphrases": [
            "Set up a furnace for smelting.",
            "Install a furnace at your location.",
            "Position a smelter in the area.",
            "Drop a furnace for crafting needs.",
            "Put a furnace on the ground to use."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.PLACE_FURNACE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_coal": {
        "instruction": "Collect coal.",
        "instruction_paraphrases": [
            "Mine coal from the ground.",
            "Harvest coal for smelting purposes.",
            "Retrieve black ore to use as fuel.",
            "Extract coal from a nearby deposit.",
            "Gather some coal for crafting."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_COAL.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_iron": {
        "instruction": "Collect iron.",
        "instruction_paraphrases": [
            "Mine iron ore from the earth.",
            "Gather raw iron for forging.",
            "Extract iron from a nearby vein.",
            "Retrieve iron ore for crafting tools.",
            "Harvest metallic resources for use."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_IRON.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_diamond": {
        "instruction": "Collect a diamond.",
        "instruction_paraphrases": [
            "Mine a shiny diamond from the ground.",
            "Retrieve a precious gem from a deposit.",
            "Harvest a sparkling jewel for crafting.",
            "Find and collect a valuable diamond.",
            "Extract a rare diamond from a hidden vein."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_DIAMOND.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_iron_pickaxe": {
        "instruction": "Craft an iron pickaxe.",
        "instruction_paraphrases": [
            "Forge a durable pickaxe using iron.",
            "Construct a mining tool from iron ingots.",
            "Create a pickaxe made of iron.",
            "Assemble an iron pickaxe for digging.",
            "Build a reliable pickaxe forged from iron."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_iron_sword": {
        "instruction": "Craft an iron sword.",
        "instruction_paraphrases": [
            "Forge a blade out of iron.",
            "Create a sword using iron ingots.",
            "Build a weapon made of iron.",
            "Construct a sharp sword forged from iron.",
            "Fashion an iron sword for combat."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_IRON_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "1. collect_wood_place_table": {
        "instruction": "Collect wood and place a crafting table.",
        "instruction_paraphrases": [
            "Gather wood and set down a crafting table.",
            "Harvest timber and position a crafting bench nearby.",
            "Chop some wood and install a workbench.",
            "Retrieve logs and put a crafting table on the ground.",
            "Procure wood and place a crafting table in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.PLACE_TABLE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "2. make_wood_pickaxe_collect_stone": {
        "instruction": "Craft a wooden pickaxe and collect stone.",
        "instruction_paraphrases": [
            "Create a wooden pickaxe and mine stone.",
            "Assemble a wood pickaxe and gather stone.",
            "Craft a mining tool from wood and extract stone.",
            "Build a wooden pickaxe and harvest stone blocks.",
            "Forge a wooden pickaxe and collect rocky materials."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.COLLECT_STONE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "3. collect_drink_eat_cow": {
        "instruction": "Collect a drink and eat a cow.",
        "instruction_paraphrases": [
            "Retrieve a beverage and consume beef.",
            "Find a drink and eat cow meat.",
            "Collect a liquid and savor cow flesh.",
            "Gather a drinkable item and devour beef.",
            "Acquire a drink and enjoy a meal from a cow."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DRINK.value, Achievement.EAT_COW.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "4. place_stone_wake_up": {
        "instruction": "Place a stone block and wake up.",
        "instruction_paraphrases": [
            "Set down a stone block and rise from sleep.",
            "Position a stone and awaken.",
            "Place a block of stone and stand up after resting.",
            "Install a stone block and shake off slumber.",
            "Drop a stone block and prepare for the day."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_STONE.value, Achievement.WAKE_UP.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "5. place_furnace_place_stone": {
        "instruction": "Place a furnace and a stone block.",
        "instruction_paraphrases": [
            "Set up a furnace and position a stone block.",
            "Install a smelter and place down a stone.",
            "Drop a furnace and a block of stone.",
            "Put a furnace and a stone in their spots.",
            "Arrange a furnace and a stone block nearby."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_FURNACE.value, Achievement.PLACE_STONE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    
    "6. eat_cow_make_stone_pickaxe": {
        "instruction": "Eat a cow and craft a stone pickaxe.",
        "instruction_paraphrases": [
            "Consume beef and create a stone pickaxe.",
            "Eat cow meat and forge a stone mining tool.",
            "Savor a meal from a cow and craft a pickaxe from stone.",
            "Devour beef and assemble a stone pickaxe.",
            "Have a cow-based dish and make a stone pickaxe."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.EAT_COW.value, Achievement.MAKE_STONE_PICKAXE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "8. place_plant_defeat_zombie": {
        "instruction": "Place a plant and defeat a zombie.",
        "instruction_paraphrases": [
            "Set a plant into the ground and eliminate a zombie.",
            "Position a plant and destroy an undead creature.",
            "Plant a shrub and take down a zombie.",
            "Drop a plant and vanquish a wandering zombie.",
            "Install a plant and defeat a night-stalking zombie."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_PLANT.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    
    "10. place_furnace_make_iron_sword_no_pickaxe": {
        "instruction": "Place a furnace and craft an iron sword but do not make an iron pickaxe.",
        "instruction_paraphrases": [
            "Set up a furnace and forge an iron blade, avoiding pickaxe crafting.",
            "Place a furnace and create an iron sword, refraining from making a pickaxe.",
            "Install a furnace and craft a sword from iron, ensuring no pickaxe is made.",
            "Drop a furnace and construct an iron weapon, skipping pickaxe creation.",
            "Put down a furnace and make an iron sword, avoiding the pickaxe blueprint."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else
            -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },

    "12. defeat_zombie_not_defeat_skeleton": {
        "instruction": "Defeat a zombie but do not kill a skeleton.",
        "instruction_paraphrases": [
            "Take down a zombie and avoid fighting skeletons.",
            "Eliminate a zombie while sparing the skeletons.",
            "Destroy a zombie but refrain from attacking skeletons.",
            "Fight and defeat a zombie, leaving skeletons untouched.",
            "Vanquish a zombie while avoiding any conflict with skeletons."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.DEFEAT_ZOMBIE.value else
            -1 if a == Achievement.DEFEAT_SKELETON.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
  
    "14. make_stone_sword_defeat_skeleton": {
        "instruction": "Craft a stone sword and defeat a skeleton.",
        "instruction_paraphrases": [
            "Forge a stone blade and eliminate a skeletal enemy.",
            "Create a weapon from stone and take down a skeleton.",
            "Build a stone sword and vanquish a bone-clad foe.",
            "Make a sword out of stone and defeat a skeleton warrior.",
            "Construct a stone sword and destroy a skeleton."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_SWORD.value, Achievement.DEFEAT_SKELETON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
   
    "16. make_stone_pickaxe_collect_coal": {
        "instruction": "Craft a stone pickaxe and collect coal.",
        "instruction_paraphrases": [
            "Forge a stone pickaxe and mine coal.",
            "Build a pickaxe from stone and harvest coal.",
            "Create a durable mining tool and extract coal deposits.",
            "Craft a stone pickaxe and gather fuel for smelting.",
            "Make a stone pickaxe and collect black ore."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.COLLECT_COAL.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    }, 
    'INSTRUCTION_0 1': 
        {
            'instruction': "Please make a stone pickaxe and avoid making an iron sword",
            'instruction_paraphrases': [
                "Kindly forge a pickaxe using stone materials and do not create an iron-blade sword.",
                "Could you create a stone tool for mining? But refrain from constructing a sword from iron",
                "We require a stone mining tool, however, it's important to avoid the creation of an iron combat weapon",
                "Could I ask you to construct a stone mining implement? Make sure not to produce a martial weapon from iron.",
                "It's really necessary for you to fabricate a mining apparatus from rock but critically vital not to assemble a combat tool using iron materials."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a == Achievement.MAKE_STONE_PICKAXE.value else -1 if a == Achievement.MAKE_IRON_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_STONE_PICKAXE.value else -1 if a == Achievement.MAKE_IRON_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

    # 'INSTRUCTION_1': 
    # {
    #     'instruction': "Collect some iron and diamond in this order",
    #     'instruction_paraphrases': [
    #         "You must gather up iron and diamonds.",
    #         "Iron and diamond are your collection targets.",
    #         "Your task is to accumulate iron in the game world, followed by diamond.",
    #         "Initially, begin by obtaining iron from the environment and proceed by collecting some diamond thereafter.",
    #         "The sequence of actions you're supposed to carry out involves initial procurement of iron, followed by the acquisition of diamond."
    #     ],
    #     'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
    #         1 if a in [Achievement.COLLECT_IRON.value, Achievement.COLLECT_DIAMOND.value] else 0
    #         for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    #     ])),
    #     'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_IRON.value, Achievement.COLLECT_DIAMOND.value] else 0 for a in range(Achievement.COLLECT_DIAMOND.value+1)]))"
    # }, 

    'INSTRUCTION_2': \
    {
        'instruction': "Collect a sapling and some iron in the game.",
        'instruction_paraphrases': [
            "Gather a young tree and iron ores.",
            "Pick up a plant shoot and mine some iron.",
            "Acquire a small tree and some iron minerals.",
            "Procure a budding plant and iron nuggets.",
            "Obtain a sprout and some iron deposit."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
            [1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.COLLECT_IRON.value] else 0
             for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.COLLECT_IRON.value] else 0 for a in range(Achievement.COLLECT_IRON.value+1)]))"
    }, 

    'INSTRUCTION_3_1': {
        'instruction': "Eat a plant, but make sure you haven’t eaten a cow.",
        'instruction_paraphrases': [
            "Consume a plant, without having ingested a cow.",
            "Partake in the eating of greenery, but avoid consumption of any bovine creatures.",
            "Take a bite out of some flora, but steer clear of taking a bite out of a moo-moo.",
            "Feast upon plant life solely, abstaining from the consumption of beef.",
            "Get your nourishment from plants, while maintaining a distance from cow meat."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.EAT_PLANT.value else
            -1 if a == Achievement.EAT_COW.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a == Achievement.EAT_PLANT.value else -1 if a == Achievement.EAT_COW.value else 0 for a in range(Achievement.EAT_PLANT.value+1)]))"
    }, 

#         'INSTRUCTION_5_1': 
#         {
#             'instruction': "Start by waking up, then gather some wood, and finally place a stone.",
#             'instruction_paraphrases': 
#             [
#                 "First, wake up. Secondly, collect some timber and lastly, drop a rock.",
#                 "Begin by rising from your slumber, then procure some logs and conclude by positioning a boulder.",
#                 "Initiate your activities by awakening, followed by harvesting lumber, and end by depositing a stone.",
#                 "Commence the sequence by stirring from your sleep, proceed to gather timber, and wrap up by placing a pebble.",
#                 "Wake from your dreams as a start, proceed with wood accumulation and then cap it off with a stoney placement."
#             ],
#             'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
#             [
#                 1 if a in [Achievement.WAKE_UP.value, Achievement.COLLECT_WOOD.value, Achievement.PLACE_STONE.value] else 0
#                 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
#             ])),
#             'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.WAKE_UP.value, Achievement.COLLECT_WOOD.value, Achievement.PLACE_STONE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
#         }
#    , 

        'INSTRUCTION_61': {
            'instruction': "Collect some wood, place a rock and make a stone pickaxe.",
            'instruction_paraphrases': [
                "Gather some timber, put a boulder in place and craft a stone pickaxe.",
                "Pick up some lumber, set a pebble and construct a stone chopping tool.",
                "Amass some wooden materials, arrange a cobble, and produce a stony axe.",
                "Assemble some fragments of tree, designate a spot for a stone, and cobble together a pickaxe made of stone.",
                "Procure wood from the surroundings, ensure a rock is situated and forge a tool for excavation out of stone."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.PLACE_STONE.value, Achievement.COLLECT_WOOD.value, Achievement.MAKE_STONE_PICKAXE.value] else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda': 'conditional_achivments(gd, np.array(['
                               '1 if a in [Achievement.PLACE_STONE.value, Achievement.COLLECT_WOOD.value, Achievement.MAKE_STONE_PICKAXE.value] else 0'
                               'for a in range(Achievement.MAKE_STONE_PICKAXE.value+1)'
                               ']))'
        }
   , 

    'INSTRUCTION_7_1':
    {
        'instruction': "Place a furnace and make an iron sword. Do not make an iron pickaxe.",
        'instruction_paraphrases': 
        [
            "Situate a forge and forge an iron blade. Abstain from crafting an iron rock breaker.",
            "Establish a hearth and shape an iron rapier. Refrain from manufacturing an iron mining instrument.", 
            "Install a stove and fabricate an iron sabre. Avoid producing an iron pick.",
            "Position an oven and construct an iron broadsword. Evade from forming an iron digging tool.",
            "Put up a smelting device and create an iron katana. Forgo shaping an iron excavation utensil."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
        [
            1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else
            -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([\n 1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else\n -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0\n for a in range(Achievement.MAKE_IRON_SWORD.value + 1)\n ]))"
    }, 

        'INSTRUCTION_8': \
        {
            'instruction': "Make sure you place a small tree, but don't craft a stone sword.",
            'instruction_paraphrases': [
                "I need you to plant a treeling, but avoid creating a stone blade.",
                "It's crucial for you to put down a sapling, but abstain from making a stone saber.",
                "Place a young tree in your environment but you must not form a rock weapon.",
                "Can you position a tree sprout but please don't manufacture a weapon from stone?",
                "Ensure a plantlet is positioned while making sure not to assemble a bladed stone tool."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(
                gd, jnp.array(
                    [1 if a == Achievement.PLACE_PLANT.value else -1 if a == Achievement.MAKE_STONE_SWORD.value else 0
                     for a in range(Achievement.MAKE_IRON_SWORD.value + 1)]
                )
            ),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a == Achievement.PLACE_PLANT.value else "
                                "-1 if a == Achievement.MAKE_STONE_SWORD.value else 0 for a in "
                                "range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

        'ACHIEVE_COLLECT_DRINK_PLACE_PLANT_DEFEAT_SKELETON': 
        {
            'instruction': "Collect a drink, place a plant, and defeat a skeleton.",
            'instruction_paraphrases': [
                "Get your hands on a drink, grow a plant and defeat the skeleton.",
                "Procure a beverage, install a herb and overcome the skeleton.",
                "Find a drinkable item, put a flora, and conquer the skeleton.",
                "Secure a quenchable, situate a vegetal, and triumph over a skeleton.",
                "Acquire a potable, establish a botanical and vanquish the skeleton."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.COLLECT_DRINK.value, Achievement.PLACE_PLANT.value, Achievement.DEFEAT_SKELETON.value] else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
            ])),
            'str_check_lambda': 
            "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DRINK.value, Achievement.PLACE_PLANT.value, Achievement.DEFEAT_SKELETON.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)]))"
        }
   , 

    # 'INSTRUCTION_10': \
    # {
    #     'instruction': "Collect some wood and defeat a zombie.",
    #     'instruction_paraphrases': [
    #         "Please gather some timber and slay a monster of the undead.",
    #         "Make it a point to amass logs and vanquish a zombie.",
    #         "It's necessary to attain lumber and conquer a walking dead.",
    #         "Your task involves procuring firewood and overwhelming a relentless corpse.",
    #         "You're required to harvest some forest offshoots and neutralize a resurrected being."
    #     ],
    #     'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
    #         1 if a in [Achievement.COLLECT_WOOD.value, Achievement.DEFEAT_ZOMBIE.value] else 0
    #         for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    #     ])),
    #     'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_WOOD.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.DEFEAT_ZOMBIE.value+1)]))"
    # }, 

    'INSTRUCTION_11': {
        'instruction': "You must wake up. Then, you need to defeat a zombie and lastly, fetch a drink for yourself.",
        'instruction_paraphrases': [
            "Get up from your sleep first. Post that, you must confront and win over a zombie. Lastly, make sure to obtain a drink.",
            "Rise from your rest first. Following this, it's important to tackle and overcome a zombie. In the end, secure a drink for your own self.",
            "You are required to awaken initially. Subsequent to this, it's crucial to resist and triumph over a zombie. To conclude, purchase a drink.",
            "You're needed to come out of your slumber at first. Upon accomplishing this, a zombie must be encountered and defeated. Ultimately, procure a beverage for yourself.",
            "You are supposed to wake up first. Then, a zombie needs to be challenged and defeated. Finally, get hold of a beverage."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.WAKE_UP.value, Achievement.DEFEAT_ZOMBIE.value, Achievement.COLLECT_DRINK.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.WAKE_UP.value, Achievement.DEFEAT_ZOMBIE.value, Achievement.COLLECT_DRINK.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_12': 
        {
            'instruction': "Don't create an Iron Sword. You need to, however, craft a Stone Sword and gather a Sapling.",
            'instruction_paraphrases': [
                "Make sure you aren't crafting an Iron Sword. However, you need to both collect a Sapling and manufacture a Stone Sword.",
                "You must not create an Iron Sword. Still, gather a Sapling and fashion a Stone Sword.",
                "Without making an Iron Sword, you have to garner a Sapling and bring into existence a Stone Sword.",
                "Regardless of what you do, don't craft an Iron Sword. But be sure to make a Stone Sword and collect some saplings.",
                "Without producing an Iron Sword, you must collect a Sapling and build a Stone Sword with all haste."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.MAKE_STONE_SWORD.value, Achievement.COLLECT_SAPLING.value] else
                -1 if a == Achievement.MAKE_IRON_SWORD.value else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_STONE_SWORD.value, Achievement.COLLECT_SAPLING.value] else -1 if a == Achievement.MAKE_IRON_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

    'INSTRUCTION_13': {
    
        'instruction': "Gather some iron, make a wooden pickaxe and defeat a skeleton.",
        'instruction_paraphrases': [
            "Please defeat a skeleton, after crafting a wooden pickaxe and collecting some iron.", 
            
            "Your tasks: Craft a pickaxe from wood, defeat a bone-rattler, and acquire some iron.", 
            
            "Collect some iron ores, construct a pickaxe using wood, and also ensure you vanquish a skeleton.", 
            
            "I need you to construct a wood-tipped pickaxe, defeat a skeleton warrior and gather some iron ores."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
           1 if a in [Achievement.COLLECT_IRON.value, Achievement.MAKE_WOOD_PICKAXE.value, Achievement.DEFEAT_SKELETON.value] else 0 
           for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_IRON.value, Achievement.MAKE_WOOD_PICKAXE.value, Achievement.DEFEAT_SKELETON.value] else 0 for a in range(Achievement.DEFEAT_SKELETON.value + 1)]))"
    }, 

        # 'Collect_wood_and_stone': 
        # {
        #     'instruction': "Collect both wood and stone",
        #     'instruction_paraphrases': [
        #         "Acquire timber and rock",
        #         "Procure planks and pebbles",
        #         "Get your hands on some wood and stone",
        #         "You need to gather some wood and stones",
        #         "You have to accumulate logs and rubble"
        #     ],
        #     'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array
        #     (
        #         [
        #             1 if a in [Achievement.COLLECT_WOOD.value, Achievement.COLLECT_STONE.value]
        #             else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        #         ]
        #     )),
        #     'str_check_lambda': """lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a in 
        #     [Achievement.COLLECT_WOOD.value, Achievement.COLLECT_STONE.value] else 0 for a in 
        #     range(Achievement.COLLECT_STONE.value+1)]))"""
        # }
   
'INSTRUCTION_15_1': {
    'instruction': "Collect some stone, defeat a skeleton and gather coal",
    'instruction_paraphrases': [
        "Please procure some rock, vanquish a skeleton and amass coal",
        "Can you amass stone, exterminate a skeleton and gather some coal?",
        "Your task is to gather rock, obliterate a skeleton and accumulate some coal",
        "You ought to collect rock, defeat a bony undead creature and mine some coal",
        "Your duty is to procure stone, overcome a skeleton and obtain coal",
    ],
    'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
        1 if a in [Achievement.COLLECT_STONE.value, Achievement.DEFEAT_SKELETON.value, Achievement.COLLECT_COAL.value] else 0
        for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    ])),
    'str_check_lambda': "conditional_achivments(...)"
}, 

    'INSTRUCTION_16': \
    {
        'instruction': "Please, make a stone sword and then wake up.",
        'instruction_paraphrases': [
            "Create a stone sword and then you must wake up.",
            "I need you to wake up after you have constructed a stone blade.",
            "Can you craft a rock sword and then arise?",
            "Could you produce a weapon made of stone, and then awaken?",
            "Can you manufacture an armament composed of sedimentary material, subsequent to your awakening?"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.MAKE_STONE_SWORD.value, Achievement.WAKE_UP.value] else 0 
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_STONE_SWORD.value, Achievement.WAKE_UP.value] else 0 for a in range(Achievement.DEFEAT_ZOMBIE.value+1)]))"
    }, 

        'CREATE_SWORD_AND_PLANT_IN_GAME': {
            'instruction': "Craft a wooden sword and cultivate a plant",
            'instruction_paraphrases': [
                "Make your own timber blade and place a sapling.",
                "Create a wooden weapon and improve the environment around you by growing a plant.",
                "Chisel a wooden falcata and utilize it in horticulture to grow a sapling.",
                "Craftsmanship with timber should result in a sword and some efforts of botany to place a young plant.",
                "Focus your woodworking skills to craft a wooden saber and engage in some green landscaping by cultivating a plant."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.MAKE_WOOD_SWORD.value, Achievement.PLACE_PLANT.value] else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_WOOD_SWORD.value, Achievement.PLACE_PLANT.value] else 0 for a in range(Achievement.MAKE_WOOD_SWORD.value+1)]))"
        }   
   , 

    'INSTRUCTION_18_1': {
        'instruction': "Please gather rock and then make an iron sword.",
        'instruction_paraphrases': [
            "Go on a mission to collect some stones and then transform some iron into a sword.",
            "Your task is to obtain some cobblestone, and after that, forge a blade utilizing iron.",
            "You need to secure boulders before proceeding to manufacture a tool of offense made of iron.",
            "I require you to accumulate rocky resources and then proceed to the creation of a sword made solemnly of iron."
            "Could you please head out, gather some dense rock and then utilize the collected iron to create a strong offensive tool?"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_STONE.value, Achievement.MAKE_IRON_SWORD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_STONE.value, Achievement.MAKE_IRON_SWORD.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_191': {
        'instruction': "It's essential that you collect some diamonds, gather up some iron, and feed yourself with some cow meat.",
        'instruction_paraphrases': [
            "Acquire valuable diamonds, collect an amount of iron, and make sure to eat some cow.",
            "You need to mine for both diamonds and iron, and don't forget to fill your hunger bar with beef.",
            "Gather up some precious gems and metals, specifically diamonds and iron, and replenish your food bar with some beef.",
            "You must find and collect diamonds, iron ore, and consume bovine food.",
            "Finding diamonds, collecting iron mineral deposits, and incorporating beef into your diet is crucial"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.COLLECT_IRON.value, Achievement.EAT_COW.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"""conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.COLLECT_IRON.value, Achievement.EAT_COW.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))"""
    },

  'INSTRUCTION_20_1': 
  {
    'instruction': "Make a wooden axe and defeat a skeleton, but in no specific order",
    'instruction_paraphrases': [
      "Your target is to craft a wooden pickaxe and defeat a skeleton. You can complete these tasks in any sequence.",
      "Please, take on the skeleton after or before crafting a wooden pickaxe.",
      "Your missions are to construct a wooden axe and take down a skeleton. Order doesn't matter.",
      "You need to accomplish two tasks. First is to build a wood pickaxe and second is to conquer a skeleton. You are free to choose the order.",
      "Your objectives include two tasks, creating a wooden tool for chopping, specifically a pickaxe, in addition to overcoming a skeleton enemy. You're allowed to tackle them in any order you prefer."
    ],
    'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
      1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.DEFEAT_SKELETON.value]
      else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
      ])),
    'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.DEFEAT_SKELETON.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
  }, 

    # 'INSTRUCTION_21': {
    #     'instruction': "Defeat a zombie and then consume a plant.",
    #     'instruction_paraphrases': [
    #         "Fight and kill a zombie, then eat a plant.",
    #         "Bring down a zombie first and after that, ingest a vegetation.",
    #         "Neutralize a ghoul in combat, subsequently, dine on a flora.",
    #         "Engage and slay an undead creature, followed by the ingestion of botanical specimen.",
    #         "Successfully execute a showdown with a member of the undead, post which, partake in the consumption of a plant-based diet."
    #     ],
    #     'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
    #         1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.EAT_PLANT.value] else 0
    #         for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    #     ])),
    #     'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    # }, 

    'COLLECT_WOOD_IRON': {
        'instruction': "You need to gather some wood and iron.",
        'instruction_paraphrases': [
            "It's essential for you to collect both timber and iron ore.",
            "Your task is harvesting wood and mining for some iron.",
            "You are required to amass both lumber and iron materials.",
            "Procure both wooden logs and iron elements as a part of the task.",
            "It's incumbent on you to accumulate components of both wood and iron."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.COLLECT_IRON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_WOOD.value, Achievement.COLLECT_IRON.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))'
    }, 

    # 'INSTRUCTION_24_1': {
    #     'instruction': "You need to eat a cow and collect some iron.",
    #     'instruction_paraphrases': [
    #         "Remember to consume beef and gather iron.",
    #         "It's important to gather iron and make sure you eat a cow.",
    #         "Don't forget to partake in some beef and to also collect iron.",
    #         "Remember, it's crucial that you eat some beef and also pick up iron.",
    #         "I need you to consume the meat of a cow and also accumulate some iron."
    #     ],
    #     'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
    #         1 if a in [Achievement.EAT_COW.value, Achievement.COLLECT_IRON.value] else
    #         0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    #     ])),
    #     'str_check_lambda': "conditional_achivments(gd, jnp.array([\
    #         1 if a in [Achievement.EAT_COW.value, Achievement.COLLECT_IRON.value] else\
    #         0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)\
    #     ]))"
    # }, 

    'INSTRUCTION_25_1': {
        'instruction': "Find a tree sapling and collect a piece of coal.",
        'instruction_paraphrases': [
            "Search for a small tree, then obtain a piece of charcoal.",
            "Scour for a young tree, followed by securing a piece of black mineral known as coal.",
            "Hunt down a tree sprout and then gather a coal fragment.",
            "Begin by locating a tree seedling, conclude by retrieving a lump of coal.",
            "Initiate your quest by identifying a sapling, following it up by the collection of a piece of coal."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.COLLECT_COAL.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"conditional_achivments(!gd, jnp.array([1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.COLLECT_COAL.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }
    , 

    
    # 'INSTRUCTION_26': {
        
    #     'instruction': "Gather coal, iron and eat a plant.",
        
    #     'instruction_paraphrases': [
            
    #         "Make sure to Eat a plant after you've collected some coal and iron.",
            
    #         "You must obtain minerals such as coal and iron and then have a plant for lunch.",
            
    #         "Only after you've consumed a plant, start extracting coal and iron.",
            
    #         "Focus on munching on a plant and scavenging for iron and coal.",
            
    #         "Once you've enjoyed your plant meal, your tasks are to collect iron and coal."
            
    #     ],
        
    #     'check_lambda': (lambda gd, ix: conditional_achivments(gd, jnp.array([
            
    #         1 if a in [Achievement.COLLECT_COAL.value, Achievement.COLLECT_IRON.value, Achievement.EAT_PLANT.value] else 0
            
    #         for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            
    #     ]))),
        
    #     'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_COAL.value, Achievement.COLLECT_IRON.value, Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        
    # }, 

    # 'INSTRUCTION_30': \
    #     {
    #     'instruction': "Make a stone pickaxe and set up a furnace please.",
    #     'instruction_paraphrases': [
    #         "Can you construct a stone pick and also place a furnace for me?",
    #         "Do me a favor by building a stone tool for picking and installing a furnace.",
    #         "I need you to fashion a pick made from stone and also position a furnace.",
    #         "Kindly fabricate a stone implement for excavation and implement the installation of a furnace.", 
    #         "If you could, dedicate your efforts towards the assembly of a granite digging apparatus and orchestrate the emplacement of a heating unit."
    #     ],
    #     'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
    #         1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.PLACE_FURNACE.value] else 0
    #         for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    #     ])),
    #     'str_check_lambda':"conditional_achivments(gd, jnp.array([ \
    #         1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.PLACE_FURNACE.value] else 0 \
    #         for a in range(Achievement.MAKE_IRON_SWORD.value+1) \
    #     ])))"        
    #     },

    'INSTRUCTION_31': {
        'instruction': "Collect a drink yet do not construct a stone sword.",
        'instruction_paraphrases': [
            "Grab any type of beverages, but avoid making a rock sword.",
            "Obtain any kind of drinkable liquid, while abstaining from stone blade production.",
            "Acquire a sort of potable yet bypass from producing stone cutter.",
            "Procure liquid meant for drinking but make sure not to forge weapon made of stone.",
            "In your journey, make sure to gather any kind of liquid for hydration, however, be cautious not to fabricate a deadly sword out of a hard mineral rock for your protection."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_DRINK.value else
            -1 if a == Achievement.MAKE_STONE_SWORD.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a == Achievement.COLLECT_DRINK.value else -1 if a == Achievement.MAKE_STONE_SWORD.value else 0 for a in range(Achievement.MAKE_STONE_SWORD.value+1)]))"
    }, 

    # "EAT_AND_PICK": {
    #     "instruction": "Consume a plant, then craft a stone pickaxe and consume another plant.",
    #     "instruction_paraphrases": [
    #         "Eat a vegetable, afterwards create a stone tool for digging and chomp down another vegetable.",
    #         "Feast on some greens, next, fabricate a quarrying tool out of rock and feast on more greens.",
    #         "Nourish yourself with a plant, then manufacture a stone tool meant for excavation and then nourish yourself with another plant.",
    #         "Ingest a plant, after that forge a mining instrument from stone and ingest another plant.",
    #         "Bite into a botanical creation, subsequently create a rock-based excavation implement and finish by taking another bite of the botanical bounty."
    #     ],
    #     "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
    #         1 if a in [Achievement.EAT_PLANT.value, Achievement.MAKE_STONE_PICKAXE.value, Achievement.EAT_PLANT.value] else 0
    #         for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    #     ])),
    #     "str_check_lambda": "conditional_achivments(gd, jnp.array([1 if a in [Achievement.EAT_PLANT.value, Achievement.MAKE_STONE_PICKAXE.value, Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_STONE_PICKAXE.value+1)]))"
    # }, 

    'INSTRUCTION_35': {
        'instruction': "Craft a wooden pickaxe but avoid planting anything",
        'instruction_paraphrases': [
            "Make sure to construct a wooden digging tool, but don't place any plants.",
            "Your task is to build a pickaxe out of wood, however, refrain from cultivating any plants.",
            "I want you to create a wooden mining instrument, but it's important that you don't engage in any plantations.",
            "It's essential that you manufacture a pickaxe utilizing wood, though make sure not to involve yourself in any gardening activities.",
            "The objective is to fabricate a timber-sourced pickaxe while deliberately abstaining from any form of plant deployment."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_WOOD_PICKAXE.value else
            -1 if a == Achievement.PLACE_PLANT.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_PICKAXE.value else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.PLACE_FURNACE.value+1)]))'
    }, 

    'INSTRUCTION_36_1': {
    
        'instruction': "Make sure you have placed the furnace and not built an iron pickaxe. You can make the iron sword though.",
        'instruction_paraphrases': [
            "Confirm that you have set up the furnace. You should not create an iron pickaxe, yet making an iron sword is allowed.",
            
            "Please ensure that you have sited the furnace but have not constructed an iron pickaxe. Nonetheless, forging an iron sword is permissible.",
            "It is imperative that you have installed the furnace and refrained from assembling an iron pickaxe, while on the other hand the creation of an iron sword is acceptable.",
            "The prerequisites are that the furnace has been positioned appropriately and the formation of an iron pickaxe has been avoided. However, you are permitted to produce an iron sword as it's necessary.",
            "You are required to verify the placement of the furnace and abstain from manufacturing an iron pickaxe. However, you have the authorization to craft an iron sword."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            
            1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else
            
            -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0
            
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':
        
            "lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    },

    # 'COLLECT_DRINK_AND_NOT_PLACE_TABLE': \
    #     {
            
    #         'instruction': "Obtain a drink but do not place a table.",
    #         'instruction_paraphrases': [
    #             "From what's available, get a beverage, but don't set up the workbench.",
    #             "Please collect a drinkable liquid, without proceeding to arrange the bench.",
    #             "Grasp a liquid substance and make sure not to lay down the wooden workspace.",
    #             "Ensure to procure an edible liquid, however, dispatching the furniture for work must not be performed.",
    #             "Although nothing should prevent you from acquiring a consumable fluid, do bear in mind to abstain from installing the crafting platform."
    #         ],
    #         'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
    #             [1 if a == Achievement.COLLECT_DRINK.value else -1 if a == Achievement.PLACE_TABLE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]
    #         )),
    #         'str_check_lambda': """conditional_achivments(gd, jnp.array([1 if a==Achievement.COLLECT_DRINK.value else -1 if a==Achievement.PLACE_TABLE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"""
    #     }, 

    'INSTRUCTION_38 1': \
    {
        'instruction': "Collect some coal, place a plant and gather a drink.",
        'instruction_paraphrases': [
            "Gather coal, place foliage and obtain some beverages.",
            "Accumulate some coal, deposit a shrub and acquire liquid refreshment.",
            "Assemble a quantity of carbon, install flora and collect hydration.",
            "Amass lumps of coal, situate a herbaceous plant and garner a type of drinkable liquid.",
            "Cull a pile of burnable carbon substance, position a botanical entity and stock up on a form of potable."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_COAL.value, Achievement.PLACE_PLANT.value, Achievement.COLLECT_DRINK.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_COAL.value, Achievement.PLACE_PLANT.value, Achievement.COLLECT_DRINK.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    # 'INSTRUCTION_39_1': \
    # {
    #     'instruction': "Defeat a zombie and place a table.",
    #     'instruction_paraphrases': [
    #         "Attempt to slay a zombie and after that, set a table.",
    #         "Overcome a zombie challenge and thereafter, station a table.",
    #         "Eliminate a threat from one of the zombies, then, install a workbench.",
    #         "Conquer and take out a zombie, following that, position a counter.",
    #         "Subjugate a zombie and as a subsequent task, situate a desk."
    #     ],
    #     'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
    #         [
    #             1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.PLACE_TABLE.value] else 0
    #             for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    #         ])),
    #     'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.PLACE_TABLE.value] else 0 for a in range(Achievement.PLACE_TABLE.value+1)]))"
    # }, 

    # 'INSTRUCTION_40': {
    #     'instruction': "Collect stones, place them and get a drink.",
    #     'instruction_paraphrases': [
    #         "Gather some rocks, arrange them and get yourself a beverage.",
    #         "Procure pebbles, put them down and secure an elixir.",
    #         "Amass stones, fix them in positions and fetch a drink.",
    #         "Acquire boulders, set them up and score yourself a gulpable liquid.",
    #         "Round up some rocks, plant them firmly and snag a sip."
    #     ],
    #     'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
    #         1 if a in [Achievement.COLLECT_STONE.value, Achievement.PLACE_STONE.value, Achievement.COLLECT_DRINK.value ] else 0
    #         for a in range(Achievement.MAKE_IRON_SWORD.value+1)
    #     ])),
    #     'str_check_lambda': "conditional_achivments(...)"
    # }, 

    'INSTRUCTION_41_1': 
    {
        'instruction': "Place a plant first, then place the furnace",
        'instruction_paraphrases': [
            "First put the plant, then put the furnace",
            "Before you put the furnace, make sure you have placed the plant",
            "Only after placing a plant should you place the furnace",
            "In order of operations, place the plant before you place the furnace",
            "After putting down the plant, your next task should be to put down the furnace"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_PLANT.value, Achievement.PLACE_FURNACE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)]
        )),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_PLANT.value, Achievement.PLACE_FURNACE.value] else 0 for a in range(Achievement.PLACE_FURNACE.value+1)]))"
    }, 
    'INSTRUCTION_43': {
        'instruction': "First, collect some rocks. Then, make a stone sword.",
        'instruction_paraphrases': [
            "Gather some stones and then craft a sword out of stone.",
            "To start off, you should amass some rocks. Following that, use the rocks to put together a sword made of stone.",
            "You should begin by acquiring a couple of stones. Subsequently, you should manufacture a sword by using the stones.",
            "At first, you need to get a hold of some stones. Once you've done that, the next step is to construct a stone sword.",
            "To begin, amass a stockpile of rocks. With those in hand, your next objective is to assemble a weapon of warfare, specifically a sword, composed entirely of stone."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            
            1 if a in [Achievement.COLLECT_STONE.value, Achievement.MAKE_STONE_SWORD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
            
        'str_check_lambda': "conditional_achivments(gd, jnp.array([ \
            1 if a in [Achievement.COLLECT_STONE.value, Achievement.MAKE_STONE_SWORD.value] else 0 \
            for a in range(Achievement.MAKE_STONE_SWORD.value+1)]))"
        }
   , 

    'INSTRUCTION_44': 
        {
        'instruction': "Gather some diamonds and produce a stone pickaxe.",
        'instruction_paraphrases': [
            "Acquire a few gems and make a stone mining tool.",
            "Procure some sparklers and construct a rock digging instrument.",
            "Obtain several diamonds and fabricate a stone pick.",
            "Secure a handful of precious stones and assemble a stone picker.",
            "Ensure to amass diamonds and put together a stone excavator."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.MAKE_STONE_PICKAXE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.MAKE_STONE_PICKAXE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }, 

    'INSTRUCTION_45': {
        'instruction': "Create a wooden pickaxe and then gather some diamonds.",
        'instruction_paraphrases': [
            "Begin by crafting a wood pickaxe and proceed to collect diamonds.",
            "An essential task for you is to fabricate a timber chopping tool, and subsequently, you must discover precious stones.",
            "The first chore is to construct a lumber pick and follow it up with diamond acquisition.",
            "Primarily, the technical creation of a wooden pick is required. Pursue this with the excavation of diamonds.",
            "Post the manufacture of a pick out of wood, search and harvest the glistening diamonds."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.COLLECT_DIAMOND.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.COLLECT_DIAMOND.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_46': 
    {
        'instruction': "Please, gather some coal, create an iron pickaxe and eat a plant.",
        'instruction_paraphrases': 
        [
            "Collect coal, craft an iron pick, and consume vegetable.",
            "Can you obtain some coal, forge a pickaxe made of iron and ingest a herb?",
            "I need you to secure some coal, manufacture a pickax using iron and devour a flora.",
            "Kindly accumulate charred remains, produce a mining tool of iron and munch on greens.",
            "Please assemble some cinder remnants, fabricate an excavation instrument out of iron, and ingest a leafy foodstuff."
        ],
        'check_lambda': 
            lambda gd, ix: conditional_achivments(gd, 
                jnp.array([1 if a in [Achievement.COLLECT_COAL.value,
                Achievement.MAKE_IRON_PICKAXE.value,
                Achievement.EAT_PLANT.value] 
            else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_COAL.value,Achievement.MAKE_IRON_PICKAXE.value,Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
     }, 

        'INSTRUCTION_47': 
        {
            'instruction': "Wake up, place the workbench and avoid planting anything",
            'instruction_paraphrases': [
                "Rise, set up the woodworking table, and refrain from putting down any flora",
                "Get up, put the craft table in place and avoid any cultivation",
                "Stand up, position your bench for crafting and make sure you don't do any planting",
                "Get yourself up, position your carpentry table strategically and make sure not to engage in any cultivation activities",
                "Arise, systematically set up your table for crafting and remember you aren't supposed to plant anything",
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd,jnp.array([
                1 if a in [Achievement.WAKE_UP.value, Achievement.PLACE_TABLE.value] else
                -1 if a == Achievement.PLACE_PLANT.value else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.WAKE_UP.value, Achievement.PLACE_TABLE.value] else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

        # 'INSTRUCTION_48_COLLECT_COAL': 
        #     {
        #         'instruction': "Collect some coal.",
        #         'instruction_paraphrases': [
        #             "Please acquire some coal.",
        #             "Your task is to get some anthracite.",
        #             "Will you please gather some coals for me?",
        #             "It is necessary for you to procure some carbon.",
        #             "Can you kindly proceed with the collection of some charcoal?"
        #         ],
        #         'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
        #             1 if a == Achievement.COLLECT_COAL.value else 0
        #             for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        #         ])),
        #         'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a == Achievement.COLLECT_COAL.value else 0 for a in range(Achievement.COLLECT_COAL.value+1)]))"
        #     },
#             'INSTRUCTION_48_WAKE_UP': 
#             {
#                 'instruction': "Please wake up.",
#                 'instruction_paraphrases': [
#                     "Can you wake up?",
#                     "I'd like you to awaken.",
#                     "Would you kindly stir yourself?",
#                     "It's time for you to rise.",
#                     "It would be great if you could rouse yourself."
#                 ],
#                 'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
#                     1 if a == Achievement.WAKE_UP.value else 0
#                     for a in range(Achievement.MAKE_IRON_SWORD.value+1)
#                 ])),
#                 'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a == Achievement.WAKE_UP.value else 0 for a in range(Achievement.WAKE_UP.value+1)]))"
#             }
#    , 
   
    'COLLECT_IRON_AND_NOT_PLACE_PLANT': 
        {
            'instruction': "Get your hands on some iron ore but make sure not to plant anything.",
            
            'instruction_paraphrases': [
                "Procure an iron mineral, and refrain from engaging in any horticultural activities.",
                "Your task is to accumulate the iron substance, and remember, you must abstain from executing any planting or cultivation.",
                "Acquire an iron compound ensuring that you do not contribute to planting anything.",
                "It's crucial to gather iron yet abstain from placing any flora.",
                "Initiate the collection of iron entities, but the planting of any vegetation is prohibited."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                    1 if a == Achievement.COLLECT_IRON.value else
                    -1 if a == Achievement.PLACE_PLANT.value else 0
                    for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
                ])), 
                
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a == Achievement.COLLECT_IRON.value else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.PLACE_PLANT.value + 1)]))"
        } ,
    
}
    

if __name__ == "__main__":
    import json
    instructions = []
    for key in easy.keys():
        instructions.append(easy[key]['instruction'])
        instructions += easy[key]['instruction_paraphrases']
    with open("instructions_achivments_easy_train.json", "w", encoding="utf-8") as json_file:
        json.dump(instructions, json_file, ensure_ascii=False, indent=4)