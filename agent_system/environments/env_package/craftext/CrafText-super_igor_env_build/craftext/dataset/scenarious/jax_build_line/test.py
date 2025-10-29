from craftext.enviroment.craftext_constants import BlockType
from craftext.enviroment.scenarious.checkers.target_state import BuildLineState as AchievmentClass
from craftext.enviroment.scenarious.checkers.target_state import TargetState
from craftext.enviroment.craftext_constants import Scenarios

def create_target_state(block_type:int, size:int, is_diagonal:bool):
    target_achievements = AchievmentClass(block_type=block_type, size=size, is_diagonal=is_diagonal, radius=10)
    return TargetState(building_line=target_achievements)

easy_test_parafrased = {
    "INSTRUCTION_STONE_4": {
        "instruction": "Make a line of stone with four blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a streak of rock comprising quartet sections.",
            "By gathering four disparate blocks, one can assemble a formation of solid stone.",
            "Through the accumulation of four distinct sections, one can erect an arrangement of unyielding rock.",
            "In the silent working of hands and the stubborn resolve of the heart, let an array of stone come to being, born of four separate entities, each a testament of the earth's might.",
            "In the patient dance of construction, four pieces of stone find unity, giving birth to a line of stone, a small but sturdy testament to the foundational elements of our world."
        ],
        "arguments": create_target_state(BlockType.STONE, 4, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False)"
    },
    "INSTRUCTION_FURNACE_3_DIAGONAL": {
        "instruction": "Check if there is a diagonal line of furnace with a size of three on the map.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify if a three-unit-long slanting row of stove is evident on the chart.",
            "Could you examine whether the map carries a depiction of a furnace line running diagonally for three units?",
            "Inspect whether the chart holds an evidence of a slanting sequence of hearth that notably extends to three units.",
            "In sooth, make thine inquiry if a diagonal lineage of hearth, measured three in number by a master's hand, be marked upon yon parchment map.",
            "Within the kindness of the map, canvass the possibility of a diagonally coursing river of three furnaces speaking their own geometric language."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 3, check_diagonal=True)"
    },
    "INSTRUCTION_PLANT_4": {
        "instruction": "Check if there's any line of Plants arranged in four.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Inspect whether there exists a row of vegetation set in quartet.",
            "Could you verify if, perchance, there is a sequence of shrubbery assembled as a quartet?",
            "Ascertain if, perchance, there exists an arrangement of flora organized in sets of four.",
            "In the verdant domain, you shall find no rest until you seek out whether, indeed, there exists an orderly row of leaf-bearing life, meticulously placed in counts of four.",
            "Within the world's breathtaking tapestry of green, seek out truths: might there be an understated pattern woven of flora, assembled as whispering quaternities?"
        ],
        "arguments": create_target_state(BlockType.PLANT, 4, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 4, check_diagonal=False)"
    },
    "INSTRUCTION_PLANT_3": {
        "instruction": "Check for a line of plants with a length of three",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Look for a row of vegetation three units long",
            "Ensure to verify the existence of a sequence of plants that extends to a distance equivalent to three",
            "Investigate to ascertain a trail of flora extending a distance of trio units",
            "Seek thou the string of greenery, thrice segmented in length, deep-rooted beneath the unbroken heavens",
            "In your search, let your eyes find a path; a vessel inhabited by plants, each connected to the next, forming a string that spans just three of our known units."
        ],
        "arguments": create_target_state(BlockType.PLANT, 3, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 3, check_diagonal=False)"
    },
    "INSTRUCTION_STONE_3": {
        "instruction": "Form a line of stones of length 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a row of pebbles spanning three units.",
            "Assemble the pebbles, standing one next to another, extending to a total of three units in length.",
            "Constitute a series of rock fragments, standing shoulder to shoulder, stretching across the expanse equal to three units.",
            "Draw forth a series of stones, thrice counted, in a line as straight as a ranger's arrow, stretching under the ancient sky.",
            "In the depth of your quiet contemplation, summon into being a row of stones, each a story unto itself, and let this procession extend to the reach of three."
        ],
        "arguments": create_target_state(BlockType.STONE, 3, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 3, check_diagonal=False)"
    },
    "INSTRUCTION_PLANT_2": {
        "instruction": "Check if there is a line of Plant blocks of size 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify if a succession of Plant units of size 2 exists.",
            "Determine if a continuous sequence, precisely of Plant blocks, amounting to a size of 2 is present.",
            "Contemplate whether or not there is a steady chain of synonymous Plant entities, specifically of measurement 2.",
            "Perchance there lies an unbroken lineage, two score in size, of blocks hewn of Plant?",
            "Question the presence of a seam, woven not of thread but of Plant blocks, each a duo in quantification."
        ],
        "arguments": create_target_state(BlockType.PLANT, 2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 2, check_diagonal=False)"
    },
    "INSTRUCTION_CRAFTING_TABLE_3_DIAGONAL": {
        "instruction": "Create a diagonal line of crafting tables, each one block apart for a size of three.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a skew series of workshop benches, each separated by a single cube for a dimension of three.",
            "In order to form a length of three, set up a lineup of crafting tables, each arranged diagonally and separated by an individual block.",
            "For the purpose of creating a dimension of three, fabricate a slanting arrangement of work stations, each segregated by one solitary cube.",
            "In the realm of the fabricators, compose ye a line skewed in its path, filled with crafting pedestals, every single one distanced by a lone block in semblance of the trio's size.",
            "Drawing upon the elements in the crafters’ space, weave together an inclined file of creation tables, each solitary cube held separate, to construct the triple expanse."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True)"
    },
    "INSTRUCTION_CRAFTING_TABLE_2_DIAGONAL": {
        "instruction": "Check on the game map for a diagonal line of Crafting Table blocks that are of length 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Inspect the gaming landscape for a skewed array of Crafting Table units of dimension two.",
            "Look for a line, one that takes a diagonal path across the game map, comprised of Crafting Table blocks and has a length equating to two.",
            "Scout the virtual terrain, seeking a diverging concatenation of Crafting Table entities, whose magnitude is two.",
            "Let thine eyes linger upon the parchment of gameplay, there to find an oblique lineage of blocks, called yonder Crafting Table, which extend but by twain in measure.",
            "In the intricate landscape of the game, search with quiet determination for a line of Crafting Table elements, subtly interrupted, stretching in a manner oblique, and barely extending beyond the count of one to a modest two."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=True)"
    }
}

easy_test_other_paramets = {
    "INSTRUCTION_CRAFTING_TABLE_2": {
        "instruction": "Check for the existence of a line of crafting table blocks of size 2 within the game.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Look for a line containing 2 blocks of crafting tables.",
            "Can you spot a line within the game that constitutes 2 blocks, specifically crafting table blocks?",
            "Explore the surroundings for a linear arrangement of 2 crafting table blocks.",
            "Ensure to identify a sequence containing a pair of crafting table blocks in a linear arrangement.",
            "In the game, confirm if there is a linear pattern consisting of a pair of crafting table blocks."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, size=2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
    "INSTRUCTION_STONE_2_DIAGONAL": {
        "instruction": "Form a diagonal line made up of stones of length 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a slanted row of rocks spanning a distance of 2 spaces.",
            "Position two stones in a diagonal formation.",
            "Erect a diagonal pathway composed of 2 stones.",
            "Fabricate a two-part diagonal series using stones.",
            "Assemble a sequence of rocks arranged diagonally with a size twice that of a single rock."
        ],
        "arguments": create_target_state(BlockType.STONE, size=2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=True)"
    },
    "INSTRUCTION_PLANT_4_DIAGONAL": {
        "instruction": "Create a diagonal line of Plant type blocks that is 4 blocks long.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Form a 4-block long diagonal line with blocks of the Plant type.",
            "On the game map, arrange Plant blocks in a diagonal line that spans four squares.",
            "Can you build a diagonal line, with a size of four blocks, and use only Plant blocks?",
            "Construct a line that's 4 units long, using only Plant blocks. Make sure the line is diagonal.",
            "Could you form a stretch of Plant blocks diagonally across 4 grid spaces?"
        ],
        "arguments": create_target_state(BlockType.PLANT, size=4, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 4, check_diagonal=True)"
    },
    "INSTRUCTION_STONE_2": {
        "instruction": "Make sure there is a row of two stones in your proximity.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Ensure a line of couple of rocks exists somewhere close to you.",
            "Verify if you find a couple of stones lined up nearby.",
            "Ascertaining there are two stones in a sequence situated near you.",
            "Could you confirm for me if a two-stone line is in your vicinity?",
            "Authenticate whether or not a row of two stones is located nearby."
        ],
        "arguments": create_target_state(BlockType.STONE, size=2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=False)"
    },
    "INSTRUCTION_PLANT_3_DIAGONAL": {
        "instruction": "Create a diagonal line of plant blocks with a length of 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Form a 3 block long diagonal line using plant blocks.",
            "Position three plant blocks in a diagonal line.",
            "Create a diagonal row consisting of three types of plant blocks.",
            "Build a line of plant blocks diagonally with a size of three.",
            "Fabicate a line diagonally with three blocks and make sure they are plant blocks."
        ],
        "arguments": create_target_state(BlockType.PLANT, size=3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 3, check_diagonal=True)"
    },
    "INSTRUCTION_PLANT_2_DIAGONAL": {
        "instruction": "Make a diagonal line of plants, 2 blocks long.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a line of flora, 2 blocks in length, in a diagonal orientation.",
            "Place two plant blocks in a line, arranging them diagonally.",
            "Form a line composing of two plants, positioned diagonally.",
            "Organize a 2-block long diagonal sequence with plants.",
            "Construct a diagonal sequence using two pieces of flora."
        ],
        "arguments": create_target_state(BlockType.PLANT, size=2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 2, check_diagonal=True)"
    },
    "INSTRUCTION_FURNACE_4": {
        "instruction": "Verify if there is a line of Furnaces of length 4 in the game.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Please ensure there is a straight line consisting of four Furnaces in the game.",
            "Confirm if you can find a continuous line of four Furnaces in the game.",
            "Please Authenticate a straight alignment of Furnaces with a count of 4 in the game.",
            "Can you authenticate a linear arrangement of Furnaces, where the quantity is 4, in our current game?",
            "Would you be so kind as to ascertain the existence of a sequence of Furnaces, comprising four units in a straight line within the game?"
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=4, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 4, check_diagonal=False)"
    },
    "INSTRUCTION_STONE_4_DIAGONAL": {
        "instruction": "Form a diagonal line of stone blocks of size 4.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a diagonal streak of stone of four blocks.",
            "Arrange a series of four boulders in a diagonal line.",
            "Produce a diagonal row of quartet rocks.",
            "Set up an oblique chain of stone blocks, each with a total of four units.",
            "Structure a diagonal stone linkage of exactly four."
        ],
        "arguments": create_target_state(BlockType.STONE, size=4, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=True)"
    },
    "INSTRUCTION_CRAFTING_TABLE_4_DIAGONAL": {
        "instruction": "Check if there's a diagonal line of crafting tables at least four blocks long.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "See if there is a slanted line of at least four workbenches.",
            "Can you find a diagonal array of 4 or more craft tables in a sequence?",
            "Examine if there's a sequence of a minimum of four artisan tables in a diagonal line.",
            "Check to see if there's a series of craft stations in a slanting line, with at least four blocks.",
            "Look for a sequence of crafting tables aligned diagonally, totaling a minimum of four items."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, size=4, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 4, check_diagonal=True)"
    }
}



# TODO: check complexity
medium_test_parafrased = {
    "INSTRUCTION_STONE_2_DIAGONAL": {
        "instruction": "Create a diagonal line of stones of size two.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Formulate a slanted course of pebbles of magnitude two.",
            "Begin the process of constructing a line, diagonal in direction, utilizing stones that are of the second size.",
            "Incline a trajectory, utilizing not immense rocks but those of the second magnitude, carefully placed at a slant.",
            "In the hands of skilled creators did the line manifest, veering not straight nor curved but diagonal, crafted of hard stones, neither most large nor insignificant but of size two in the counting.",
            "Two-sized stones whisper of their unwilling straightness and yearn to form diagonal lines, their voices creating stories in the patterns they form."
        ],
        "arguments": create_target_state(BlockType.STONE, 2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=True)"
    },
    "INSTRUCTION_CRAFTING_TABLE_2": {
        "instruction": "Check if there is a line of Crafting Tables, at least two in size.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Inspect for an array of Crafting Tables, spanning at least two in length.",
            "Examine if a consecutive series of Crafting Tables exists, a series that is no less than two in size.",
            "Scrutinize for a consecutive series of facilities for crafting, a series that must span a minimum of two.",
            "Beyond the shadows and flickering torchlight, seek yea whether there exists a succession of Masters' benches, with two being in their count at the least.",
            "Gaze deeply, and see if there unfolds a path of Crafting Tables, a path little less than two in its measure, like a constellation newly discovered."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
    "INSTRUCTION_FURNACE_3_DIAGONAL": {
        "instruction": "Form a diagonal line of furnaces of length 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a slanting row of kilns that is 3 units long.",
            "Establish a line of furnaces, measuring 3 units in an oblique orientation.",
            "Arrange in an inclined manner a series of smelters extending to a measure of 3 units.",
            "In the manner of the great craftsman, an askew path of fire chambers, of an expanse thrice, is to be sculpted.",
            "Draw forth into existence, a skewed pattern of burning hearths, three times the length of the singular form, a sentiment of craft and creation distilled into its simple measure."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 3, check_diagonal=True)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_2_DIAGONAL": {
        "instruction": "Form a diagonal line of enchantment ice tables of length 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a slanted row of bewitching frost furniture measuring two units long",
            "In order to establish a two-unit long mesmeric chain of icy furniture, one needs to align them diagonally",
            "To manifest a hypnotic sequence of glacial desks, sorted on the bias and extent to the course of two measures, is a task in hand",
            "Call forth the creation of a line of magical chill-crafted tables, aligned unswervingly across the slant, and reaching the measure of two in length, just so",
            "In the timeless act of forming, dream into existence a diagonal lineage of icy constructs, tables imbued with charm, holding two lengths within their existence."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, 2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_ICE, 2, check_diagonal=True)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_7_DIAGONAL": {
        "instruction": "Check a diagonal line of enchantment table fire blocks of size 7",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Inspect a slanted row of bewitching table flame cubes of magnitude 7",
            "Ensure that there exists a line, diagonally placed, full of enchantment table fire blocks, their size being 7",
            "Evaluate a beveled series of captivating table inferno elements of capacity 7, composed in a slanting manner",
            "Verify thou the direction of the oblique path flaunting the enthralled table's blazing sqares of seven in size, akin to the fire-dance of dragons",
            "Gaze towards the askew alignment of mystical table pyro blocks, seven in their count, resembling constellations woven in the loom of existence."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, 7, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_FIRE, 7, check_diagonal=True)"
    },
    "INSTRUCTION_FURNACE_2": {
        "instruction": "Check if there is a line of two furnaces",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Examine whether a row of two stoves exists",
            "Can one confirm the presence of a line that hosts two furnaces?",
            "Is it possible to ascertain the presence of a queue containing a duo of heaters?",
            "Peer thou into the case and seek if there exists a row of twain hearths of heat and flame",
            "In the vast expanse of possibility, one might ponder upon the existence of a sequence of two fire-bearers."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=False)"
    },
    "INSTRUCTION_PLANT_7_DIAGONAL": {
        "instruction": "Create a diagonal line of plants that is seven units long.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Design a slanting row of flora that measures seven units in length.",
            "A task at hand is to produce a line that tilts diagonally, made of plants and reaching a distance of seven units.",
            "Fashion a sloping array of greenery, maintaining its measure to be seven units, thus creating an angling line.",
            "From thy first unit, proceed in the vein of a compass point 'twix North and East to the seventh, planting all the while in the fashion of a line drawn askew.",
            "In the quiet geometry of existence, conceive a path for life itself, born from seven units of space and time, articulated as flora leaning towards the setting sun."
        ],
        "arguments": create_target_state(BlockType.PLANT, 7, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 7, check_diagonal=True)"
    },
    "INSTRUCTION_FURNACE_7_DIAGONAL": {
        "instruction": "Create a diagonal line of Furnace with 7 blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a slanted row of Furnace containing seven units.",
            "A noteworthy task at hand would be the formation of a diagonal line that will feature Furnace with precisely seven blocks.",
            "As we consider the task of fabricating a slanting row that, importantly, includes Furnace, one will observe that the count of blocks utilized is seven.",
            "One must bring forth, with meticulous care, a skewed succession of Furnace, born of seven sturdy stones, akin to the spine of a mighty dragon lying beneath the solemn heavens.",
            "Bring into existence a diagonally drawn line of Furnace, home to seven blocks, a symbol of balance, merging simple geometry with a hint of mechanical vigour."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 7, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 7, check_diagonal=True)"
    },
    "INSTRUCTION_STONE_2_STRAIGHT": {
        "instruction": "Verify a 2-block line of stone arranged linearly in the game space.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Confirm a twin-section row of rock ordered in a straight pattern within the gaming environment.",
            "Inspect whether a row of stone, distributed in a linear fashion and constituting two blocks, has been properly laid out in the confines of the game world.",
            "Affirm the linear configuration of a twin-part sequence of rock that has been systematically coordinated within the boundaries of the interactive arena.",
            "Perchance, in the wide and wondrous realm of the game, seek to assure thyself of the true alignment of a line of stone, two blocks in their count, laid in the manner of the ancients in a straight and noble line.",
            "In the vast stretches of the game space, let it be a task to affirm the existence of a duet of rocks, orderly in their alliance, exhibited as a straight line, a hint of the patterns of the universe within the playful world."
        ],
        "arguments": create_target_state(BlockType.STONE, 2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=False)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_6_DIAGONAL": {
        "instruction": "Form a line containing six Enchantment Fire Tables arranged diagonally.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Establish a queue consisting of half a dozen Enchantment Fire Tables positioned slantingly.",
            "By positioning six Enchantment Fire Tables slantingly, you can create a line.",
            "In the action of establishing a queue, include half a dozen Enchantment Fire Tables that are strategically positioned in a slanting manner.",
            "In the realm where shadows and light dance in harmony, heed my words. Bestow order onto the wild, creating a phalanx of six Enchantment Fire Tables, arranged in the mysterious language of the diagonal.",
            "In a world where fires have a place of their own, imagine-six Enchantment Fire Tables. Let them form a line that bends to some unseen geometry, hinting at unseen forces at work - do arrange them diagonally."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, 6, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_FIRE, 6, check_diagonal=True)"
    },
    "INSTRUCTION_STONE_6_DIAGONAL": {
        "instruction": "Form a diagonal line of stone blocks with length 6.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a slanting sequence of rock units measuring six in length.",
            "Incorporate six stone blocks in the formation of a skew line.",
            "Build a bias arrangement of boulder fragments which shall measure six in their combined length.",
            "At once, my good sir, weave ye a line from yonder stone blocks, set askew, running to a total length of six units in your noble task.",
            "In the grand scheme, a diagonal passage made of solid stone blocks is to be formed, with their collective length amounting to the count of six."
        ],
        "arguments": create_target_state(BlockType.STONE, 6, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 6, check_diagonal=True)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_4_DIAGONAL": {
        "instruction": "Make a diagonal line of enchantment table ice blocks with a size of four.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Craft an oblique sequence of charm bench ice cubes of a magnitude of four.",
            "Construct an ice path, of four blocks in length, made from enchanted table ice.",
            "Under the concept of an oblique formation, fashion a row of bewitchment slab frost squares that carry the property of four in magnitude.",
            "Under the ancient starlight, craft thou a line of witchery, crossing the virgin snow in diagonal fashion, wherein the count shall not exceed four, the number of harmony.",
            "Shape a walkway of mystery, a frozen road crafted from the heart of enchantment itself, oblique in its path and not more than four in the realm of its physicality."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, 4, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_ICE, 4, check_diagonal=True)"
    },
    "INSTRUCTION_FURNACE_3": {
        "instruction": "Form a line of furnace blocks, 3 blocks in length.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a row of heating units, each 3 units long.",
            "The task at hand is to produce a sequence of furnace blocks, each counting to a total of three in terms of length.",
            "Fabricate a succession of thermal blocks, which are, each, a trio in extent.",
            "By the hands of Ye who forms and molds, there shall come into being a line of blocks as to dress the furnace, each measuring a length of thrice.",
            "In this vast world of ours, your hands are to create - giving birth to a beautiful series of blocks, hewn for the heart of a furnace, each extending over three lengths of space itself."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 3, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 3, check_diagonal=False)"
    },
    "INSTRUCTION_PLANT_7": {
        "instruction": "Create a line of plants with a length of 7 blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Form a series of flora spanning 7 units.",
            "Introduce an arrangement where, with a length equivalent to 7 sections, fits a succession of plants.",
            "Establish a sequence of verdure expanded across 7 segments.",
            "In the dominion of the horticulturist, it is desired to lay a progression of green living things, a length of which should obtain the count of seven fragments.",
            "Envision crafting an assembly, a living poem, where the verses are plants and each of them holds its place for seven steps in time."
        ],
        "arguments": create_target_state(BlockType.PLANT, 7, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 7, check_diagonal=False)"
    },
    "INSTRUCTION_STONE_4_DIAGONAL": {
        "instruction": "Form a diagonal line of four rocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a slanting row of four stones",
            "By the arrangement of four stones, one can introduce a diagonal line",
            "Establish a leaning line by organizing four pebbles",
            "By craftsmanship of keen sight and steady hand, ordain thou a twisting line of four ancient stones",
            "In the silent tongue of the world, whisper a trail of four stones, diagonal, as if tracing the slope of a far-off mountain"
        ],
        "arguments": create_target_state(BlockType.STONE, 4, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=True)"
    },
    "INSTRUCTION_CRAFTING_TABLE_7_DIAGONAL": {
        "instruction": "Check if there is a diagonal line of Crafting Tables of size 7.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify the presence of a diagonal chain of Crafting Tables with a length of 7",
            "Could it be that a diagonal line consisting of Crafting Tables, each one precisely aligning to form a collection of 7, might be present?",
            "Ascertain if indeed one might discover a slanted sequence constituting Crafting Tables, its scale measuring to unite seven elements",
            "Dost thou witness a slanting lineage of Crafting Tables, a gathering amassed by seven, perhaps?",
            "In the vast expanse of this realm, one might question if a thread links seven Crafting Tables, aligning them just so to form a diagonal pattern that tells a tale of intricate design and precise count."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 7, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 7, check_diagonal=True)"
    }
}


medium_test_other_paramets = {
    "INSTRUCTION_STONE_7": {
        "instruction": "Create a diagonal line of stones with a size of 7.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Make a slanted line of rocks with a span of 7.",
            "Form a line of seven stones on an angle.",
            "Sketch a diagonal sequence using seven rocks.",
            "Develop a diagonal row of stones that stretches to a length of 7.",
            "Construct a diagonal arrangement of boulders with a total count of seven."
        ],
        "arguments": create_target_state(BlockType.STONE, size=7, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 7, check_diagonal=True)"
    },
    "INSTRUCTION_FURNACE_5": {
        "instruction": "Form a line of furnaces that is five blocks in length.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a straight line made up of five furnaces.",
            "In a straight line, place five furnace blocks.",
            "Assemble a linear collection of five furnace blocks.",
            "Set up a raw of five furnaces in sequential order.",
            "Arrange five furnaces in a straight line."
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=5, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 5, check_diagonal=False)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_3": {
        "instruction": "Check if there's a diagonal line of enchantment fire tables of size 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify a line of three fire enchantment tables aligned diagonally.",
            "Can you identify a diagonal sequence of three enchantment fire tables?",
            "Ensure there's a triplet of enchantment fire tables configured diagonally.",
            "Confirm the presence of a row of three fire enchantment tables that's diagonally arranged.",
            "Find a diagonal alignment of three enchantment fire tables and validate it."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, size=3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_FIRE, 3, check_diagonal=True)"
    },
    "INSTRUCTION_CRAFTING_TABLE_4": {
        "instruction": "Check if there is a line of crafting tables of length 4.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Examine if four crafting tables are aligned in a line.",
            "Scan if there exist a consecutive row of four crafting tables.",
            "Can you confirm if there is a straight line made up of four workbenches?",
            "Investigate if there's a linear arrangement of four crafting stations.",
            "Please verify if we have a sequence of four workstations forming a straight line."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, size=4, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 4, check_diagonal=False)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_6": {
        "instruction": "Check if there's a line of enchantment tables filled with ice, with a length of six.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify whether a line of icy enchantment tables, six in length, exists.",
            "Explore the presence of a sequentially arranged line, containing six enchantment tables of the ice type.",
            "Can you find a consecutive line made up of six frost-laden enchantment tables?",
            "Look for a linear arrangement comprising six icy magic workstations laid out in succession.",
            "Confirm the existence of an uninterrupted line formation consisting of precisely six enchantment tables, each infused with the element of ice."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, size=6, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_ICE, 6, check_diagonal=False)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_2": {
        "instruction": "Check if there is a diagonal line of enchantment table of fire with a length of two",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Please verify whether there's a sequence of two fire enchantment tables in a diagonal line",
            "Inspect the surroundings and see if there is a diagonal arrangement of two fire enchanting tables",
            "Could you check for a series of two fire enchantment platforms that are placed in a diagonal line?",
            "Ascertain whether you can see a pattern of two fire enchantment tables arranged diagonally?",
            "Evaluate the vicinity for a diagonal sequence of two fire enchantment stations"
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, size=2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_FIRE, 2, check_diagonal=True)"
    },
    "INSTRUCTION_STONE_3": {
        "instruction": "Check for a line of rocks that has a length of 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Confirm if there's a three-block-long line of stones.",
            "Can you find a stretch of rocks that spans three blocks?",
            "Do you see a stone line that consists of 3 consecutive blocks?",
            "I want you to verify if there exists a sequence of stones arranged in a line with a length of three.",
            "Establish whether there is a direct arrangement of three blocks worth of stones."
        ],
        "arguments": create_target_state(BlockType.STONE, size=3, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 3, check_diagonal=False)"
    },
    "INSTRUCTION_FURNACE_2": {
        "instruction": "Make a diagonal line of two blocks using furnaces.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a furnace line diagonally made of two blocks.",
            "Arrange two furnace blocks in a diagonal line.",
            "Fabricate a diagonal sequence that consists of a pair of furnace blocks.",
            "Using furnaces, form a bidirectional line that is positioned diagonally.",
            "A pair of furnace cubes is required to be assembled into a diagonal formation."
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=True)"
    },
    "INSTRUCTION_PLANT_3": {
        "instruction": "Make a diagonal line of plant blocks with length 3",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "I need a line of plant blocks, stretching out diagonally for 3 blocks",
            "Create a series of three plant blocks in a diagonal pattern",
            "Formulate a line with three plant items, but ensure it's arranged diagonally",
            "Can you construct a diagonal line, that is 3 blocks long, with plant blocks?",
            "Place plant blocks to form a diagonal sequence that extends for three blocks"
        ],
        "arguments": create_target_state(BlockType.PLANT, size=3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 3, check_diagonal=True)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_3": {
        "instruction": "Ensure that there is a straight line of three Enchantment Ice Table blocks in the game map.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Check if there exists a sequence of three consecutive Enchantment Ice Table blocks in the game layout.",
            "Isn't the game map arranged in such a way that it consecutively delivers three Enchantment Ice Tables?",
            "Confirm if we can spot three successive blocks of Enchantment Ice Tables in the current game map.",
            "Verify the presence of a successive trio of Enchantment Ice Table blocks in the game.",
            "Can you determine whether or not there exists a line of three Enchantment Ice Table blocks in a row in our current game map?"
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, size=3, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_ICE, 3, check_diagonal=False)"
    },
    "INSTRUCTION_FURNACE_5_DIAGONAL": {
        "instruction": "Please find and check for a diagonal line of furnaces that contains 5 in a row.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "We need to inspect for furnace alignment. Could you verify if there's a sequence of 5 furnaces constructed in a diagonal pattern?",
            "You need to identify if there's a series of 5 furnaces, arranged diagonally.",
            "What I want you to do is locate a row of 5 furnaces positioned in an oblique line.",
            "Overall objective is the identification of a sequence of 5 furnaces in a line, but not a straightforward one, rather a diagonal line.",
            "The mission I have for you involves furnaces. Look for a formation where 5 of them are aligned in a diagonal sequence."
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=5, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 5, check_diagonal=True)"
    },
    "INSTRUCTION_CRAFTING_TABLE_6": {
        "instruction": "Check if there's a line of six crafting tables in the game.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Inspect if a sequence of six crafting tables is present in the game",
            "Do examine if a row of six crafting benches exists in the game",
            "Can you verify if there's a line of six workbenches in the current game state?",
            "Conduct a check to confirm if the game contains a succession of six carpentry tables.",
            "Could you ascertain if there prevails a continuous array of six joiner's benches in the game environment?"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, size=6, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 6, check_diagonal=False)"
    },
    "INSTRUCTION_PLANT_5": {
        "instruction": "Check if there is a line of plant blocks of length 5.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Can you check for a string of five plant blocks in a row?",
            "Look out for an arrangement of vegetation blocks that stretches out for five blocks.",
            "Are we able to find any sequence of plant blocks that consists of a length of five?",
            "Validate the existence of a consecutive line of five blocks which are of plant type.",
            "Could you verify whether we have a series of five blocks arranged consecutively that represent plants?"
        ],
        "arguments": create_target_state(BlockType.PLANT, size=5, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 5, check_diagonal=False)"
    },
    "INSTRUCTION_FOUNTAIN_7": {
        "instruction": "Arrange a diagonal line of torches that will be seven blocks long.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Set a line of torches to run diagonally and make it seven blocks in length.",
            "Assemble a torch line of seven blocks on a diagonal.",
            "Place seven torches in a diagonal line.",
            "Construct a line using seven torches and angle it diagonally.",
            "Create a diagonal alignment using seven torches."
        ],
        "arguments": create_target_state(BlockType.FOUNTAIN, size=7, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FOUNTAIN, 7, check_diagonal=True)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_2_STRAIGHT": {
        "instruction": "Form a line of Enchantment Table Ice with a length of 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a straight line consisting of 2 Ice Enchantment Tables",
            "Construct a row of 2 Enchantment Table Ice units",
            "Arrange two Enchantment Table Ices in a linear format",
            "Using two Ice Enchantment Tables, create a line",
            "Put two Enchantment Table Ices in a row."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, size=2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.ENCHANTMENT_TABLE_ICE, 2, check_diagonal=False)"
    },
    "INSTRUCTION_FURNACE_3": {
        "instruction": "Identify if there is a line of three furnaces",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Can you spot a chain of three furnaces?",
            "Look for a series of three heating units aligned together.",
            "Is there a linear sequence of three smelting structures?",
            "You should check and see if there's a trio of furnaces arranged in a single line.",
            "Verify if a consecutive arrangement of three units dedicated for heating and smelting has been formed."
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=3, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 3, check_diagonal=False)"
    }
}

