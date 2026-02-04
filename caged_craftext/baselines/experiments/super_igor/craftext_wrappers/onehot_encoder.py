import numpy as np

synonyms = {
        "DEFEAT": ["conquer", "overcome", "vanquish", "crush", "subdue", "beat", "triumph", "outsmart", "overpower", "annihilate", "smash", "outplay", "dominate", "obliterate", "outclass"],
        "EAT": ["consume", "devour", "ingest", "bite", "nibble", "swallow", "chew", "feast", "gulp", "snack", "munch", "taste", "partake", "digest", "stuff"],
        "FIND": ["discover", "locate", "uncover", "detect", "spot", "identify", "track down", "retrieve", "pinpoint", "reveal", "expose", "recognize", "hunt down", "notice", "search"],
        "COLLECT": ["gather", "accumulate", "assemble", "harvest", "acquire", "procure", "obtain", "amass", "mine", "store", "stockpile", "hoard", "aggregate", "consolidate", "retrieve", "reap"],
        "MAKE": ["create", "craft", "build", "produce", "forge", "assemble", "construct", "compose", "generate", "manufacture", "fabricate", "form", "develop", "invent", "design"],
        "PLACE": ["position", "set", "put", "arrange", "install", "deposit", "situate", "establish", "lay", "plant", "fit", "station", "organize", "embed", "align"],
        "WAKE": ["awaken", "rise", "stir", "rouse", "get up", "revive", "arouse", "emerge", "alert", "recover", "snap out", "come to", "shake up", "waken", "reanimate"],
        "ZOMBIE": ["undead", "walker", "ghoul", "corpse", "monster", "revenant", "ghost", "spirit", "wight", "creature", "fiend", "husk", "shade", "specter", "phantom"],
        "SKELETON": ["bones", "frame", "remains", "carcass", "skull", "structure", "skeleton warrior", "ribs", "skeleton knight", "spine", "vertebrae", "skeleton soldier", "mummy", "skeletal figure", "skeletal being"],
        "COW": ["bovine", "heifer", "calf", "bull", "ox", "steer", "cowherd", "livestock", "dairy cow", "bullock", "milker", "cattle", "beef cow", "bull calf", "jersey"],
        "WOOD": ["tree", "timber", "plank", "lumber", "log", "board", "branch", "trunk", "stick", "kindling", "pole", "twig", "firewood", "panel", "beam", "plank"],
        "STONE": ["rock", "pebble", "boulder", "slab", "cobblestone", "granite", "marble", "limestone", "sandstone", "quartz", "flint", "shard", "paving stone", "gemstone", "crystal"],
        "COAL": ["charcoal", "coke", "anthracite", "bituminous coal", "lignite", "carbon", "fuel", "fossil fuel", "cinder", "briquette", "peat", "smoke coal", "soft coal", "black rock", "ember"],
        "IRON": ["metal", "steel", "iron ore", "ingot", "cast iron", "wrought iron", "ferrous metal", "ore", "pig iron", "sheet iron", "bar iron", "iron bar", "nugget", "iron chunk", "iron plate"],
        "DIAMOND": ["gem", "jewel", "crystal", "precious stone", "sparkler", "carat", "brilliant", "facet", "cut stone", "hard gem", "rough diamond", "polished diamond", "flawless stone", "gemstone"],
        "SAPLING": ["seedling", "young tree", "shoot", "sprout", "plantlet", "twig", "shrub", "offshoot", "sucker", "sapwood", "baby tree", "treelet", "little tree", "plantling", "starter plant"],
        "DRINK": ["beverage", "liquid", "refreshment", "cocktail", "juice", "soda", "water", "smoothie", "shake", "tonic", "brew", "wine", "beer", "milk", "tea"],
        "PLANT": ["vegetation", "flora", "greenery", "shrub", "bush", "tree", "sapling", "vine", "crop", "herb", "seedling", "weed", "foliage", "succulent", "flower"],
        "PICKAXE": ["axe", "tool", "hammer", "mining tool", "pick", "digging tool", "chisel", "hoe", "mattock", "crowbar", "shovel", "claw tool", "sledge", "miner's tool", "spade"],
        "SWORD": ["blade", "weapon", "rapier", "cutlass", "sabre", "broadsword", "katana", "dagger", "claymore", "longsword", "shortsword", "scimitar", "saber", "gladius", "foil"],
        "TABLE": ["desk", "counter", "surface", "workbench", "platform", "stand", "coffee table", "dining table", "writing desk", "console", "board", "bench", "countertop", "high table", "low table"],
        "FURNACE": ["oven", "kiln", "forge", "heater", "stove", "smelter", "foundry", "incinerator", "furnace room", "blast furnace", "fireplace", "melter", "burner", "grill", "brazier"],
        "WOOD": ["wooden", "timber", "lumber", "oak", "pine", "mahogany", "teak", "maple", "beech", "walnut", "cedar", "birch", "spruce", "cherry", "ash"],
        "STONE": ["rocky", "stony", "granite", "marble", "slate", "limestone", "hard", "solid", "igneous", "sedimentary", "metamorphic", "smooth", "rough", "textured", "gritty"],
        "IRON": ["metallic", "steel", "ferrous", "rusty", "hard", "strong", "durable", "solid", "heavy", "cold", "wrought", "cast", "unbending", "sturdy", "tempered"],
        "?":[]
}


def transform_step(step):
    words = step.split()
    deteministic_step = []
   # print(words)
    for word in words:
        word = word.lower().replace(".", "").replace(",", "")
        for key in synonyms.keys():
            if word in synonyms[key] or word == key.lower():
                deteministic_step.append(key)
    if len(deteministic_step)==0:
        deteministic_step.append("?")
    return "_".join(deteministic_step)

def transform_plan(steps):
    steps = steps.split("\n")
    determenistic_plan = []
    for step in steps:
        determenistic_step = transform_step(step)
        determenistic_plan.append(determenistic_step)
    return "\n".join(determenistic_plan)

def onehot_step(step):
    keys_list = list(synonyms.keys())
    words = step.split("_")
    onehot_vector = [0 for i in range(len(keys_list))]
    for word in words:
        if word in keys_list:
            idx = keys_list.index(word)
            onehot_vector[idx] = 1
    return onehot_vector

def onehot_plans(steps, max_steps_count=16):
    steps = steps.split("\n")
    vectors = []
    for step in steps:
        vector = onehot_step(step)
        vectors.append(vector)
    if len(vectors)<max_steps_count:
        none_step = onehot_step("?")
        count_to_add = [none_step] * (max_steps_count- len(vectors))
        vectors+= count_to_add
    if len(vectors)>max_steps_count:
        vectors = vectors[:max_steps_count]
    return vectors
        
        
def join_plans(vectors):
    joined_vectors = []
    for vector in vectors:
        joined_vectors += vector
    return joined_vectors


def encode_plans(plans):
    derermenistic_plans = []
    for i in range(len(plans)):
        upd_plan = join_plans(onehot_plans(transform_plan(plans[i])))
        derermenistic_plans.append(upd_plan)
    return np.array(derermenistic_plans)
    
