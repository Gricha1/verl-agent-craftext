"""Facts of a natural-language state description, to score text written by the world model.

Both sides are prose: the world model's imagined text and Gemma's description of the real frame, so accuracy means
agreement with Gemma, not with the environment. facts() reads what the agent faces, whether it sleeps, its inventory,
the objects in view and their coarse directions (distances are ignored). FactScore counts agreement per fact type and,
because neighbouring states read almost the same, also on the facts that changed since x_t: copying the description
of x_t scores 0 there.
"""
from __future__ import annotations

import re

NUMBER = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
SCENE = {"tree": "tree", "trees": "tree", "stone": "stone", "water": "water", "sand": "sand", "sapling": "sapling",
         "saplings": "sapling", "cow": "cow", "cows": "cow", "zombie": "zombie", "zombies": "zombie",
         "skeleton": "skeleton", "skeletons": "skeleton", "table": "table", "furnace": "furnace", "coal": "coal",
         "iron": "iron", "diamond": "diamond", "lava": "lava", "arrow": "arrow", "path": "path", "grass": "grass"}
ITEM = {"wood": "wood", "stone": "stone", "coal": "coal", "iron": "iron", "diamond": "diamond", "sapling": "sapling",
        "saplings": "sapling"}
MATERIAL, TOOL = {"wood", "stone", "iron"}, {"pickaxe", "pickaxes", "sword", "swords"}
DIRECTION = {"up": "up", "above": "up", "top": "up", "upper": "up", "north": "up", "down": "down", "below": "down",
             "bottom": "down", "lower": "down", "south": "down", "left": "left", "west": "left", "right": "right",
             "east": "right"}


def facts(text: str) -> dict:
    """faces: object or None; asleep: bool; inventory: {item: count} or None if unstated; objects, directions: sets."""
    faces, asleep, seen_inventory = None, False, False
    inventory, objects, directions = {}, set(), set()
    for sentence in re.split(r"[.;]", text.lower()):
        words = re.findall(r"[a-z]+|\d+", sentence)
        pending, after_direction, want_face, faced_only = [], False, False, False
        skip = 0
        for i, word in enumerate(words):
            if skip:
                skip -= 1
                continue
            following = words[i + 1] if i + 1 < len(words) else ""
            if word in ("inventory", "items"):
                seen_inventory = True
            elif word == "asleep":
                asleep = True
            elif word in ("faces", "facing"):
                want_face = True
            elif word in MATERIAL and following in TOOL:
                inventory[f"{word}_{following.rstrip('s')}"] = 1
                skip = 1
            elif (word.isdigit() or word in NUMBER) and following in ITEM:
                inventory[ITEM[following]] = int(word) if word.isdigit() else NUMBER[word]
                skip = 1
            elif word in ITEM and following.isdigit():
                inventory[ITEM[word]] = int(following)
                skip = 1
            elif word in SCENE:
                name = SCENE[word]
                if faced_only:  # "faces stone with trees to the left": the direction is not the faced object's
                    pending, faced_only = [], False
                if name != "grass":
                    objects.add(name)
                    if after_direction:
                        pending, after_direction = [], False
                    pending.append(name)
                    faced_only = want_face
                if want_face and faces is None:
                    faces = name
                want_face = False
            elif word in DIRECTION:
                directions.update((name, DIRECTION[word]) for name in pending)
                after_direction, faced_only = True, False
    return dict(faces=faces, asleep=asleep, inventory=inventory if inventory or seen_inventory else None,
                objects=objects, directions=directions)


class FactScore:
    """Agreement of one kind of guess (model text, copy of x_t, ...) with the description of the real next state."""

    def __init__(self):
        self.count = dict(n=0, faces=0, asleep=0, inventory=0, inventory_n=0, faces_changed=0, faces_changed_n=0,
                          inventory_changed=0, inventory_changed_n=0)
        self.sets = {k: dict(tp=0, fp=0, fn=0) for k in ("objects", "directions")}
        self.new = {k: dict(right=0, n=0) for k in ("objects", "directions")}

    def add(self, guess: str, real: str, first: str):
        g, r, f, c = facts(guess), facts(real), facts(first), self.count
        c["n"] += 1
        c["faces"] += g["faces"] == r["faces"]
        c["asleep"] += g["asleep"] == r["asleep"]
        if r["inventory"] is not None:
            c["inventory_n"] += 1
            c["inventory"] += g["inventory"] == r["inventory"]
        if r["faces"] != f["faces"]:
            c["faces_changed_n"] += 1
            c["faces_changed"] += g["faces"] == r["faces"]
        if r["inventory"] is not None and f["inventory"] is not None and r["inventory"] != f["inventory"]:
            c["inventory_changed_n"] += 1
            c["inventory_changed"] += g["inventory"] == r["inventory"]
        for key, total in self.sets.items():
            total["tp"] += len(g[key] & r[key])
            total["fp"] += len(g[key] - r[key])
            total["fn"] += len(r[key] - g[key])
            self.new[key]["n"] += len(r[key] - f[key])
            self.new[key]["right"] += len((r[key] - f[key]) & g[key])

    def result(self) -> dict:
        c = self.count
        share = lambda right, n: round(right / n, 4) if n else None
        out = dict(samples=c["n"], faces=share(c["faces"], c["n"]), asleep=share(c["asleep"], c["n"]),
                   inventory=share(c["inventory"], c["inventory_n"]),
                   faces_on_changed=share(c["faces_changed"], c["faces_changed_n"]), faces_changed_n=c["faces_changed_n"],
                   inventory_on_changed=share(c["inventory_changed"], c["inventory_changed_n"]),
                   inventory_changed_n=c["inventory_changed_n"])
        for key, t in self.sets.items():
            precision, recall = share(t["tp"], t["tp"] + t["fp"]), share(t["tp"], t["tp"] + t["fn"])
            out[f"{key}_precision"], out[f"{key}_recall"] = precision, recall
            out[f"{key}_f1"] = share(2 * t["tp"], 2 * t["tp"] + t["fp"] + t["fn"])
            out[f"{key}_new_recall"], out[f"{key}_new_n"] = share(self.new[key]["right"], self.new[key]["n"]), self.new[key]["n"]
        return out
