
from craftext.environment.scenarious.checkers.target_state import Achievements, TargetState
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState

def create_target_state(required=[], forbidden=[]):
    base_vector = [AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    return TargetState(achievements=target_achievements)


easy = {
"EXPLORE":{
    "instruction": "Explore the environment",
    "instruction_paraphrases":  [
          "Take a walk around and see what you can find",
          "Look around the area and investigate your surroundings",
          "Wander through the environment and discover what's there",
          "Move freely through the world and explore its corners",
          "Travel around the map and check out different places",
          "Roam the landscape and look for interesting things",
          "Survey the environment and observe what’s nearby",
          "Navigate through the terrain and see what it holds",
          "Go out and examine the world around you",
          "Search the area for anything unusual or interesting",
          "Step outside and explore the nearby environment",
          "Stroll through the world and take in your surroundings",
          "Leave your starting point and venture into the unknown",
          "Move through the world and see what’s waiting",
          "Uncover hidden parts of the map by walking around",
          "Explore different directions to learn about the area",
          "Walk through the terrain and learn about the world",
          "Head out and explore wherever your path leads",
          "Investigate your environment and find something new",
          "Discover your surroundings by moving through them"
        ],
    "scenario_checker": Scenarios.EXPLORE,
    "arguments": create_target_state(
      required=[Achievement.MAKE_IRON_SWORD],
      forbidden=[]
    )
  }
}

medium = {}