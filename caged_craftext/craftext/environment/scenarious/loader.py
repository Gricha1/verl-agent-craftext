import os
import importlib
import flax.struct
import yaml
import craftext

import pathlib
import inspect
import flax

import craftext.dataset


CONFIG_DIR_NAME = "configs"

@flax.struct.dataclass
class ScenariosConfig:
    """Scenarios configuration structure
    
    Keyword arguments:
        - dataset_key      -- task type
        - subset_key       -- task complexity and if `test` - paraphrases / items
        - base_environment -- use `Classic` or not
        - use_parafrases   -- use `paraPhrases` in loading or not
        - test             -- is it `test` data or not
    """ 
    dataset_key: str
    subset_key: str
    base_environment: str
    use_parafrases: str
    test: str
    use_constraints_parafrases: str
    

class ScenariosConfigLoader:
    @staticmethod
    def get_config_path(config_name: str) -> pathlib.PurePath:
        """
        search file if exist in root of module in `CONFIG_DIR_NAME`
        
        Keyword arguments:
            config_name : `str`
            
        Return arguments:

        """
        module = inspect.getmodule(craftext.dataset)
        
        if not module:
            raise ModuleNotFoundError
        
        print(module.__path__)
    
        module_path = pathlib.PurePath(module.__path__[0])
        
        config_path = module_path.joinpath(f'{CONFIG_DIR_NAME}/{config_name}.yaml')

        return config_path

    @staticmethod
    def load_config(config_name: str) -> ScenariosConfig:
    
        config_path = ScenariosConfigLoader.get_config_path(config_name)
        
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        return ScenariosConfig(
            dataset_key      =config_data.get("dataset_key"),
            subset_key       =config_data.get("subset_key"),
            base_environment =config_data.get("base_environment"),
            use_parafrases   =config_data.get("use_parafrases", False),
            test             =config_data.get("test", False),
            use_constraints_parafrases  = config_data.get("use_constraints_parafrases", False)
        )


def get_default_scenario_path():
    """Gets the default absolute path to the scenarios directory based on module installation."""
   
    module_path = inspect.getmodule(craftext.dataset).__path__[0]
    print(module_path)
    return os.path.join(module_path, 'scenarious')

def load_scenarios(scenarious_config):
    scenarios = {}
    scenarios_dir = get_default_scenario_path()
   
    module = "test" if scenarious_config.test else "instructions"
    mode = scenarious_config.dataset_key
    data_key = scenarious_config.subset_key
   
    if scenarios_dir is None:
        raise ValueError("Scenario path could not be determined.")

    for file in os.listdir(scenarios_dir):
        if mode in file:
            scenario_module_name = f"craftext.dataset.scenarious.{file}.{module}"
            scenario_module = importlib.import_module(scenario_module_name)
            
            if hasattr(scenario_module, data_key):
                scenarios.update(getattr(scenario_module, data_key))
    
    # print(scenarios)
   
    return scenarios