import sys
import os

settings = None


class Settings:
    _settings_instance = None

    @staticmethod
    def get_settings():
        if Settings._settings_instance is None:
            raise Exception('Create settings object with create_settings')
        return Settings._settings_instance

    @staticmethod
    def create_settings(model_settings_dir='model_settings/', model_settings_name="model_settings.txt", settings_dict = None, verbose=True):
        cur_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        Settings._settings_instance = Settings.read_from_file(os.path.join(cur_dir, model_settings_dir, model_settings_name), verbose=verbose)
        if settings_dict:
            Settings._settings_instance.settings_dict.update(settings_dict)

    def __init__(self, settings_dict=None):
        # Configurable settings for each model
        if settings_dict is None:
            settings_dict = {}
        self.settings_dict = settings_dict

    def has_value(self, name):
        return name in self.settings_dict

    def get_value(self, name):
        return self.settings_dict[name]

    def set_value(self, name, value):
        self.settings_dict[name] = value

    @classmethod
    def read_from_file(cls, file_name, verbose=True):
        settings_dict = {}
        with open(file_name, "r") as f:
            for line in f.readlines():
                if line != '':
                    name = line.split(":")[0]
                    if "," in line.split(": ")[1]:
                        value = line.split(": ")[1].split(",")
                    else:
                        value = eval(line.split(": ")[1])
                    settings_dict[name] = value
        if verbose:
            print("Finished reading settings")
        return cls(settings_dict)

    @classmethod
    def read_from_arguments(cls, verbose=True):
        settings_dict = {}

        f = sys.argv[1:]
        for i in range(0,len(f),2):
            name = f[i]
            val = f[i+1]
            if "," in val:
                val = val.split(",")
            else:
                val = eval(val)
            settings_dict[name] = val
        if verbose:
            print("Finished reading settings")
        return cls(settings_dict)