'''Configuration handling.

Dictionary-based configuration with a TOML-based on-disk represenation.

See https://github.com/toml-lang/toml
'''

import toml


def load_config(path):
    '''Loads a dictionary from configuration file.

    Args:
      path: the path to load the configuration from.

    Returns:
      The configuration dictionary loaded from the file.
    '''

    return toml.load(path)


def save_config(attrs, path):
    '''Saves a configuration dictionary to a file.
    Args:
      path: the path to save the configuration dictionary to.
    '''

    toml.dump(attrs, path)
