import yaml
from collections import Counter


def check_generator(config):
    generator_config = config['Generator']
    generator_name = generator_config['Generator_Name']
    generator_counter = Counter(generator_name)
    assert len(generator_counter) == len(generator_name), \
        f'The Same Name is in the {generator_name} '


def check_discriminator(config):
    discriminator_config = config['Discriminator']
    discriminator_name = discriminator_config['Discriminator_Name']
    discriminator_counter = Counter(discriminator_name)
    assert len(discriminator_counter) == len(discriminator_name), \
        f'The Same Name is in the {discriminator_name} '


def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
        check_generator(config)
        check_discriminator(config)
        return config
