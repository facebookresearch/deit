import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
with open('deit_small_nxm.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    print(data)

