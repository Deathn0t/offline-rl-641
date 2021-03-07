import json
from glob import glob
import os

here = os.path.dirname(os.path.abspath(__file__))

all_dicts = []
for path in glob(os.path.join(here, 'csvs/*.json'), recursive=True):
    dict_i = json.load(open(path))
    all_dicts += dict_i

with open("data_polynome_all.json", "w") as f:
    json.dump(all_dicts, f)


if __name__ == "__main__":
    pass