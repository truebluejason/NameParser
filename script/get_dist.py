import pandas as pd
import string
import sys
from torch import tensor
from torch.distributions import Categorical

"""
Run from project directory as:

python script/get_dist.py <number of full names to generate> <max full name length> <csv file path>
"""
NUM_NAME = int(sys.argv[1])
LEN_NAME = int(sys.argv[2])
CSV_PATH = sys.argv[3]

INCLUDE_MIDDLE_NAME = True

FIRST_NAMES = pd.read_csv("data/first.csv")
MIDDLE_NAMES = pd.read_csv("data/middle.csv")
LAST_NAMES = pd.read_csv("data/last.csv")

firstname_dist = Categorical(tensor(FIRST_NAMES['count'].tolist()/FIRST_NAMES['count'].sum()))
middlename_dist = Categorical(tensor(MIDDLE_NAMES['count'].tolist()/MIDDLE_NAMES['count'].sum()))
lastname_dist = Categorical(tensor(LAST_NAMES['count'].tolist()/LAST_NAMES['count'].sum()))

fullnames = []
def pad_string(original: str, desired_len: int, pad_character: str = ' ') -> str:
    # Returns the padded version of the original string to length: desired_len
    return original + (pad_character * (desired_len - len(original)))
i = 0
while len(fullnames) < NUM_NAME:
    if i%1000 == 0: print(f"Created {i}/{NUM_NAME} names...")
    i += 1
    fname = FIRST_NAMES['name'][firstname_dist.sample().item()]
    mname = MIDDLE_NAMES['name'][middlename_dist.sample().item()]
    lname = LAST_NAMES['name'][lastname_dist.sample().item()]
    if INCLUDE_MIDDLE_NAME: 
        fullname = f"{fname} {mname} {lname}"
    else: 
        fullname = f"{fname} {lname}"
    if len(fullname) > LEN_NAME:
        i -= 1
        continue
    fullname = pad_string(fullname, LEN_NAME).lower()
    fullnames.append(fullname)

df = pd.DataFrame(data={"name": fullnames})
df.to_csv(CSV_PATH, sep=',',index=False)

"""
print(f"Sample Name: {fullnames[0]}")
ALL_CHARACTERS = string.ascii_lowercase+" .,-'"

probs = []
for i in range(LEN_NAME):
    print(f"Computing character probability for index {i}...")
    prob = [0] * len(ALL_CHARACTERS)
    for name in fullnames:
        ith_char = name[i]
        ith_char_index = ALL_CHARACTERS.find(ith_char)
        if ith_char_index == -1: raise Exception(f"Invalid character {ith_char} in name {name}.")
        prob[ith_char_index] += 1
    for i in range(len(ALL_CHARACTERS)):
        prob[i] = prob[i] / len(fullnames)
    probs.append(prob)


df = pd.DataFrame(data={"probs": probs})
df.to_csv(CSV_PATH, sep=',',index=False)
"""