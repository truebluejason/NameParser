import pandas as pd
import string
import torch

import pyprob
from pyprob import Model
import pyprob.distributions as dists

def format_name(first, middle, last, name_format) -> str:
    if name_format == 0: return f"{first} {last}"
    elif name_format == 1: return f"{last}, {first}"
    elif name_format == 2: return f"{first} {middle} {last}"
    elif name_format == 3: return f"{last}, {first} {middle}"
    elif name_format == 4: return f"{first} {middle}. {last}"
    else: return f"{last}, {first} {middle}."

def letter_to_index(letter: str) -> int:
    index = ALL_LETTERS.find(letter)
    if index == -1: raise Exception(f"letter {letter} is not permitted.")
    return index

def character_to_index(char: str) -> int:
    index = ALL_CHARACTERS.find(char)
    if index == -1: raise Exception(f"letter {char} is not permitted.")
    return index

def pad_string(original: str, desired_len: int, pad_character: str = ' ') -> str:
    # Returns the padded version of the original string to length: desired_len
    return original + (pad_character * (desired_len - len(original)))


MAX_STRING_LEN = 50
ALL_LETTERS = string.ascii_lowercase
ALL_CHARACTERS = string.ascii_lowercase+" .,-'"
N_LETTERS = len(ALL_LETTERS)
N_CHARACTERS = len(ALL_CHARACTERS)
FIRST_NAMES = pd.read_csv("data/first.csv")
MIDDLE_NAMES = pd.read_csv("data/middle.csv")
LAST_NAMES = pd.read_csv("data/last.csv")


class OneHot2DCategorical(dists.Categorical):
    def sample(self):
        s = self._torch_dist.sample()
        one_hot = self._probs * 0
        for i, val in enumerate(s):
            one_hot[i, int(val.item())] = 1
        return one_hot
    
    def log_prob(self, x, *args, **kwargs):
        # vector of one hot vectors
        non_one_hot = torch.tensor([row.nonzero() for row in x])
        return super().log_prob(non_one_hot, *args, **kwargs)


class NameParser(Model):
    def __init__(self, peak_prob=0.9):
        self.peak_prob = peak_prob
        super().__init__(name="Name String with Unknown Format")

    def forward(self):
        format_index = int(pyprob.sample(dists.Categorical(torch.tensor([1/6]*6))).item())

        firstname_probs = FIRST_NAMES['count'].tolist()/FIRST_NAMES['count'].sum()
        firstname_index = int(pyprob.sample(dists.Categorical(torch.tensor(firstname_probs))).item())
        firstname = FIRST_NAMES['name'][firstname_index].lower()

        lastname_probs = LAST_NAMES['count'].tolist()/LAST_NAMES['count'].sum()
        lastname_index = int(pyprob.sample(dists.Categorical(torch.tensor(lastname_probs))).item())
        lastname = LAST_NAMES['name'][lastname_index].lower()

        if format_index == 0 or format_index == 1:
            # The person has no middle name
            middlename = ""
        if format_index == 2 or format_index == 3:
            # The person has a middle name
            middlename_probs = MIDDLE_NAMES['count'].tolist()/MIDDLE_NAMES['count'].sum()
            middlename_index = int(pyprob.sample(dists.Categorical(torch.tensor(middlename_probs))).item())
            middlename = MIDDLE_NAMES['name'][middlename_index].lower()
        if format_index == 4 or format_index == 5:
            # The person has a middle name initial
            middlename_index = int(pyprob.sample(dists.Categorical(torch.tensor([1/26]*26))).item())
            middlename = ALL_LETTERS[middlename_index]

        # make a categorical distribution that observes each letter independently (like 50 independent categoricals)
        output = pad_string(original=format_name(firstname, middlename, lastname, format_index), desired_len=MAX_STRING_LEN)
        
        probs = torch.ones(MAX_STRING_LEN, N_CHARACTERS)*((1-self.peak_prob)/(N_CHARACTERS-1))
        for i, character in enumerate(output):
            probs[i, character_to_index(character)] = self.peak_prob
        pyprob.observe(OneHot2DCategorical(probs), name=f"name_string")

        return output, {'firstname': firstname,'middlename': middlename, 'lastname': lastname}
    
    def get_observes(self, name_string):
        if len(name_string) > 50: raise Exception("Name string length cannot exceed 50.")
        name_string = name_string.lower()
        one_hot = torch.zeros(MAX_STRING_LEN, N_CHARACTERS)
        name_string = pad_string(original=name_string, desired_len=MAX_STRING_LEN)
        for i, letter in enumerate(name_string):
            one_hot[i, character_to_index(letter)] = 1.
        
        return {'name_string': one_hot}
