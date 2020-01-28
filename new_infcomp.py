import json
import numpy as np
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

def check_middle_name(name_format) -> bool:
    return name_format == 2 or name_format == 4 or name_format == 5

def parse_name(canonical_name, name_format) -> tuple:
    has_middle_name = check_middle_name(name_format)
    first, middle, last = '', '', ''
    split = unpad_string(canonical_name).split(' ')
    if has_middle_name:
        first, middle, last = split[0], split[1], split[-1]
        if (name_format == 4 or name_format == 5) and len(middle)>1:
            middle = middle[0]
    else:
        first, last = split[0], split[-1]
    return first, middle, last

def reformat_name(canonical_name, name_format) -> str:
    first, middle, last = parse_name(canonical_name, name_format)
    return pad_string(original=format_name(first, middle, last, name_format), desired_len=MAX_STRING_LEN)

def pad_string(original: str, desired_len: int, pad_character: str = ' ') -> str:
    # Returns the padded version of the original string to length: desired_len
    return original + (pad_character * (desired_len - len(original)))

def unpad_string(original: str, pad_character: str = ' ') -> str:
    padding_index = MAX_CANONICAL_LEN
    for i in range(MAX_CANONICAL_LEN-1,0,-1):
        if original[i] != pad_character: break
        padding_index = i
    return original[:padding_index]

MAX_CANONICAL_LEN = 20
MAX_STRING_LEN = 22
ALL_LETTERS = string.ascii_lowercase
ALL_CHARACTERS = string.ascii_lowercase+" .,-'"
N_LETTERS = len(ALL_LETTERS)
N_CHARACTERS = len(ALL_CHARACTERS)
CHAR_TO_INDEX = {}
for index, char in enumerate(ALL_CHARACTERS):
    CHAR_TO_INDEX[char] = index


class PreProcessor():
    """
    Return probability distribution over i-th character (0<=i<50) given 0 to i-1th characters
    """
    def __init__(self):
        self.names_w_middle = pd.read_csv("data/names_w_middle.csv")['name'].tolist()[:100000]
        self.names_wo_middle = pd.read_csv("data/names_wo_middle.csv")['name'].tolist()[:100000]
        self.saved_w_middle = {}
        self.saved_wo_middle = {}
    
    def get_next_character_dist(self, canonical_name, middle_name = True) -> np.array:
        index = len(canonical_name)
        if middle_name:
            names_to_search = self.names_w_middle
            saved = self.saved_w_middle
        else:
            names_to_search = self.names_wo_middle
            saved = self.saved_wo_middle
        
        if canonical_name in saved: 
            return saved[canonical_name]
        relevant_names = list(filter(lambda name: name[:index] == canonical_name, names_to_search))
        prob = torch.zeros(len(ALL_CHARACTERS))
        for n in relevant_names: prob[CHAR_TO_INDEX[n[index]]] += 1

        prob = prob / len(relevant_names)
        if len(canonical_name) < 5 and len(saved) < 50000:
            saved[canonical_name] = prob
        return prob


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
        self.preprocessor = PreProcessor()
        super().__init__(name="Name String with Unknown Format")

    def forward(self):
        """
        Sample the latent name
        """
        format_index = int(pyprob.sample(dists.Categorical(torch.tensor([1/6]*6))).item())
        has_middle_name = check_middle_name(format_index)

        canonical_name = ""
        for i in range(MAX_CANONICAL_LEN):
            probs = self.preprocessor.get_next_character_dist(canonical_name, middle_name=has_middle_name)
            character_index = int(pyprob.sample(dists.Categorical(probs)).item())
            canonical_name += ALL_CHARACTERS[character_index]
        firstname, middlename, lastname = parse_name(canonical_name, format_index)
        
        """
        Add format level noise based on format index then character level white noise
        """
        noised_name = reformat_name(canonical_name, format_index)
        probs = torch.ones(MAX_STRING_LEN, N_CHARACTERS)*((1-self.peak_prob)/(N_CHARACTERS-1))
        for i, character in enumerate(noised_name):
            probs[i, CHAR_TO_INDEX[character]] = self.peak_prob
        pyprob.observe(OneHot2DCategorical(probs), name=f"name_string")

        return canonical_name, {'firstname': firstname,'middlename': middlename, 'lastname': lastname, 'format_index': format_index}
    
    def get_observes(self, name_string):
        if len(name_string) > MAX_STRING_LEN: raise Exception(f"Name string length cannot exceed {MAX_STRING_LEN}.")
        name_string = name_string.lower()
        one_hot = torch.zeros(MAX_STRING_LEN, N_CHARACTERS)
        name_string = pad_string(original=name_string, desired_len=MAX_STRING_LEN)
        for i, letter in enumerate(name_string):
            one_hot[i, CHAR_TO_INDEX[letter]] = 1.
        
        return {'name_string': one_hot}
