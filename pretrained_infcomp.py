import json
import numpy as np
import pandas as pd
import string
import torch
import torch.nn as nn

import pyprob
from pyprob import Model
import pyprob.distributions as dists


MAX_OUTPUT_LEN = 10
MAX_STRING_LEN = 35
SOS, EOS = "1", "2"
ALL_CHARACTERS = string.ascii_lowercase+" .,-'"+SOS+EOS
N_CHARACTERS = len(ALL_CHARACTERS)
CHAR_TO_INDEX = {}
for index, char in enumerate(ALL_CHARACTERS):
    CHAR_TO_INDEX[char] = index
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_name(first, middle, last, name_format) -> str:
    if name_format == 0: return f"{first} {last}"
    elif name_format == 1: return f"{last}, {first}"
    elif name_format == 2: return f"{first} {middle} {last}"
    elif name_format == 3: return f"{last}, {first} {middle}"
    elif name_format == 4: return f"{first} {middle}. {last}"
    else: return f"{last}, {first} {middle}."

def has_middle_name(name_format) -> bool:
    return name_format == 2 or name_format == 3

def has_initial(name_format) -> bool:
    return name_format == 4 or name_format == 5

def pad_string(original: str, desired_len: int, pad_character: str = ' ') -> str:
    # Returns the padded version of the original string to length: desired_len
    return original + (pad_character * (desired_len - len(original)))

def character_to_tensor(char) -> torch.tensor:
    result = torch.zeros(1,N_CHARACTERS).to(DEVICE)
    result[0,CHAR_TO_INDEX[char]] = 1
    return result

"""
class PretrainedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_seq_len=MAX_OUTPUT_LEN, num_layers=1, dropout=0.):
        super(PretrainedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.i2h = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        #input: <batch size x input size>
        #hidden: <num layers x batch size x hidden size>
        #gru_input: <1 x batch size x input size>
        #gru_output: <1 x batch size x hidden size>
        input = self.i2h(input).unsqueeze(0)
        gru_output, gru_hidden = self.gru(input, hidden)
        output = self.o2o(gru_output.squeeze(0))
        output = self.softmax(output)
        return output, gru_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
"""
class PretrainedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_seq_len=10, num_layers=4, dropout=0.):
        super(PretrainedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = 10
        self.num_layers = num_layers

        self.o2o = nn.Linear(hidden_size, output_size)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        input: <batch size x input size>
        hidden: <num layers x batch size x hidden size>
        gru_input: <1 x batch size x input size>
        gru_output: <1 x batch size x hidden size>
        """
        gru_output, gru_hidden = self.gru(input.unsqueeze(0), hidden)
        output = self.o2o(gru_output.squeeze(0))
        output = self.softmax(output)
        return output, gru_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)


class OneHot2DCategorical(dists.Categorical):
    def sample(self):
        s = self._torch_dist.sample()
        one_hot = self._probs * 0
        for i, val in enumerate(s):
            one_hot[i, int(val.item())] = 1
        return one_hot
    
    def log_prob(self, x, *args, **kwargs):
        # vector of one hot vectors
        non_one_hot = torch.tensor([row.nonzero() for row in x]).to(DEVICE)
        return super().log_prob(non_one_hot, *args, **kwargs)


class NameParser(Model):
    def __init__(self, peak_prob=0.9):
        # Initialize RNNs with pretrained weights
        self.firstname_rnn = PretrainedRNN(input_size=N_CHARACTERS, hidden_size=512, output_size=N_CHARACTERS, num_layers=4).to(DEVICE)
        self.middlename_rnn = PretrainedRNN(input_size=N_CHARACTERS, hidden_size=512, output_size=N_CHARACTERS, num_layers=4).to(DEVICE)
        self.lastname_rnn = PretrainedRNN(input_size=N_CHARACTERS, hidden_size=512, output_size=N_CHARACTERS, num_layers=4).to(DEVICE)
        self.firstname_rnn.load_state_dict(torch.load('nn_model/pretrained/first', map_location=DEVICE))
        self.middlename_rnn.load_state_dict(torch.load('nn_model/pretrained/middle', map_location=DEVICE))
        self.lastname_rnn.load_state_dict(torch.load('nn_model/pretrained/last', map_location=DEVICE))

        self.peak_prob = peak_prob
        super().__init__(name="Name String with Unknown Format")

    def forward(self):
        # Sample format
        format_index = int(pyprob.sample(dists.Categorical(torch.tensor([1/6]*6).to(DEVICE))).item())

        firstname, middlename, lastname = '', '', ''

        # Sample first name
        input = character_to_tensor(SOS)
        hidden = self.firstname_rnn.init_hidden(1)
        for _ in range(MAX_OUTPUT_LEN):
            output, hidden = self.firstname_rnn(input, hidden)
            character = ALL_CHARACTERS[int(pyprob.sample(dists.Categorical(output)).item())]
            input = character_to_tensor(character)
            if character == EOS: break
            firstname += character

        # Sample middle name if it should exist
        if has_middle_name(format_index):
            input = character_to_tensor(SOS)
            hidden = self.middlename_rnn.init_hidden(1)
            for _ in range(MAX_OUTPUT_LEN):
                output, hidden = self.middlename_rnn(input, hidden)
                character = ALL_CHARACTERS[int(pyprob.sample(dists.Categorical(output)).item())]
                input = character_to_tensor(character)
                if character == EOS: break
                middlename += character

        # Sample middle name initial if it should exist
        if has_initial(format_index):
            character_index = int(pyprob.sample(dists.Categorical(torch.tensor([1/26]*26).to(DEVICE))).item())
            middlename = ALL_CHARACTERS[character_index]

        # Sample last name
        input = character_to_tensor(SOS)
        hidden = self.lastname_rnn.init_hidden(1)
        for _ in range(MAX_OUTPUT_LEN):
            output, hidden = self.lastname_rnn(input, hidden)
            character = ALL_CHARACTERS[int(pyprob.sample(dists.Categorical(output)).item())]
            input = character_to_tensor(character)
            if character == EOS: break
            lastname += character
        
        # Add format level noise based on format index then character level white noise
        noised_name = format_name(firstname, middlename, lastname, format_index)
        probs = torch.ones(MAX_STRING_LEN, N_CHARACTERS).to(DEVICE)*((1-self.peak_prob)/(N_CHARACTERS-1))
        for i, character in enumerate(noised_name):
            probs[i, CHAR_TO_INDEX[character]] = self.peak_prob
        pyprob.observe(OneHot2DCategorical(probs), name=f"name_string")
    
        return noised_name, {'firstname': firstname,'middlename': middlename, 'lastname': lastname, 'format_index': format_index}
    
    def get_observes(self, name_string):
        if len(name_string) > MAX_STRING_LEN: raise Exception(f"Name string length cannot exceed {MAX_STRING_LEN}.")
        name_string = name_string.lower()
        one_hot = torch.zeros(MAX_STRING_LEN, N_CHARACTERS).to(DEVICE)
        name_string = pad_string(original=name_string, desired_len=MAX_STRING_LEN)
        for i, letter in enumerate(name_string):
            one_hot[i, CHAR_TO_INDEX[letter]] = 1.
        
        return {'name_string': one_hot}
