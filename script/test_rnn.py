import argparse
import torch
from train_rnn import ALL_CHARACTERS, SOS, EOS, Model

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help='Path to save the saved model', type=str)
parser.add_argument('--num_samples', help='# names to generate', nargs='?', default=10, type=int)
parser.add_argument('--max_name_len', help='Maximum name length', nargs='?', default=10, type=int)
parser.add_argument('--hidden_size', help='Hidden layer size for the model', nargs='?', default=256, type=int)
args = parser.parse_args()

MAX_NAME_LEN = args.max_name_len
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(input_size=len(ALL_CHARACTERS), hidden_size=args.hidden_size, output_size=len(ALL_CHARACTERS), max_seq_len=MAX_NAME_LEN)
model.load_state_dict(torch.load(args.model_path), map_location=DEVICE)
def strings_to_tensor(names: list, max_name_len: int = 10):
    # Convert a list of names into a 3D tensor
    tensor = torch.zeros(max_name_len, len(names), len(ALL_CHARACTERS)).to(DEVICE)
    for i_name, name in enumerate(names):
        for i_char, letter in enumerate(name):
            tensor[i_char][i_name][ALL_CHARACTERS.find(letter)] = 1
    return tensor

def test(model):
    num_sample = args.num_samples
    print(num_sample)

    # characters sampled from GRU output
    sample_names = [''] * num_sample
    input = strings_to_tensor([SOS]*num_sample, 1)[0]
    hidden = model.init_hidden(num_sample)
    for _ in range(MAX_NAME_LEN):
        output, hidden = model(input, hidden)
        sampled_indexes = torch.distributions.Categorical(output.exp()).sample()
        sampled_characters = [ALL_CHARACTERS[index] for index in sampled_indexes]
        input = strings_to_tensor(sampled_characters, 1)[0]
        for i in range(len(sample_names)):
            sample_names[i] += sampled_characters[i]
    sample_names = list(map(lambda x: x[:x.find(EOS)] if x.find(EOS) >= 0 else x, sample_names))

    # characters chosen from argmax of GRU output
    mode_names = ['a','b','c','d','e','f','g'] 
    input = strings_to_tensor(['a','b','c','d','e','f','g'] , 1)[0]
    hidden = model.init_hidden(len(mode_names))
    for _ in range(MAX_NAME_LEN):
        output, hidden = model(input, hidden)
        mode_indexes = torch.argmax(output, dim=1)
        mode_characters = [ALL_CHARACTERS[index] for index in mode_indexes]
        input = strings_to_tensor(mode_characters, 1)[0]
        for i in range(len(mode_names)):
            mode_names[i] += mode_characters[i]
    mode_names = list(map(lambda x: x[:x.find(EOS)] if x.find(EOS) >= 0 else x, mode_names))
    return sample_names, mode_names

samples, modes = test(model)
print(f"Sampled Names: {samples}")
print(f"Most Probable Names [A-G]: {modes}")