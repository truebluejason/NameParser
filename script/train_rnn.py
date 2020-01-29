import argparse
import pandas as pd
import string
import torch
import torch.nn as nn

SOS = "1"
EOS = "2"
ALL_CHARACTERS = string.ascii_lowercase+" .,-'"+SOS+EOS

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_seq_len=10, num_layers=1, dropout=0.):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = 10
        self.num_layers = num_layers

        self.i2h = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        input: <batch size x input size>
        hidden: <num layers x batch size x hidden size>
        gru_input: <1 x batch size x input size>
        gru_output: <1 x batch size x hidden size>
        """
        input = self.i2h(input).unsqueeze(0)
        gru_output, gru_hidden = self.gru(input, hidden)
        output = self.o2o(gru_output.squeeze(0))
        output = self.softmax(output)
        return output, gru_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='first/middle/last', nargs='?', default='first', type=str)
    parser.add_argument('--model_path', help='Path to save the saved model', nargs='?', default='nn_model/pretrained/first', type=str)
    parser.add_argument('--max_name_len', help='Maximum name length', nargs='?', default=10, type=int)
    parser.add_argument('--hidden_size', help='Hidden layer size for the model', nargs='?', default=128, type=int)
    parser.add_argument('--batch_size', help='Batch size for training', nargs='?', default=128, type=int)
    parser.add_argument('--num_steps', help='# epochs', nargs='?', default=10000, type=int)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    HIDDEN_SIZE = args.hidden_size
    MAX_NAME_LEN = args.max_name_len
    NAME = args.name
    num_steps = args.num_steps
    if NAME not in ['first','middle','last']: raise Exception("--name must be first, middle, or last")

    def clean_names(df):
        # Only include names with all valid characters and are shorter than max name length
        return df[df['name'].apply(lambda x: set(x.lower()).issubset(ALL_CHARACTERS) and len(x)<MAX_NAME_LEN)]
    def pad_string(original: str, desired_len: int, pad_character: str = ' ') -> str:
        # Returns the padded version of the original string to length: desired_len
        return original + (pad_character * (desired_len - len(original)))
    def strings_to_tensor(names: list, max_name_len: int = MAX_NAME_LEN):
        # Convert a list of names into a 3D tensor
        tensor = torch.zeros(max_name_len, len(names), len(ALL_CHARACTERS))
        for i_name, name in enumerate(names):
            for i_char, letter in enumerate(name):
                tensor[i_char][i_name][ALL_CHARACTERS.find(letter)] = 1
        return tensor
    def test(model):
        num_sample = 3
        names = [''] * num_sample
        input = strings_to_tensor([SOS]*num_sample, 1)[0]
        hidden = model.init_hidden(3)
        for _ in range(MAX_NAME_LEN):
            output, hidden = model(input, hidden)

            sampled_indexes = torch.distributions.Categorical(output.exp()).sample()
            sampled_characters = [ALL_CHARACTERS[index] for index in sampled_indexes]
            input = strings_to_tensor(sampled_characters, 1)[0]
            for i in range(len(names)):
                names[i] += sampled_characters[i]

        return names


    df = pd.read_csv(f"data/{NAME}.csv")
    df = clean_names(df)
    names = df['name'].tolist()

    model = Model(input_size=len(ALL_CHARACTERS), hidden_size=HIDDEN_SIZE, output_size=len(ALL_CHARACTERS), max_seq_len=MAX_NAME_LEN)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    losses = []
    dist = torch.distributions.Categorical(torch.tensor((df['count']/df['count'].sum()).tolist()))
    for i in range(num_steps):
        batch = []
        for _ in range(BATCH_SIZE):
            name = names[dist.sample().item()].lower()+EOS
            batch.append(pad_string(name, MAX_NAME_LEN))

        batch = strings_to_tensor(batch, MAX_NAME_LEN)
        optimizer.zero_grad()

        loss = 0.
        input = strings_to_tensor([SOS]*BATCH_SIZE, 1)[0]
        hidden = model.init_hidden(BATCH_SIZE)
        for j in range(MAX_NAME_LEN):
            output, hidden = model.forward(input, hidden)
            _, correct_indexes = batch[j].topk(1, dim=1)
            loss += criterion(output, correct_indexes.squeeze())

            true_characters = [ALL_CHARACTERS[index] for index in correct_indexes.squeeze()]
            input = strings_to_tensor(true_characters, 1)[0]

            #max_indexes = torch.argmax(output, dim=1)
            #top_characters = [ALL_CHARACTERS[index] for index in max_indexes]
            #input = strings_to_tensor(top_characters, 1)[0]

            #sampled_indexes = torch.distributions.Categorical(output.exp()).sample()
            #sampled_characters = [ALL_CHARACTERS[index] for index in sampled_indexes]
            #input = strings_to_tensor(sampled_characters, 1)[0]
            
        loss.backward()
        optimizer.step()
        losses.append(loss)

        if i%50 == 0:
            import matplotlib.pyplot as plt
            print(f"step {i} loss: {loss}")
            print(f"sample name: {test(model)}")
            plt.plot(losses)
            plt.savefig(f"result/{NAME}.png")
            plt.close()
        if i%1000 == 0:
            torch.save(model.state_dict(), args.model_path)
