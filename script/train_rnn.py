import argparse
import pandas as pd
import string
import torch
import torch.nn as nn

SOS = "1"
EOS = "2"
ALL_CHARACTERS = string.ascii_lowercase+" .,-'"+SOS+EOS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_seq_len=10, num_layers=2, dropout=0.):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = 10
        self.num_layers = num_layers

        #self.i2h = nn.Linear(input_size, hidden_size)
        #self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
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
        #input = self.i2h(input).unsqueeze(0)
        #gru_output, gru_hidden = self.gru(input, hidden)
        #output = self.o2o(gru_output.squeeze(0))
        gru_output, gru_hidden = self.gru(input.unsqueeze(0), hidden)
        output = self.o2o(gru_output.squeeze(0))
        output = self.softmax(output)
        return output, gru_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='first/middle/last', nargs='?', default='first', type=str)
    parser.add_argument('--model_path', help='Path to save the saved model', nargs='?', default='nn_model/pretrained/first', type=str)
    parser.add_argument('--max_name_len', help='Maximum name length', nargs='?', default=10, type=int)
    parser.add_argument('--hidden_size', help='Hidden layer size for the model', nargs='?', default=256, type=int)
    parser.add_argument('--batch_size', help='Batch size for training', nargs='?', default=256, type=int)
    parser.add_argument('--num_epoch', help='# epochs', nargs='?', default=100, type=int)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    HIDDEN_SIZE = args.hidden_size
    MAX_NAME_LEN = args.max_name_len
    NAME = args.name
    NUM_EPOCH = args.num_epoch
    if NAME not in ['first','middle','last']: raise Exception("--name must be first, middle, or last")

    def clean_names(df):
        # Only include names with all valid characters and are shorter than max name length
        return df[df['name'].apply(lambda x: set(x.lower()).issubset(ALL_CHARACTERS) and len(x)<MAX_NAME_LEN)]
    def pad_string(original: str, desired_len: int, pad_character: str = ' ') -> str:
        # Returns the padded version of the original string to length: desired_len
        return original + (pad_character * (desired_len - len(original)))
    def strings_to_tensor(names: list, max_name_len: int = MAX_NAME_LEN):
        # Convert a list of names into a 3D tensor
        tensor = torch.zeros(max_name_len, len(names), len(ALL_CHARACTERS)).to(DEVICE)
        for i_name, name in enumerate(names):
            for i_char, letter in enumerate(name):
                tensor[i_char][i_name][ALL_CHARACTERS.find(letter)] = 1
        return tensor
    def test(model):
        num_sample = 3
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
        sample_names = list(map(lambda x: x[:x.find(EOS)+1] if x.find(EOS) >= 0 else x, sample_names))
        # characters chosen from argmax of GRU output
        mode_names = ['a','b','c'] 
        input = strings_to_tensor(['a','b','c'], 1)[0]
        hidden = model.init_hidden(num_sample)
        for _ in range(MAX_NAME_LEN):
            output, hidden = model(input, hidden)
            mode_indexes = torch.argmax(output, dim=1)
            mode_characters = [ALL_CHARACTERS[index] for index in mode_indexes]
            input = strings_to_tensor(mode_characters, 1)[0]
            for i in range(len(mode_names)):
                mode_names[i] += mode_characters[i]
        mode_names = list(map(lambda x: x[:x.find(EOS)+1] if x.find(EOS) >= 0 else x, mode_names))
        return sample_names, mode_names

    """
    df = pd.read_csv(f"data/{NAME}.csv")
    df = clean_names(df)
    names = df['name'].tolist()
    dist = torch.distributions.Categorical(torch.tensor((df['count']/df['count'].sum()).tolist()))
    """
    df = pd.read_csv("data/Train.csv")
    df = clean_names(df)
    print(f"Data Size: {len(df)}")

    model = Model(input_size=len(ALL_CHARACTERS), hidden_size=HIDDEN_SIZE, output_size=len(ALL_CHARACTERS), max_seq_len=MAX_NAME_LEN).to(DEVICE)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    losses = []

    for e in range(NUM_EPOCH):
        df = df.sample(frac=1).reset_index(drop=True)
        for i in range(int(len(df)/BATCH_SIZE)):
            batch = df['name'][i*BATCH_SIZE:(i+1)*BATCH_SIZE].tolist()
            batch = list(map(lambda x: pad_string(x.lower()+EOS, MAX_NAME_LEN), batch))
            batch = strings_to_tensor(batch, MAX_NAME_LEN)
            optimizer.zero_grad()
        
            loss = 0.
            input = strings_to_tensor([SOS]*BATCH_SIZE, 1)[0]
            hidden = model.init_hidden(BATCH_SIZE)
            for j in range(MAX_NAME_LEN):
                output, hidden = model.forward(input, hidden)
                _, correct_indexes = batch[j].topk(1, dim=1)

                loss += criterion(output, correct_indexes.squeeze(1))

                true_characters = [ALL_CHARACTERS[index] for index in correct_indexes.squeeze(1)]
                input = strings_to_tensor(true_characters, 1)[0]

                #max_indexes = torch.argmax(output, dim=1)
                #top_characters = [ALL_CHARACTERS[index] for index in max_indexes]
                #input = strings_to_tensor(top_characters, 1)[0]

                #sampled_indexes = torch.distributions.Categorical(output.exp()).sample()
                #sampled_characters = [ALL_CHARACTERS[index] for index in sampled_indexes]
                #input = strings_to_tensor(sampled_characters, 1)[0]
            
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                losses.append(loss)
            if i%100 == 0:
                import matplotlib.pyplot as plt
                print(f"epoch {e} step {i} loss: {loss}")
                samples, modes = test(model)
                print(f"sample names: {samples}")
                print(f"mode names: {modes}")
                plt.plot(losses)
                plt.savefig(f"result/{NAME}.png")
                plt.close()
            if i%1000 == 0:
                torch.save(model.state_dict(), args.model_path)
