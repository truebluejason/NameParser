import argparse
import math
import pandas as pd
import string
import torch
import torch.nn as nn
import torch.nn.functional as F


NAME = "transformer"
MODEL_PATH = "nn_model/transformer"
SOS = "1"
EOS = "2"
PAD = "3"
MAX_NAME_LEN = 35
ALL_CHARACTERS = string.ascii_lowercase+" .,-'"+SOS+EOS+PAD
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCH = 3
NUM_LAYERS = 2
NUM_HEADS = 2
DROPOUT = 0.2
HIDDEN_SIZE = 256

class TransformerModel(nn.Module):

    def __init__(self, n_chars, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(n_chars, dropout)
        encoder_layers = TransformerEncoderLayer(n_chars, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(n_chars, n_chars)

        self.init_weights()

    def _generate_square_subsequent_mask(self, src):
        #mask = torch.zeros(src.shape)
        #mask[src==ALL_CHARACTERS.find(PAD)] = float('-inf')
        sz = len(src)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):        
        src_mask = self._generate_square_subsequent_mask(src).to(DEVICE)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



def pad_string(original: str, desired_len: int, pad_character: str = PAD) -> str:
    # Returns the padded version of the original string to length: desired_len
    return original + (pad_character * (desired_len - len(original)))

def strings_to_tensor(names: list, max_name_len: int = MAX_NAME_LEN):
    # Convert a list of names into a 3D tensor
    tensor = torch.zeros(max_name_len, len(names), len(ALL_CHARACTERS)).to(DEVICE)
    for i_name, name in enumerate(names):
        for i_char, letter in enumerate(name):
            tensor[i_char][i_name][ALL_CHARACTERS.find(letter)] = 1
    return tensor


df = pd.read_csv("data/Train.csv")
df = df[df['name'].apply(lambda x: set(x.lower()).issubset(ALL_CHARACTERS) and len(x)<MAX_NAME_LEN)]
print(f"Data Size: {len(df)}")

model = TransformerModel(len(ALL_CHARACTERS), NUM_HEADS, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def evaluate(eval_model):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        batch = [SOS] * 1
        data = strings_to_tensor(batch, MAX_NAME_LEN)
        output = eval_model(data)
        print(output.shape)
        print(output.exp().tolist())
    exit(0)
evaluate(model)


def train(df):
    losses = []
    for e in range(NUM_EPOCH):
        df = df.sample(frac=1).reset_index(drop=True)
        model.train() # Turn on the train mode
        for i in range(int(len(df)/BATCH_SIZE)):
            batch = df['name'][i*BATCH_SIZE:(i+1)*BATCH_SIZE].tolist()
            batch = list(map(lambda x: pad_string(SOS+x.lower()+EOS, MAX_NAME_LEN), batch))
            optimizer.zero_grad()
            loss = 0.
                
            batch_tensor = strings_to_tensor(batch, MAX_NAME_LEN)
            data = batch_tensor[:batch_tensor.shape[0]-1]
            target = batch_tensor[1:]
            new_target = torch.zeros(target.shape[0], target.shape[1], dtype=torch.long)
            for j in range(target.shape[0]):
                for k in range(target.shape[1]):
                    new_target[j,k] = torch.argmax(target[j,k])
            target = new_target

            output = model(data)
            loss += criterion(output.view(-1, len(ALL_CHARACTERS)), target.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if i%10 == 0:
                print(f"epoch {e} step {i} loss: {loss}")
                losses.append(loss)
            if i%100 == 0:
                import matplotlib.pyplot as plt
                plt.plot(losses)
                plt.savefig(f"result/{NAME}.png")
                plt.close()
            if i%1000 == 0:
                torch.save(model.state_dict(), MODEL_PATH)

"""
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
"""

train(df)
