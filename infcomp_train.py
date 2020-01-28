import argparse
import pyprob
from new_infcomp import NameParser

parser = argparse.ArgumentParser()
parser.add_argument('--peak_prob', help='How error intolerant the model will be from 0 to 1', nargs='?', default=0.99, type=float)
parser.add_argument('--num_traces', help='# traces to evaluate per training step', nargs='?', default=5000000, type=int)
parser.add_argument('--batch_size', help='Batch size for training', nargs='?', default=32, type=int)
parser.add_argument('--model_path', help='Path to save the saved model', nargs='?', default='/scratch/name_parser', type=str)
parser.add_argument('--cont', help='Continue training an existing model', nargs='?', default=False, type=bool)
args = parser.parse_args()

PEAK_PROB = args.peak_prob
NUM_TRACES = args.num_traces
BATCH_SIZE = args.batch_size
MODEL_PATH = args.model_path
CONTINUE = args.cont
print("========================================================")
print(f"Number of Traces: {NUM_TRACES}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Model Save Path: {MODEL_PATH}")
print(f"Continue Training: {CONTINUE}")
print("========================================================")

model = NameParser(peak_prob=PEAK_PROB)
if CONTINUE: model.load_inference_network(MODEL_PATH)
model.learn_inference_network(
    inference_network=pyprob.InferenceNetwork.LSTM,
    observe_embeddings={'name_string': {'dim' : 256}},
    num_traces=NUM_TRACES,
    batch_size=BATCH_SIZE,
    save_file_name_prefix=MODEL_PATH,
)

model.save_inference_network(MODEL_PATH)
