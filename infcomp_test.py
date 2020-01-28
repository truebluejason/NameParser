import argparse
import pyprob
from new_infcomp import NameParser

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name to parse', nargs='?', default='jason yoo', type=str)
parser.add_argument('--num_traces', help='# traces to evaluate during inference', nargs='?', default=10, type=int)
parser.add_argument('--num_samples', help='# samples to sample from the posterior', nargs='?', default=10, type=int)
parser.add_argument('--model_path', help='Path to the saved model', nargs='?', default='/scratch/name_parser', type=str)
args = parser.parse_args()

OBSERVED = args.name
NUM_TRACES = args.num_traces
NUM_SAMPLES = args.num_samples
MODEL_PATH = args.model_path
print("========================================================")
print(f"Data to Parse: {OBSERVED}")
print(f"Number of Traces / Samples: {NUM_TRACES}, {NUM_SAMPLES}")
print(f"Model Path: {MODEL_PATH}")
print("========================================================")


model = NameParser()
model.load_inference_network(MODEL_PATH)
post = model.posterior_distribution(
    observe=model.get_observes(OBSERVED),
    inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
    num_traces=NUM_TRACES
)

for i in range(NUM_SAMPLES):
    print(post.sample())
