import argparse
from train import ConditionalGeneration
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str)
parser.add_argument("--model_binary", default=None, type=str)
parser.add_argument("--output_dir", default="kobart_summary", type=str)
args = parser.parse_args()

with open(args.hparams) as f:
    hparams = yaml.load(f)

inf = ConditionalGeneration.load_from_checkpoint(args.model_binary, hparams=hparams)

inf.model.save_pretrained(args.output_dir)
