import argparse, yaml
from worldmodel.train.trainer import train

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
train(cfg)
