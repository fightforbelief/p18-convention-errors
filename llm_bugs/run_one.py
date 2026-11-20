"""Run one pipeline module on cached spec inputs: python run_one.py <pipeline.py> <inputs_dir> <out.parquet>"""
import importlib.util
import sys

import pandas as pd

spec = importlib.util.spec_from_file_location("pipe", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import glob, os
inputs = {os.path.basename(f)[:-8]: pd.read_parquet(f) for f in sorted(glob.glob(f"{sys.argv[2]}/*.parquet"))}
out = mod.build_panel(**inputs)  # keyword names = input file stems (calls/attrs, or calls/directory/history)
out = pd.DataFrame(out).reset_index(drop=True)
out.to_parquet(sys.argv[3])
