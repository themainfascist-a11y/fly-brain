#!/usr/bin/env python3
"""Build the PyTorch FlyWire sparse weight caches.

This reproduces the weight construction used by code/run_pytorch.py:
  row = Postsynaptic_Index
  col = Presynaptic_Index
  value = Excitatory x Connectivity

Outputs:
  data/weight_coo.pkl
  data/weight_csr.pkl

The pickles contain torch sparse tensors and are intentionally generated rather
than committed as ordinary Git blobs because each is hundreds of MB.
"""

from pathlib import Path
import pickle

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONN = DATA / "2025_Connectivity_783.parquet"
COMP = DATA / "2025_Completeness_783.csv"
COO = DATA / "weight_coo.pkl"
CSR = DATA / "weight_csr.pkl"


def main() -> None:
    data_conn = pd.read_parquet(CONN)
    data_name = pd.read_csv(COMP)
    num_neurons = data_name.shape[0]

    # Match the original repository's construction exactly.
    idx = [
        data_conn["Postsynaptic_Index"].to_list(),
        data_conn["Presynaptic_Index"].to_list(),
    ]
    val = data_conn["Excitatory x Connectivity"].to_list()

    weight_coo = torch.sparse_coo_tensor(
        idx, val, (num_neurons, num_neurons)
    ).to(torch.float32)

    # Coalesce so the serialized COO has canonical indices when duplicate
    # synaptic entries are present.
    weight_coo = weight_coo.coalesce()

    with COO.open("wb") as f:
        pickle.dump(weight_coo, f, protocol=pickle.HIGHEST_PROTOCOL)

    weight_csr = weight_coo.to_sparse_csr()
    with CSR.open("wb") as f:
        pickle.dump(weight_csr, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"neurons: {num_neurons:,}")
    print(f"nonzero entries: {weight_coo._nnz():,}")
    print(f"COO: {COO} ({COO.stat().st_size / 1024**2:.1f} MiB)")
    print(f"CSR: {CSR} ({CSR.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
