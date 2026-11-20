"""Build the four quarterly panel inputs from the NYU 1976--2020 Call Report file."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SHA256 = "40692c2ec9af404b3148ffbd7ad970a4ce191d9ac8f781b415e8a10909533ad0"
BANKS = ("bank_leverage", "bank_cash_liquidity", "bhc_leverage", "bhc_cash_liquidity")


def construct(raw, panel):
    key = "bhcid" if panel.startswith("bhc") else "rssdid"
    columns = ["assets", "equity", "cash"]
    values = raw[[key, "date", *columns]].dropna(subset=[key, "date"])
    if key == "bhcid":
        values = values.groupby([key, "date"])[columns].sum().reset_index()
        values = values[values[key] != "0"]
    if panel.endswith("leverage"):
        values = values[(values.equity > 1e-10) & (values.assets > 0)].copy()
        values["y"] = values.assets / values.equity
    else:
        values = values[(values.assets > 1e-10) & (values.cash >= 0)].copy()
        values["y"] = values.cash / values.assets
        values = values[np.isfinite(values.y)]
        values = values[values.groupby(key).y.transform("std") != 0]
    data = values.rename(columns={key: "unique_id", "date": "ds"})[["unique_id", "ds", "y"]]
    data = data.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    data["y"] = data.y.astype(np.float32).astype(np.float64)
    data = data[np.isfinite(data.y)].reset_index(drop=True)
    if data.duplicated(["unique_id", "ds"]).any():
        raise ValueError(f"{panel}: duplicate entity-quarter keys")
    return data


def canonical(frame):
    frame = frame[["unique_id", "ds", "y"]].copy()
    frame["unique_id"] = frame.unique_id.astype(str)
    frame["ds"] = pd.to_datetime(frame.ds).astype("datetime64[ns]")
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stata_file", type=Path)
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--verify-existing", action="store_true", help="compare every key and value with an existing snapshot")
    args = parser.parse_args()
    with args.stata_file.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SOURCE_SHA256:
            raise ValueError("Source differs from the 1976--2020 release used for the paper")
    raw = pd.read_stata(args.stata_file, columns=["rssdid", "bhcid", "date", "assets", "equity", "cash"],
                        convert_categoricals=False, convert_dates=False)
    for key in ("rssdid", "bhcid", "date"):
        raw[key] = raw[key].fillna(99999).astype("int64").astype(str).replace("99999", pd.NA)
    raw["date"] = pd.to_datetime(raw.date, format="%Y%m%d")
    args.directory.mkdir(exist_ok=True, parents=True)
    for panel in BANKS:
        frame = construct(raw, panel)
        path = args.directory / f"{panel}_panel.parquet"
        if args.verify_existing:
            pd.testing.assert_frame_equal(canonical(frame), canonical(pd.read_parquet(path)), check_exact=True, check_dtype=False)
        else:
            if path.exists():
                raise FileExistsError(path)
            frame.to_parquet(path, index=False)
        print(panel, len(frame), "rows", flush=True)


if __name__ == "__main__":
    main()
