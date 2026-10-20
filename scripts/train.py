from __future__ import annotations

import argparse
import json

from hypoguard_twin.runtime import Runtime
from hypoguard_twin.modeling import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HypoGuard Twin using raw or deterministic synthetic CSV files.")
    parser.add_argument("--mode", choices=["auto", "synthetic", "real"], default="auto")
    args = parser.parse_args()
    runtime = Runtime(mode=args.mode)
    metadata = train_model(runtime.features.training_rows)
    runtime.storage.save_model_metadata(metadata)
    print(json.dumps({"data_mode": runtime.mode, "model_version": metadata["model_version"], "evaluation": metadata["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
