import argparse
import json
import math
import os
from pathlib import Path


STAT_FIELDS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def _validate_vector(name: str, value: object, expected_dim: int) -> None:
    if not isinstance(value, list) or len(value) != expected_dim:
        raise ValueError(f"{name} must contain {expected_dim} values, got {value!r}")
    if not all(
        isinstance(item, (int, float)) and math.isfinite(item) for item in value
    ):
        raise ValueError(f"{name} contains a non-finite or non-numeric value")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot meta/stats.json into a LingBot-compatible norm_stats file."
    )
    parser.add_argument("--stats-json", required=True, help="Path to LeRobot meta/stats.json")
    parser.add_argument("--output", required=True, help="Path to output LingBot norm stats json")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["observation.state", "action"],
        help="Keys to keep from LeRobot stats.json",
    )
    parser.add_argument("--expected-dim", type=int, default=None)
    args = parser.parse_args()

    stats_path = Path(args.stats_json)
    output_path = Path(args.output)

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    missing = [key for key in args.keys if key not in stats]
    if missing:
        raise KeyError(f"Missing keys in {stats_path}: {missing}")

    if args.expected_dim is not None:
        for key in args.keys:
            for field in STAT_FIELDS:
                if field in stats[key]:
                    _validate_vector(
                        f"{key}.{field}", stats[key][field], args.expected_dim
                    )

    count = None
    for key in args.keys:
        key_count = stats[key].get("count")
        if isinstance(key_count, list) and key_count:
            count = int(key_count[0])
            break

    payload = {
        "norm_stats": {key: stats[key] for key in args.keys},
        "count": count,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    for key in args.keys:
        if written["norm_stats"][key] != stats[key]:
            raise RuntimeError(f"LingBot norm verification failed for {key}")

    print(f"Wrote and verified LingBot norm stats: {output_path}")


if __name__ == "__main__":
    main()
