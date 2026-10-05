"""Command line entry point: ``python -m chronocleave.cli <command>``."""
from __future__ import annotations

import argparse
import json
import logging

from chronocleave.config import load_config, resolve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chronocleave", description="ChronoCleave AI pipeline")
    parser.add_argument("command", choices=["simulate", "train", "report", "all"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logging.getLogger("alembic").setLevel(logging.WARNING)
    cfg = load_config(args.config)

    if args.command in {"simulate", "all"}:
        from chronocleave.data.simulate import simulate_cohort

        path = resolve(cfg.data.raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = simulate_cohort(cfg.data.n_patients, cfg.data.day3_transfer_share, cfg.data.annotation_gap_rate, cfg.seed)
        frame.to_csv(path, index=False)
        logging.info("Wrote %d embryos from %d patients to %s", len(frame), frame["patient_id"].nunique(), path)
    if args.command in {"train", "all"}:
        from chronocleave.training.pipeline import run_pipeline

        report = run_pipeline(cfg)
        print(json.dumps({"selected": report["selected_model"], "test": report["results"][report["selected_model"]]["test"]}, indent=2))
    if args.command in {"report", "all"}:
        from chronocleave.evaluation.report import generate_figures

        print("Figures:", ", ".join(generate_figures(cfg)))


if __name__ == "__main__":
    main()
