#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from typing import Any

from experience_bank import load_current_experience_bank
from SelfEvolving.evolve_storage import (
    load_current_tree,
    load_guide_summary,
    load_validation_results,
)
from SelfEvolving.evolve_validation import NoEligibleBestGuideSelectionError, choose_best_tree_node
from utils.generated_paths import configure_generated_path_env


def select_best_assets(
    *,
    dataset_type: str,
    memguide_dir: Path,
    experience_bank_dir: Path,
) -> dict[str, Any]:
    configure_generated_path_env(
        memguide_dir=str(memguide_dir),
        experience_bank_dir=str(experience_bank_dir),
    )
    tree = load_current_tree()
    validation_results = load_validation_results()
    guide_summary = load_guide_summary()
    experience_bank = load_current_experience_bank(dataset_type)

    validation_context = guide_summary.get("validation_context")
    validation_key = (
        str(validation_context.get("validation_key", "") or "").strip()
        if isinstance(validation_context, dict)
        else ""
    )
    bank_hash = str(experience_bank.get("bank_hash", "") or "").strip()

    guide_file: str | None = None
    selection_strategy = "root_fallback"
    selection_info: dict[str, Any] | None = None
    if validation_key:
        for candidate_bank_hash, strategy_suffix in ((bank_hash, "current_bank"), (None, "latest_validation")):
            try:
                # Selection logs are useful during evolution but this helper's stdout is
                # machine-readable (JSON or one filename), so keep those logs out of it.
                with redirect_stdout(io.StringIO()):
                    guide_file, selection_strategy, selection_info = choose_best_tree_node(
                        tree=tree,
                        dataset_type=dataset_type,
                        validation_enabled=True,
                        validation_results=validation_results,
                        validation_key=validation_key,
                        experience_bank_hash=candidate_bank_hash or None,
                    )
                selection_strategy = f"{selection_strategy}:{strategy_suffix}"
                break
            except NoEligibleBestGuideSelectionError:
                continue

    if guide_file is None:
        guide_file = str(tree.get("root_guide_file", "guide_initial.json") or "guide_initial.json")

    return {
        "dataset_type": dataset_type,
        "guide_file": guide_file,
        "guide_path": str((memguide_dir / guide_file).resolve()),
        "experience_file": "current.json",
        "experience_path": str((experience_bank_dir / "current.json").resolve()),
        "experience_bank_version_id": experience_bank.get("version_id"),
        "experience_bank_hash": bank_hash or None,
        "selection_strategy": selection_strategy,
        "validation": selection_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the best validated ForeDreamer assets.")
    parser.add_argument("--dataset_type", required=True, choices=["futurex", "prophet_arena"])
    parser.add_argument("--memguide_dir", required=True)
    parser.add_argument("--experience_bank_dir", required=True)
    parser.add_argument("--field", choices=["json", "guide_file"], default="json")
    args = parser.parse_args()

    selection = select_best_assets(
        dataset_type=args.dataset_type,
        memguide_dir=Path(args.memguide_dir).expanduser().resolve(),
        experience_bank_dir=Path(args.experience_bank_dir).expanduser().resolve(),
    )
    if args.field == "guide_file":
        print(selection["guide_file"])
    else:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
