"""Inspect ICD-10 feature names stored in a fitted RSF model.

Run on the remote analysis server, for example:
    python Inspect_ICD_Features.py --ratio 1 --rep 0 --setting tra75

Optionally provide the prevalence table used to construct the diagnosis
features for an exact comparison:
    --prevalence-csv /path/to/prevalence_icd10_sub.csv
"""

import argparse
import os
import pickle
import re
import sys
from difflib import SequenceMatcher

import pandas as pd

# Dynamically discover project import directories without hardcoded user paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    SCRIPT_DIR,
    os.path.abspath("."),
    os.path.join(SCRIPT_DIR, ".."),
    os.path.join(SCRIPT_DIR, "../test"),
    os.path.join(SCRIPT_DIR, "../.."),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)
from rsf_pickle_compat import load_rsf_model  # noqa: E402

STRICT_ICD10 = re.compile(r"^[A-Z][0-9]{2}$")
ICD_LIKE = re.compile(r"^(?:[A-Za-z][._-]?\d{2}(?:[._-]?\d+)?|\d{3})$")


def get_model_feature_names(model):
    """Read fitted feature names from a calibrated RSF wrapper or estimator."""
    candidates = [model, getattr(model, "base_model", None)]
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "named_steps"):
        candidates.extend(base.named_steps.values())
    for estimator in candidates:
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None:
            return [str(name) for name in names]
    raise AttributeError(
        "Could not find feature_names_in_ on the loaded model. "
        "Check that this is a fitted RSF model trained from a pandas DataFrame."
    )


def normalize_code(value):
    """Normalize punctuation and case for spotting likely formatting variants."""
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def inspect_features(feature_names, expected_codes=None, expected_count=395):
    strict = [name for name in feature_names if STRICT_ICD10.fullmatch(name)]
    icd_like = [name for name in feature_names if ICD_LIKE.fullmatch(name.strip())]
    nonstandard = [name for name in icd_like if not STRICT_ICD10.fullmatch(name)]
    report = {
        "model_feature_count": len(feature_names),
        "strict_icd10_count": len(strict),
        "icd_like_count": len(icd_like),
        "nonstandard_icd_like": nonstandard,
        "expected_icd_count": expected_count,
        "count_difference": expected_count - len(strict),
    }

    if expected_codes is not None:
        model_lookup = {normalize_code(name): name for name in feature_names}
        missing = [code for code in expected_codes if code not in feature_names]
        variants = []
        for code in missing:
            normalized = normalize_code(code)
            exact_normalized = [
                name for key, name in model_lookup.items() if key == normalized
            ]
            prefix_variants = [
                name for name in icd_like
                if normalize_code(name).startswith(normalized)
                or normalized.startswith(normalize_code(name))
            ]
            close = sorted(
                feature_names,
                key=lambda name: SequenceMatcher(None, normalized, normalize_code(name)).ratio(),
                reverse=True,
            )[:3]
            candidates = list(dict.fromkeys(exact_normalized + prefix_variants + close))
            variants.append({"expected_code": code, "possible_model_names": candidates})
        report["expected_code_count"] = len(expected_codes)
        report["expected_codes_missing_exactly"] = missing
        report["possible_name_variants"] = variants
    return report


def load_expected_codes(args):
    if not args.prevalence_csv:
        return None
    table = pd.read_csv(args.prevalence_csv, dtype={args.icd_column: str})
    if args.icd_column not in table.columns:
        raise ValueError(f"Column {args.icd_column!r} not found in {args.prevalence_csv}")
    if args.prevalence_column in table.columns:
        prevalence = pd.to_numeric(table[args.prevalence_column], errors="coerce")
        table = table.loc[prevalence > args.prevalence_threshold]
    return table[args.icd_column].dropna().astype(str).str.strip().drop_duplicates().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--setting", choices=("tra75", "over75calib"), default="tra75")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--expected-count", type=int, default=395)
    parser.add_argument("--prevalence-csv", default=None)
    parser.add_argument("--icd-column", default="icd10")
    parser.add_argument("--prevalence-column", default="prevalence")
    parser.add_argument("--prevalence-threshold", type=float, default=0.01)
    parser.add_argument("--feature-list", default=None,
                        help="Text file for all fitted feature names (default: current folder)")
    default_root = os.environ.get("DESC_DATA_ROOT", "./data")
    parser.add_argument("--root", default=default_root,
                        help="Data root directory (default: ./data or $DESC_DATA_ROOT)")
    parser.add_argument("--output-csv", default=None,
                        help="Audit CSV path (default: current folder)")
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(
        args.root, f"ratio_{args.ratio}", f"Model_{args.rep}_x1_{args.setting}.pkl"
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"RSF model not found: {model_path}")
    model = load_rsf_model(model_path, args.setting)

    names = get_model_feature_names(model)
    stem = f"RSF_features_ratio{args.ratio}_rep{args.rep}_{args.setting}"
    feature_list_path = args.feature_list or os.path.join(os.getcwd(), f"{stem}.txt")
    with open(feature_list_path, "w", encoding="utf-8") as feature_file:
        feature_file.write("\n".join(names) + "\n")
    print(f"All fitted feature names saved: {feature_list_path}")

    expected = load_expected_codes(args)
    report = inspect_features(names, expected, args.expected_count)

    print(f"Model: {model_path}")
    print(f"Fitted feature count: {report['model_feature_count']}")
    print(f"Strict ICD-10 three-character features: {report['strict_icd10_count']}")
    print(f"Expected ICD feature count: {report['expected_icd_count']}")
    print(f"Difference (expected - strict): {report['count_difference']}")
    print("ICD-like feature names that do not match [A-Z][0-9]{2}:")
    if report["nonstandard_icd_like"]:
        for name in report["nonstandard_icd_like"]:
            print(f"  {name!r}")
    else:
        print("  (none detected by the broad code-like pattern)")

    if expected is not None:
        print(f"Expected codes missing by exact name: {len(report['expected_codes_missing_exactly'])}")
        for code in report["expected_codes_missing_exactly"]:
            candidates = next(
                row["possible_model_names"]
                for row in report["possible_name_variants"]
                if row["expected_code"] == code
            )
            print(f"  {code!r}; possible model-name matches: {candidates}")

    audit_path = args.output_csv or os.path.join(os.getcwd(), f"{stem}_audit.csv")
    rows = [{
        "model_path": model_path,
        "feature_index": index,
        "feature_name": name,
        "strict_icd10_3char": bool(STRICT_ICD10.fullmatch(name)),
        "icd_like_nonstandard": bool(
            ICD_LIKE.fullmatch(name.strip()) and not STRICT_ICD10.fullmatch(name)
        ),
    } for index, name in enumerate(names)]
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    print(f"Feature audit saved: {audit_path}")


if __name__ == "__main__":
    main()
