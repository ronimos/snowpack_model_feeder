"""
predict_punstable.py

Run Mayer et al. (2022) trained RandomForestClassifier on the features
produced by extract_punstable_features.py. Outputs P_unstable per valid row.

Must run in an env with scikit-learn 0.22.x (the version the model was
trained with) — modern sklearn 1.x cannot load the pickle correctly.

Per Mayer's code: predict_proba(...)[:, 0] is P(unstable), since the model
was trained with class 0 = unstable, class 1 = stable.

Usage:
  # in .venv-punstable
  python predict_punstable.py
  python predict_punstable.py --features path/to/features.npz \
                              --model path/to/RF_instability_model.sav \
                              --out path/to/predictions.npz
"""
import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features',
        default='/data/snowpack/little_prof/output/'
                'punstable_features.npz')
    p.add_argument('--model',
        default='/data/snowpack/little_prof/'
                'external_models/RF_instability_model.sav')
    p.add_argument('--out',
        default='/data/snowpack/little_prof/output/'
                'punstable_predictions.npz')
    p.add_argument('--batch-size', type=int, default=1_000_000,
                   help='Rows per predict_proba call (memory control)')
    args = p.parse_args()

    print(f"Loading model from {args.model}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # sklearn version warnings
        model = joblib.load(args.model)
    print(f"  type={type(model).__name__}  n_estimators={model.n_estimators}  "
          f"classes_={model.classes_}  n_features_={model.n_features_}")
    if list(model.classes_) != [0., 1.] and list(model.classes_) != [0, 1]:
        raise RuntimeError(
            f"Unexpected classes_ {model.classes_}. Mayer's convention is "
            f"[0=unstable, 1=stable]; rerun with the correct .sav file."
        )

    print(f"Loading features from {args.features}")
    npz = np.load(args.features, allow_pickle=False)
    features = npz['features']      # (N, 6) float32
    indices = npz['indices']        # (N, 3) int32: loc, time, layer
    locations = npz['locations']
    times = npz['times']
    feature_cols = npz['feature_cols']
    n_rows = features.shape[0]
    print(f"  {n_rows:,} rows, features={list(feature_cols)}")
    if features.shape[1] != 6:
        raise RuntimeError(f"Expected 6 features, got {features.shape[1]}")

    # Predict in batches to keep RAM reasonable
    p_unstable = np.empty(n_rows, dtype=np.float32)
    t0 = time.time()
    for start in range(0, n_rows, args.batch_size):
        end = min(start + args.batch_size, n_rows)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # column 0 = P(unstable) under Mayer's convention
            p_unstable[start:end] = model.predict_proba(
                features[start:end].astype(np.float64)
            )[:, 0].astype(np.float32)
        elapsed = time.time() - t0
        pct = end / n_rows
        eta = elapsed / pct * (1 - pct) if pct > 0 else 0
        print(f"  rows {start:>11,}:{end:<11,}  "
              f"[{elapsed:5.0f}s, ETA {eta:4.0f}s]")

    print(f"\nP_unstable distribution:")
    print(f"  min   = {p_unstable.min():.4f}")
    print(f"  p25   = {np.percentile(p_unstable, 25):.4f}")
    print(f"  median= {np.median(p_unstable):.4f}")
    print(f"  p75   = {np.percentile(p_unstable, 75):.4f}")
    print(f"  p95   = {np.percentile(p_unstable, 95):.4f}")
    print(f"  p99   = {np.percentile(p_unstable, 99):.4f}")
    print(f"  max   = {p_unstable.max():.4f}")
    print(f"  frac > 0.77 (Mayer poor threshold) = "
          f"{(p_unstable > 0.77).mean()*100:.2f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        p_unstable=p_unstable,
        indices=indices,
        locations=locations,
        times=times,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"\nSaved to {out} ({size_mb:.0f} MB)")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()

