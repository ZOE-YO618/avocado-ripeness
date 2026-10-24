"""
Scan one avocado photo, like a shopper at the kiosk.

The final linear-regression model is trained on the saved features in
results/features_200.csv, then used to predict the ripeness of a new photo.
The "days until ready" estimate uses the fitted 20 degC ripening curve
from results/metrics.json.

Usage:
    python scripts/scan_one.py "path/to/photo.jpg"
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so Python can find the avocado_ripeness package

from avocado_ripeness.classifier import fit_model, kiosk_message  # noqa: E402
from avocado_ripeness.features import (FEATURE_NAMES, LIGHTING_INVARIANT_FEATURES,  # noqa: E402
                                       features_from_file)


def main():
    parser = argparse.ArgumentParser(description="Predict the ripeness of one avocado photo.")
    parser.add_argument("image", help="path to an avocado photo")
    args = parser.parse_args()

    # Train the final model on the same 7 lighting-invariant features used in the report.
    data = pd.read_csv(ROOT / "results" / "features_200.csv")
    X = data[LIGHTING_INVARIANT_FEATURES].to_numpy()
    y = data["Ripening Index Classification"].to_numpy()
    model, _ = fit_model("linear", X, y)

    # Room-temperature (20 degC) ripening curve, used for "days until ready".
    curve = json.loads((ROOT / "results" / "metrics.json").read_text())["ripening_curves"]["T20"]["bounded"]["params"]

    # Extract features from the new photo. Unsupported inputs (missing file,
    # no avocado in the picture) get a clear message instead of a crash.
    try:
        feats = dict(zip(FEATURE_NAMES, features_from_file(args.image)))
    except FileNotFoundError:
        print("Could not open that photo - please check the file path.")
        return
    except ValueError:
        print("No avocado detected - please place the fruit in the scanner and try again.")
        return
    x_new = [[feats[name] for name in LIGHTING_INVARIANT_FEATURES]]
    index_pred = float(model.predict(x_new)[0])

    print(kiosk_message(index_pred, curve["k"], curve["t0"]))


if __name__ == "__main__":
    main()
