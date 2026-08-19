import argparse
import json
import sys
import numpy as np
import cv2

from drift_sense.matcher import locate_reference


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense navigation-error recovery inference")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--search", required=True, help="Path to search image")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument("--visualize", default=None, help="Optional path to save an annotated overlay image")
    args = parser.parse_args()

    reference = load_gray(args.reference)
    search = load_gray(args.search)

    result = locate_reference(reference, search)

    if args.visualize:
        overlay = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(overlay, (int(result.x), int(result.y)), (0, 255, 0),
                        cv2.MARKER_CROSS, 28, 2)
        cv2.circle(overlay, (int(result.x), int(result.y)), 16, (0, 255, 0), 2)
        cv2.imwrite(args.visualize, overlay)

    if args.json:
        payload = {
            "x": round(result.x, 2),
            "y": round(result.y, 2),
            "score": round(result.score, 4),
            "scale": round(result.scale, 4),
            "angle_deg": round(result.angle, 2),
            "ambiguous_site_detected": result.ambiguous,
            "candidate_count": result.n_candidates,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"x={result.x:.2f} y={result.y:.2f} score={result.score:.4f} "
              f"scale={result.scale:.4f} angle={result.angle:.2f}deg "
              f"ambiguous={result.ambiguous} candidates={result.n_candidates}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
