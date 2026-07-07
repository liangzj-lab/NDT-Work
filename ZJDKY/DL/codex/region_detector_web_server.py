"""Local web UI server for testing the trained region detector."""

from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from predict_regions import load_model
from region_dataset import DEFAULT_OUTPUT_DIR, ID_TO_LABEL, pil_to_float_tensor
from region_postprocess import select_candidates


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HTML = REPO_ROOT / "html" / "region_detector_test.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--weights", type=Path, default=DEFAULT_OUTPUT_DIR / "best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--candidate-score-threshold", type=float, default=0.05)
    parser.add_argument("--selection", default="geometry", choices=("score", "geometry"))
    parser.add_argument("--max-candidates-per-label", type=int, default=8)
    return parser.parse_args()


def decode_data_url(value: str) -> bytes:
    if "," not in value:
        raise ValueError("Image payload must be a data URL.")
    header, payload = value.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Image payload must be base64 encoded.")
    return base64.b64decode(payload)


class RegionDetectorApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.device == "cpu" or not args.device.startswith("cuda"):
            raise RuntimeError("Web testing is configured to use CUDA. Set --device cuda or cuda:<index>.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
        self.device = torch.device(args.device)
        self.model = load_model(args.weights, self.device, None, None)

    @torch.no_grad()
    def predict_image(self, name: str, payload: str) -> dict[str, Any]:
        image_bytes = decode_data_url(payload)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = pil_to_float_tensor(image).to(self.device)
        output = self.model([tensor])[0]

        pred_boxes = output["boxes"].detach().cpu()
        pred_labels = output["labels"].detach().cpu()
        pred_scores = output["scores"].detach().cpu()
        selected = select_candidates(
            pred_boxes,
            pred_labels,
            pred_scores,
            sorted(ID_TO_LABEL),
            self.args.candidate_score_threshold,
            self.args.selection,
            self.args.max_candidates_per_label,
        )

        predictions = []
        for label_id in sorted(ID_TO_LABEL):
            if label_id not in selected:
                continue
            box, score_value, _ = selected[label_id]
            if score_value < self.args.score_threshold and self.args.selection == "score":
                continue
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            predictions.append(
                {
                    "label_id": label_id,
                    "label_name": ID_TO_LABEL[label_id],
                    "score": round(score_value, 6),
                    "x_min": round(x1, 2),
                    "y_min": round(y1, 2),
                    "x_max": round(x2, 2),
                    "y_max": round(y2, 2),
                }
            )

        missing = [
            ID_TO_LABEL[label_id]
            for label_id in sorted(ID_TO_LABEL)
            if label_id not in {row["label_id"] for row in predictions}
        ]
        return {
            "name": name,
            "width": image.width,
            "height": image.height,
            "predictions": predictions,
            "missing": missing,
            "ok": len(predictions) == len(ID_TO_LABEL),
        }


class RegionDetectorHandler(BaseHTTPRequestHandler):
    app: RegionDetectorApp
    html_path: Path

    server_version = "RegionDetectorWeb/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html", "/region_detector_test.html"):
            body = self.html_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "device": str(self.app.device),
                    "weights": str(self.app.args.weights),
                    "score_threshold": self.app.args.score_threshold,
                    "candidate_score_threshold": self.app.args.candidate_score_threshold,
                    "selection": self.app.args.selection,
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            images = payload.get("images") or []
            if not images:
                raise ValueError("No images were provided.")
            results = [
                self.app.predict_image(str(item.get("name", "image")), str(item.get("data", "")))
                for item in images
            ]
            self.send_json({"ok": True, "results": results})
        except Exception as exc:  # noqa: BLE001 - Return errors to local UI.
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def main() -> int:
    args = parse_args()
    if not args.html.exists():
        raise FileNotFoundError(f"Missing HTML file: {args.html}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights file: {args.weights}")

    app = RegionDetectorApp(args)
    RegionDetectorHandler.app = app
    RegionDetectorHandler.html_path = args.html

    server = ThreadingHTTPServer((args.host, args.port), RegionDetectorHandler)
    print(f"Serving {args.html}")
    print(f"Using weights {args.weights}")
    print(f"Open http://{args.host}:{args.port}/")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
