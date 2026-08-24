from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import threading
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


VALID_DECISIONS = {
    "valid",
    "ambiguous",
    "wrong_bbox",
    "missing_object",
    "tiny_unusable",
    "modality_misaligned",
    "query_corrected",
    "exclude",
}
VALID_CAUSES = {
    "wrong_instance",
    "ordinal_error",
    "direction_error",
    "boundary_error",
    "missed_object",
    "small_object",
    "data_ambiguous",
    "modality_needed",
    "modality_useless",
}


def box_iou(prediction: object, target: object) -> float | None:
    if not (
        isinstance(prediction, list)
        and isinstance(target, list)
        and len(prediction) == len(target) == 4
    ):
        return None
    pred = [float(value) for value in prediction]
    truth = [float(value) for value in target]
    ix1, iy1 = max(pred[0], truth[0]), max(pred[1], truth[1])
    ix2, iy2 = min(pred[2], truth[2]), min(pred[3], truth[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pred_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    truth_area = max(0.0, truth[2] - truth[0]) * max(0.0, truth[3] - truth[1])
    union = pred_area + truth_area - intersection
    return intersection / union if union else 0.0


def query_type(query: str) -> str:
    text = query.lower()
    if "nearby" in text:
        return "relational"
    if "leftmost" in text or "rightmost" in text:
        return "extreme"
    if any(word in text for word in ("second", "third", "fourth", "fifth")):
        return "ordinal"
    if any(term in text for term in ("center of", "upper part", "lower part", "left side", "right side")):
        return "absolute_position"
    return "other"


def risk_score(sample: dict, prediction_iou: float | None, same_class_count: int) -> int:
    score = {"ordinal": 5, "extreme": 3, "relational": 1}.get(query_type(sample["query"]), 0)
    score += max(0, same_class_count - 1)
    score += {"tiny": 5, "small": 3, "medium": 1}.get(sample.get("scale_bin"), 0)
    if prediction_iou is not None and prediction_iou < 0.5:
        score += 2
    return score


def load_predictions(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("id")
        if not isinstance(sample_id, str):
            raise ValueError(f"Prediction line {line_number} has no string id")
        rows[sample_id] = row
    return rows


def build_dataset(manifest_path: Path, predictions_path: Path | None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object keyed by sample id")
    predictions = load_predictions(predictions_path)
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for sample_id, sample in manifest.items():
        grouped[str(sample["visible"])].append((sample_id, sample))

    scenes = []
    for visible, rows in sorted(grouped.items()):
        class_counts = Counter(sample["class_name"] for _, sample in rows)
        samples = []
        for sample_id, sample in rows:
            prediction_row = predictions.get(sample_id, {})
            prediction = prediction_row.get("prediction")
            prediction_iou = box_iou(prediction, sample["bbox"])
            same_class_count = class_counts[sample["class_name"]]
            samples.append(
                {
                    "id": sample_id,
                    "query": sample["query"],
                    "query_type": query_type(sample["query"]),
                    "class_name": sample["class_name"],
                    "bbox": sample["bbox"],
                    "scale_bin": sample.get("scale_bin"),
                    "weak_label": bool(sample.get("weak_label", False)),
                    "prediction": prediction,
                    "prediction_iou": prediction_iou,
                    "generation_cap_hit": prediction_row.get("generation_cap_hit"),
                    "same_class_count": same_class_count,
                    "risk_score": risk_score(sample, prediction_iou, same_class_count),
                }
            )
        scenes.append(
            {
                "id": Path(visible).stem,
                "visible": visible,
                "infrared": rows[0][1]["infrared"],
                "depth": rows[0][1]["depth"],
                "samples": samples,
                "risk_score": max(sample["risk_score"] for sample in samples),
            }
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "predictions": str(predictions_path) if predictions_path else None,
        "scene_count": len(scenes),
        "sample_count": len(manifest),
        "scenes": scenes,
    }


def safe_image_path(data_root: Path, relative: str) -> Path:
    root = data_root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise FileNotFoundError(relative)
    return candidate


class ReviewStore:
    def __init__(self, path: Path, manifest_sha256: str):
        self.path = path
        self.lock = threading.Lock()
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data.get("manifest_sha256") != manifest_sha256:
                raise ValueError("Review file belongs to a different manifest")
        else:
            self.data = {
                "version": 1,
                "manifest_sha256": manifest_sha256,
                "updated_at": None,
                "reviews": {},
            }

    def update(self, sample_ids: list[str], patch: dict) -> dict:
        decision = patch.get("decision")
        cause = patch.get("cause")
        corrected_query = patch.get("corrected_query")
        if decision is not None and decision not in VALID_DECISIONS:
            raise ValueError(f"Unknown decision: {decision}")
        if cause is not None and cause not in VALID_CAUSES:
            raise ValueError(f"Unknown cause: {cause}")
        if corrected_query is not None and not isinstance(corrected_query, str):
            raise ValueError("corrected_query must be a string or null")
        allowed = {"decision", "cause", "note", "corrected_query"}
        clean_patch = {key: value for key, value in patch.items() if key in allowed}
        with self.lock:
            for sample_id in sample_ids:
                current = self.data["reviews"].setdefault(sample_id, {})
                for key, value in clean_patch.items():
                    if value in (None, ""):
                        current.pop(key, None)
                    else:
                        current[key] = value
                current["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            return {sample_id: self.data["reviews"][sample_id] for sample_id in sample_ids}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address, handler, *, dataset, data_root, store, html_path):
        super().__init__(address, handler)
        self.dataset = dataset
        self.data_root = data_root.resolve()
        self.store = store
        self.html_path = html_path

    def image_path(self, relative: str) -> Path:
        return safe_image_path(self.data_root, relative)


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(self.server.html_path.read_bytes(), "text/html; charset=utf-8")
        elif parsed.path == "/api/bootstrap":
            payload = {**self.server.dataset, "review_state": self.server.store.data}
            self._send_json(payload)
        elif parsed.path == "/api/export":
            self._send_json(self.server.store.data)
        elif parsed.path == "/image":
            try:
                relative = parse_qs(parsed.query).get("path", [""])[0]
                image_path = self.server.image_path(relative)
                mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                self._send_bytes(image_path.read_bytes(), mime_type)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            sample_ids = body.get("sample_ids")
            if not isinstance(sample_ids, list) or not all(isinstance(item, str) for item in sample_ids):
                raise ValueError("sample_ids must be a list of strings")
            updated = self.server.store.update(sample_ids, body.get("patch", {}))
            self._send_json({"ok": True, "updated": updated})
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format_string: str, *args) -> None:
        return

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_bytes(
        self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast scene-level grounding dataset reviewer")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the reviewer in the default browser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    data_root = (args.data_root or manifest_path.parent).resolve()
    predictions_path = args.predictions.resolve() if args.predictions else None
    reviews_path = (args.reviews or manifest_path.with_name(f"{manifest_path.stem}_reviews.json")).resolve()
    dataset = build_dataset(manifest_path, predictions_path)
    store = ReviewStore(reviews_path, dataset["manifest_sha256"])
    server = ReviewServer(
        (args.host, args.port),
        Handler,
        dataset=dataset,
        data_root=data_root,
        store=store,
        html_path=Path(__file__).with_name("review_grounding.html"),
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"Reviewing {dataset['sample_count']} samples in {dataset['scene_count']} scenes")
    print(f"Saving reviews to {reviews_path}")
    print(f"Open {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
