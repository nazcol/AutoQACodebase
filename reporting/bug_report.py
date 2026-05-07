import json
import time
import uuid
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from game_qa.config import BUGS_DIR

logger = logging.getLogger(__name__)


@dataclass
class BugReport:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    ts: float = field(default_factory=time.time)
    episode: int = 0
    step: int = 0
    bug_type: str = "unknown"      
    severity: str = "low"           # low | medium | high | critical
    composite_score: float = 0.0
    description: str = ""
    screenshot_path: str = ""
    reconstruction_path: str = ""
    board_state: list = field(default_factory=list)
    score: int = 0
    action: str = ""
    details: dict = field(default_factory=dict)

    def save(self, screenshot: Optional[np.ndarray] = None, recon: Optional[np.ndarray] = None):
        json_path = BUGS_DIR / f"bug_{self.id}.json"

        if screenshot is not None:
            img_path = BUGS_DIR / f"bug_{self.id}_screenshot.png"
            Image.fromarray(screenshot.astype(np.uint8)).save(img_path)
            self.screenshot_path = str(img_path)

        if recon is not None:
            recon_path = BUGS_DIR / f"bug_{self.id}_recon.png"
            Image.fromarray(recon.astype(np.uint8)).save(recon_path)
            self.reconstruction_path = str(recon_path)

        with open(json_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

        logger.info(
            "Bug saved [%s] type=%s severity=%s score=%.2f",
            self.id, self.bug_type, self.severity, self.composite_score,
        )
        return self

    @classmethod
    def load(cls, path: Path) -> "BugReport":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @staticmethod
    def load_all() -> list["BugReport"]:
        reports = []
        for p in sorted(BUGS_DIR.glob("bug_*.json")):
            try:
                reports.append(BugReport.load(p))
            except Exception as e:
                logger.warning("Failed to load %s: %s", p, e)
        reports.sort(key=lambda r: r.composite_score, reverse=True)
        return reports



# collect bugs
class BugCollector:

    def __init__(self):
        self._bugs: list[BugReport] = []

    def add(self, bug: BugReport):
        self._bugs.append(bug)

    def add_visual_anomaly(
        self,
        episode: int,
        step: int,
        score: float,
        severity: str,
        composite: float,
        screenshot: np.ndarray,
        recon: Optional[np.ndarray],
        board: Optional[np.ndarray],
        game_score: int,
        action: str,
    ):
        bug = BugReport(
            episode=episode,
            step=step,
            bug_type="visual_anomaly",
            severity=severity,
            composite_score=composite,
            description=f"Visual anomaly detected. Reconstruction MSE={score:.4f}",
            board_state=board.tolist() if board is not None else [],
            score=game_score,
            action=action,
            details={"reconstruction_mse": score},
        )
        bug.save(screenshot=screenshot, recon=recon)
        self._bugs.append(bug)
        return bug

    def add_js_error(
        self,
        episode: int,
        step: int,
        error_text: str,
        severity: int,
        screenshot: Optional[np.ndarray] = None,
    ):
        from game_qa.detectors.log_analyzer import severity_label
        bug = BugReport(
            episode=episode,
            step=step,
            bug_type="js_error",
            severity=severity_label(severity),
            composite_score=float(severity),
            description=error_text[:300],
            details={"raw_error": error_text},
        )
        bug.save(screenshot=screenshot)
        self._bugs.append(bug)
        return bug

    def add_performance_bug(
        self,
        episode: int,
        step: int,
        hitch: dict,
        screenshot: Optional[np.ndarray] = None,
    ):
        ratio = hitch.get("ratio", 1.0)
        severity = "high" if ratio > 5 else "medium"
        bug = BugReport(
            episode=episode,
            step=step,
            bug_type="performance",
            severity=severity,
            composite_score=min(ratio * 2, 10.0),
            description=f"Frame hitch: {hitch['delta_ms']:.0f}ms ({ratio:.1f}× average)",
            details=hitch,
        )
        bug.save(screenshot=screenshot)
        self._bugs.append(bug)
        return bug

    def summary(self) -> dict:
        if not self._bugs:
            return {"total": 0, "by_severity": {}, "by_type": {}}

        by_severity = {}
        by_type = {}
        for b in self._bugs:
            by_severity[b.severity] = by_severity.get(b.severity, 0) + 1
            by_type[b.bug_type] = by_type.get(b.bug_type, 0) + 1

        return {
            "total": len(self._bugs),
            "by_severity": by_severity,
            "by_type": by_type,
            "top_bugs": [
                {
                    "id": b.id,
                    "type": b.bug_type,
                    "severity": b.severity,
                    "score": b.composite_score,
                    "description": b.description[:100],
                }
                for b in sorted(self._bugs, key=lambda x: x.composite_score, reverse=True)[:10]
            ],
        }

    @property
    def bugs(self) -> list[BugReport]:
        return list(self._bugs)
