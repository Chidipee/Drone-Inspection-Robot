#!/usr/bin/env python3
"""
Structural Analysis Pipeline
=============================
Watches the inspection_images/ directory for new photos captured by the drone.
Each new image is sent to Groq (Llama 4 Scout) for structural analysis.
Results are POSTed to the Flask dashboard for real-time display.
"""

import json
import os
import re
import sys
import time
import base64
import requests
from datetime import datetime
from pathlib import Path

# Bootstrap to use venv if not already active
if sys.base_prefix == sys.prefix:
    if os.name == 'nt':
        venv_python = os.path.join(os.path.dirname(__file__), "..", "venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(os.path.dirname(__file__), "..", "venv", "bin", "python")
    
    if os.path.exists(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)

from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print("[ERROR] Set your GROQ_API_KEY in analysis/.env")
    print("[ERROR] Get a free key at https://console.groq.com/keys")
    sys.exit(1)

client = Groq(api_key=GROQ_API_KEY)

DASHBOARD_URL = "http://localhost:5000/api/analysis"
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "inspection_images")

# Groq vision model – Llama 4 Scout (free tier: 30 RPM, 1000 RPD)
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Rate-limit handling
MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 10.0     # seconds – Groq limits are generous so shorter waits
THROTTLE_BETWEEN_IMAGES = 3.0  # seconds between API calls (30 RPM = 1 every 2s)

# ---------------------------------------------------------------------------
# Structural Analysis Prompt
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are a senior structural engineer preparing a professional building inspection report for a client. This photograph was taken by a DJI Mavic 2 Pro inspection drone during a scheduled exterior structural survey.

IMPORTANT INSTRUCTIONS:
- Treat this as a real photograph from a real building inspection. Do NOT comment on image quality, rendering, or whether the image appears synthetic or simulated. Your job is structural analysis only.
- Be thorough and critical. Look for subtle signs of wear, aging, weathering, and potential structural issues. Real buildings always have some degree of wear.
- Identify AT LEAST one area of concern or maintenance recommendation, even if the structure appears generally sound. No building is perfect.
- Write as if this report will be delivered to a building owner or insurance assessor.

Return a JSON object with exactly this structure:

{
  "image_description": "Professional description of the building section visible in this inspection photo",
  "structural_elements": ["list of structural elements visible (e.g., wall, window, roof, foundation, column, beam, mortar joints, flashing, gutter)"],
  "defects_found": [
    {
      "type": "type of defect (e.g., crack, spalling, corrosion, water damage, displacement, efflorescence, settlement, erosion)",
      "severity": "Low | Medium | High | Critical",
      "location": "specific location on the structure",
      "description": "detailed professional description of the defect and its implications"
    }
  ],
  "surface_condition": {
    "overall": "Good | Fair | Poor | Critical",
    "paint_condition": "assessment of paint, coating, or surface finish condition",
    "moisture_signs": "assessment of water damage indicators, staining, or moisture intrusion",
    "biological_growth": "assessment of moss, mold, algae, or vegetation encroachment"
  },
  "risk_assessment": {
    "overall_risk": "Low | Medium | High | Critical",
    "structural_integrity": "professional assessment of structural soundness based on visible evidence",
    "immediate_concerns": ["list of issues that should be addressed within 30 days"],
    "recommended_actions": ["list of specific maintenance or repair actions with priority"]
  },
  "confidence_score": 0.85
}"""

# ---------------------------------------------------------------------------
# Image Analysis Function
# ---------------------------------------------------------------------------

def _parse_retry_delay(error_message: str) -> float:
    """Try to extract the retry delay (in seconds) from a 429 error message."""
    match = re.search(r"retry.after[:\s]*([\d.]+)", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"try again in ([\d.]+)s", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return DEFAULT_RETRY_DELAY


def _encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path: str) -> dict:
    """Send an image to Groq (Llama 4 Scout) and return the structural analysis.

    Includes automatic retry with backoff for rate-limit (429) errors.
    """
    image_b64 = _encode_image_base64(image_path)

    # Determine MIME type
    ext = os.path.splitext(image_path)[1].lower()
    mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png"}.get(ext, "image/jpeg")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ANALYSIS_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content.strip()

            # Strip markdown code fences if present (safety net)
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            analysis = json.loads(text)
            return analysis

        except json.JSONDecodeError as e:
            raw = text[:500] if text else "None"
            print(f"[WARN] Failed to parse Groq response as JSON: {e}")
            print(f"[WARN] Raw response: {raw}")
            return {
                "image_description": "Analysis parsing failed",
                "defects_found": [],
                "surface_condition": {"overall": "Unknown"},
                "risk_assessment": {
                    "overall_risk": "Unknown",
                    "structural_integrity": "Unable to parse response",
                    "immediate_concerns": [],
                    "recommended_actions": ["Re-inspect this area"],
                },
                "confidence_score": 0.0,
            }

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower()

            if is_rate_limit and attempt < MAX_RETRIES:
                delay = _parse_retry_delay(err_str) + 2
                print(f"[RATE LIMIT] Hit limit (attempt {attempt}/{MAX_RETRIES}). "
                      f"Waiting {delay:.0f}s before retry...")
                time.sleep(delay)
                continue
            else:
                print(f"[ERROR] Groq analysis failed: {e}")
                return {
                    "image_description": f"Analysis error: {str(e)[:200]}",
                    "defects_found": [],
                    "surface_condition": {"overall": "Error"},
                    "risk_assessment": {
                        "overall_risk": "Unknown",
                        "structural_integrity": "Analysis failed",
                        "immediate_concerns": [],
                        "recommended_actions": ["Re-inspect this area"],
                    },
                    "confidence_score": 0.0,
                }

    return {"image_description": "Max retries exceeded", "defects_found": [],
            "surface_condition": {"overall": "Error"},
            "risk_assessment": {"overall_risk": "Unknown", "structural_integrity": "Retries exhausted",
                                "immediate_concerns": [], "recommended_actions": ["Re-inspect"]},
            "confidence_score": 0.0}

# ---------------------------------------------------------------------------
# Send results to dashboard
# ---------------------------------------------------------------------------

def send_to_dashboard(image_name: str, image_path: str, analysis: dict):
    """POST analysis result to the Flask dashboard."""
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        image_b64 = ""

    payload = {
        "image_name": image_name,
        "image_base64": image_b64,
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
    }

    try:
        resp = requests.post(DASHBOARD_URL, json=payload, timeout=5)
        if resp.status_code == 200:
            risk = analysis.get("risk_assessment", {}).get("overall_risk", "Unknown")
            print(f"[DASHBOARD] Sent analysis for {image_name} (risk: {risk})")
        else:
            print(f"[WARN] Dashboard returned {resp.status_code}: {resp.text[:200]}")
    except requests.ConnectionError:
        print("[WARN] Dashboard not reachable – is it running on port 5000?")
    except Exception as e:
        print(f"[ERROR] Failed to send to dashboard: {e}")

# ---------------------------------------------------------------------------
# File Watcher
# ---------------------------------------------------------------------------

class ImageHandler(FileSystemEventHandler):
    """Watch for new .jpg files in the inspection_images directory."""

    def __init__(self):
        super().__init__()
        self.processed = set()
        self.last_api_call = 0.0

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle(event.src_path)

    def _handle(self, filepath):
        if not filepath.lower().endswith((".jpg", ".jpeg", ".png")):
            return
        if filepath in self.processed:
            return

        # Small delay to ensure the file is fully written
        time.sleep(0.5)

        filename = os.path.basename(filepath)
        print(f"\n[WATCHER] New image detected: {filename}")
        self.processed.add(filepath)

        # Throttle: wait between API calls to stay within rate limits
        elapsed = time.time() - self.last_api_call
        if elapsed < THROTTLE_BETWEEN_IMAGES:
            wait = THROTTLE_BETWEEN_IMAGES - elapsed
            print(f"[THROTTLE] Waiting {wait:.1f}s before next API call...")
            time.sleep(wait)

        # Analyze with Groq
        print(f"[GROQ] Analyzing {filename}...")
        self.last_api_call = time.time()
        analysis = analyze_image(filepath)

        # Send to dashboard
        send_to_dashboard(filename, filepath, analysis)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_existing_images(handler):
    """Process any images already in the directory (from a previous or ongoing run)."""
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        return

    existing = sorted(image_dir.glob("*.jpg"))
    if existing:
        print(f"[STARTUP] Found {len(existing)} existing images – processing...")
        for img_path in existing:
            handler._handle(str(img_path))


def main():
    print("=" * 60)
    print("  Structural Analysis Pipeline")
    print("  Watching: inspection_images/")
    print(f"  AI Model: Groq Llama 4 Scout")
    print(f"  Model ID: {MODEL_NAME}")
    print(f"  Dashboard: {DASHBOARD_URL}")
    print(f"  Throttle: {THROTTLE_BETWEEN_IMAGES}s between API calls")
    print(f"  Retry: up to {MAX_RETRIES}x on rate-limit errors")
    print("=" * 60)

    # Ensure image directory exists
    os.makedirs(IMAGE_DIR, exist_ok=True)

    handler = ImageHandler()

    # Process existing images first
    process_existing_images(handler)

    # Start file watcher
    observer = Observer()
    observer.schedule(handler, IMAGE_DIR, recursive=False)
    observer.start()

    print(f"\n[WATCHER] Monitoring {IMAGE_DIR} for new images...")
    print("[WATCHER] Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[WATCHER] Stopping...")
        observer.stop()

    observer.join()
    print("[WATCHER] Pipeline stopped.")


if __name__ == "__main__":
    main()
