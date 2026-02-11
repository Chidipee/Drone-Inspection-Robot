"""
Real-Time Structural Inspection Dashboard
==========================================
Flask server with Server-Sent Events (SSE) for real-time updates.

Endpoints:
  GET  /            - Main dashboard page
  POST /api/analysis - Receive analysis results from the analyzer pipeline
  GET  /stream      - SSE stream of new analysis results
  GET  /api/history - Return all past analysis results as JSON
"""

import sys
import os
import json
import queue
import threading
from datetime import datetime

# Bootstrap to use venv if not already active
if sys.base_prefix == sys.prefix:
    if os.name == 'nt':
        venv_python = os.path.join(os.path.dirname(__file__), "..", "venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(os.path.dirname(__file__), "..", "venv", "bin", "python")
    
    if os.path.exists(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)

from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory storage and SSE infrastructure
# ---------------------------------------------------------------------------

analysis_history = []   # List of all analysis results
history_lock = threading.Lock()

# SSE: each connected client gets its own queue
sse_clients = []
sse_clients_lock = threading.Lock()


def broadcast_event(data: dict):
    """Push an event to all connected SSE clients."""
    message = f"data: {json.dumps(data)}\n\n"
    with sse_clients_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(message)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/analysis", methods=["POST"])
def receive_analysis():
    """Receive an analysis result from the analyzer pipeline."""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    # Add a server-side ID and timestamp
    entry = {
        "id": len(analysis_history) + 1,
        "image_name": data.get("image_name", "unknown"),
        "image_base64": data.get("image_base64", ""),
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "analysis": data.get("analysis", {}),
    }

    with history_lock:
        analysis_history.append(entry)

    # Broadcast to SSE clients
    broadcast_event(entry)

    risk = entry["analysis"].get("risk_assessment", {}).get("overall_risk", "N/A")
    print(f"[DASHBOARD] Received analysis #{entry['id']}: {entry['image_name']} (risk: {risk})")

    return jsonify({"status": "ok", "id": entry["id"]}), 200


@app.route("/stream")
def stream():
    """SSE endpoint – streams new analysis results to the browser."""
    def event_stream():
        q = queue.Queue(maxsize=100)
        with sse_clients_lock:
            sse_clients.append(q)
        try:
            # Send a heartbeat so the browser knows the connection is alive
            yield "data: {\"type\": \"connected\"}\n\n"
            while True:
                try:
                    message = q.get(timeout=30)
                    yield message
                except queue.Empty:
                    # Send keep-alive comment to prevent timeout
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with sse_clients_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/history")
def history():
    """Return all past analysis results (for initial page load)."""
    with history_lock:
        return jsonify(analysis_history)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Structural Inspection Dashboard")
    print("  http://localhost:5000")
    print("=" * 60)
    # threaded=True is needed for SSE to work with Flask dev server
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
