# Drone Structural Inspection Robot

An autonomous drone-based structural inspection system built with **Webots**, **Groq AI (Llama 4 Scout)**, and a **Flask real-time dashboard**.

The Mavic 2 Pro drone autonomously flies a rectangular path around a building, capturing 4 photos per side (16 total). A Python pipeline analyzes each image using Groq's Llama 4 Scout vision model for structural defect detection, and results stream to a live web dashboard.

---

## Architecture

```
Webots (Mavic 2 Pro)  -->  inspection_images/  -->  Groq Vision Analyzer  -->  Flask Dashboard (SSE)
```

1. **Drone Controller** runs inside Webots, executing an autonomous rectangular flight path
2. **Analysis Pipeline** watches for new images and sends them to Groq (Llama 4 Scout) for structural assessment
3. **Web Dashboard** displays results in real-time via Server-Sent Events

---

## Prerequisites

- **Webots R2025a** (or later) -- [download](https://cyberbotics.com/download)
- **Python 3.10+**
- **Groq API key** (free) -- [get one](https://console.groq.com/keys)

---

## Setup

### 1. Install Python dependencies

```bash
cd "Drone Inspection robot"
pip install -r requirements.txt
```

### 2. Configure your API key

Edit `analysis/.env` and replace the placeholder:

```
GROQ_API_KEY=your_actual_groq_api_key
```

### 3. Configure building dimensions (optional)

Edit `controllers/drone_inspector/config.json`:

```json
{
  "building_length": 20.0,
  "building_breadth": 10.0,
  "building_height": 8.0
}
```

These values control:
- **building_length** / **building_breadth**: The rectangular flight path dimensions (meters)
- **building_height**: The drone flies at half this height

---

## Running

You need **three terminals** running simultaneously:

### Terminal 1 -- Webots Simulation

Open the world file in Webots:

```bash
webots worlds/inspection_world.wbt
```

Or open Webots GUI and load `worlds/inspection_world.wbt`. The drone controller starts automatically when you press Play.

### Terminal 2 -- Web Dashboard

```bash
cd dashboard
python app.py
```

Then open **http://localhost:5000** in your browser.

### Terminal 3 -- AI Analysis Pipeline

```bash
cd analysis
python analyzer.py
```

This watches `inspection_images/` for new photos and sends each to Groq for analysis.

---

## How It Works

### Drone Flight Path

The drone executes this sequence:

1. **Takeoff** to half the building height
2. **Stabilize** for 3 seconds
3. **Side 1** -- Strafe right for `building_length` meters (camera faces the building), 4 photos evenly spaced
4. **Turn** -- Yaw left 90 degrees
5. **Side 2** -- Strafe right for `building_breadth` meters, 4 photos
6. **Turn** -- Yaw left 90 degrees
7. **Side 3** -- Strafe right for `building_length` meters, 4 photos
8. **Turn** -- Yaw left 90 degrees
9. **Side 4** -- Strafe right for `building_breadth` meters, 4 photos
10. **Land**

Total: **16 images** captured at evenly spaced distances along the flight path.

### AI Analysis

Each image is analyzed by Groq's Llama 4 Scout vision model for:
- Crack detection (type, severity, direction)
- Surface deterioration assessment
- Structural risk classification (Low / Medium / High / Critical)
- Recommended maintenance actions

Groq free tier: **30 requests/minute**, **1,000 requests/day** -- more than enough for a full inspection.

### Dashboard

The web dashboard shows:
- **Live image feed** with the latest captured photo
- **Current analysis** with defect details and risk assessment
- **Analysis timeline** -- scrollable history of all inspections
- **Summary statistics** -- total images, risk distribution, health score

---

## Project Structure

```
Drone Inspection robot/
├── worlds/
│   └── inspection_world.wbt          # Webots world file
├── controllers/
│   └── drone_inspector/
│       ├── drone_inspector.py         # Autonomous drone controller
│       └── config.json                # Building dimensions
├── inspection_images/                 # Captured photos (auto-created)
├── analysis/
│   ├── analyzer.py                    # Groq Vision pipeline
│   └── .env                           # API key configuration
├── dashboard/
│   ├── app.py                         # Flask server
│   ├── templates/
│   │   └── index.html                 # Dashboard UI
│   └── static/
│       └── style.css                  # Dashboard styling
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Drone oscillates wildly | The PID controller needs ~3 seconds to stabilize after takeoff. This is normal. |
| "config.json not found" | Make sure you run Webots from the project root, or that the world file references the correct controller. |
| "GROQ_API_KEY not set" | Edit `analysis/.env` with your actual Groq API key from https://console.groq.com/keys |
| Dashboard shows "Reconnecting" | Make sure the Flask server is running (`cd dashboard && python app.py`). |
| No images appearing | Check that the drone simulation is running and the `inspection_images/` directory is being populated. |
| Rate limit errors | Groq free tier allows 30 req/min. The built-in retry logic handles transient 429s automatically. |
