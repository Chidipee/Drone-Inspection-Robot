"""
Drone Structural Inspection Controller
=======================================
Autonomous Mavic 2 Pro controller for building inspection in Webots.

The drone:
  1. Reads building dimensions from config.json
  2. Takes off to half the building height
  3. Flies a rectangular path around the building (strafing right, turning left 90 deg at corners)
  4. Captures 4 photos per side at evenly spaced distances
  5. Lands when it returns to the starting point
"""

import json
import math
import os
import sys

from controller import Robot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(value, high))

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def get_yaw(imu):
    """Return the yaw angle from the IMU (rotation around vertical axis)."""
    return imu.getRollPitchYaw()[2]

def decompose_displacement(dx, dy, heading):
    """Decompose a 2D displacement (dx, dy) into forward and lateral (right)
    components relative to the drone's heading.

    heading: yaw angle in radians (0 = +X, CCW positive).
    Returns (forward_component, right_component).
      - forward_component > 0 means drifted in the camera/facing direction
      - right_component  > 0 means moved to the drone's right (strafe direction)
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    # Forward direction unit vector: (cos_h, sin_h)
    # Right direction unit vector:   (sin_h, -cos_h)
    forward = dx * cos_h + dy * sin_h
    right   = dx * sin_h - dy * cos_h
    return forward, right

# ---------------------------------------------------------------------------
# Flight states
# ---------------------------------------------------------------------------

class FlightState:
    TAKEOFF    = "TAKEOFF"
    STABILIZE  = "STABILIZE"
    SIDE_1     = "SIDE_1"
    TURN_1     = "TURN_1"
    SIDE_2     = "SIDE_2"
    TURN_2     = "TURN_2"
    SIDE_3     = "SIDE_3"
    TURN_3     = "TURN_3"
    SIDE_4     = "SIDE_4"
    LAND       = "LAND"
    DONE       = "DONE"

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        print(f"[CONFIG] Building dimensions: "
              f"L={cfg['building_length']}m  W={cfg['building_breadth']}m  H={cfg['building_height']}m")
        return cfg
    except FileNotFoundError:
        print("[CONFIG] config.json not found – using defaults (20 x 10 x 8)")
        return {"building_length": 20.0, "building_breadth": 10.0, "building_height": 8.0}

# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

def main():
    # ---- Webots robot initialization ----
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # ---- Load building config ----
    config = load_config()
    building_length  = config["building_length"]
    building_breadth = config["building_breadth"]
    building_height  = config["building_height"]
    target_altitude  = building_height / 2.0

    # ---- Sensors ----
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)

    gps = robot.getDevice("gps")
    gps.enable(timestep)

    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    # ---- LEDs ----
    front_left_led  = robot.getDevice("front left led")
    front_right_led = robot.getDevice("front right led")

    # ---- Camera gimbal motors ----
    camera_roll_motor  = robot.getDevice("camera roll")
    camera_pitch_motor = robot.getDevice("camera pitch")

    # ---- Propeller motors (velocity mode) ----
    motor_names = [
        "front left propeller",
        "front right propeller",
        "rear left propeller",
        "rear right propeller",
    ]
    motors = []
    for name in motor_names:
        m = robot.getDevice(name)
        m.setPosition(float("inf"))
        m.setVelocity(1.0)
        motors.append(m)

    # ---- PID constants (from official controller) ----
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P      = 3.0
    K_ROLL_P          = 50.0
    K_PITCH_P         = 30.0

    # ---- Navigation parameters ----
    STRAFE_SPEED       = 1.0    # roll disturbance magnitude for strafing
    ANGLE_TOLERANCE    = 0.05   # radians (~3 deg) – how close before finishing a turn
    STABILIZE_TIME     = 3.0    # seconds to stabilize after takeoff

    # Forward-drift correction: pushes the drone backward when it drifts
    # toward the building during strafing.  Positive pitch = move backward.
    K_FORWARD_CORRECTION = 3.0

    # Camera gimbal tilt: angle the camera downward toward the building face
    # so photos capture the full wall instead of just a narrow horizontal slice.
    # 0.3 rad ≈ 17 degrees downward.
    CAMERA_TILT = 0.3

    # ---- Image capture setup (distance-based: 4 images per side) ----
    IMAGES_PER_SIDE = 4
    image_dir = os.path.join(os.path.dirname(__file__), "..", "..", "inspection_images")
    os.makedirs(image_dir, exist_ok=True)
    image_counter = 0
    # capture_distances holds the distances at which to take each photo on the
    # current side.  images_taken_this_side counts how many have been taken.
    capture_distances = []
    images_taken_this_side = 0

    # ---- State machine variables ----
    state = FlightState.TAKEOFF
    state_start_pos = None   # GPS position when entering a SIDE state
    target_yaw = 0.0         # target yaw for TURN states
    current_side_distance = 0.0  # how far to travel on current side

    # Distances for each side (the four sides of the rectangle)
    side_distances = [building_length, building_breadth, building_length, building_breadth]
    side_index = 0

    # Yaw tracking: initialized after sensors warm up.
    accumulated_yaw = 0.0
    stabilize_start_time = None

    def begin_side(side_dist, gps_position):
        """Prepare capture distances and state for a new side."""
        nonlocal capture_distances, images_taken_this_side
        nonlocal state_start_pos, current_side_distance
        state_start_pos = list(gps_position)
        current_side_distance = side_dist
        images_taken_this_side = 0
        # Evenly spaced capture points: at 25%, 50%, 75%, 100% of side length
        capture_distances = [
            side_dist * (i + 1) / IMAGES_PER_SIDE
            for i in range(IMAGES_PER_SIDE)
        ]
        print(f"[CAMERA] Capture points: {[f'{d:.1f}m' for d in capture_distances]}")

    print("[DRONE] Starting drone inspection controller...")
    print(f"[DRONE] Target altitude: {target_altitude} m")
    print(f"[DRONE] Flight path: {building_length}m -> turn -> {building_breadth}m -> turn -> "
          f"{building_length}m -> turn -> {building_breadth}m -> land")
    print(f"[DRONE] Capturing {IMAGES_PER_SIDE} images per side ({IMAGES_PER_SIDE * 4} total)")

    # ---- Wait 1 second for sensors to initialise ----
    while robot.step(timestep) != -1:
        if robot.getTime() > 1.0:
            break

    # ---- Read initial yaw from IMU to calibrate heading ----
    initial_yaw = get_yaw(imu)
    accumulated_yaw = initial_yaw
    print(f"[DRONE] Initial yaw: {math.degrees(initial_yaw):.1f} deg")

    # ---- Main loop ----
    while robot.step(timestep) != -1:
        time = robot.getTime()

        # -- Read sensors --
        roll  = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        altitude = gps.getValues()[2]
        roll_velocity  = gyro.getValues()[0]
        pitch_velocity = gyro.getValues()[1]
        gps_pos = gps.getValues()  # [x, y, z]
        yaw = get_yaw(imu)

        # -- Blink LEDs --
        led_state = int(time) % 2
        front_left_led.set(led_state)
        front_right_led.set(not led_state)

        # -- Stabilize camera gimbal --
        camera_roll_motor.setPosition(-0.115 * roll_velocity)
        camera_pitch_motor.setPosition(-0.1 * pitch_velocity + CAMERA_TILT)

        # -- Default disturbances --
        roll_disturbance  = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance   = 0.0

        # ===================================================================
        #  STATE MACHINE
        # ===================================================================

        if state == FlightState.TAKEOFF:
            # Simply ascend – target_altitude is already set
            if altitude > target_altitude - 0.3:
                print(f"[DRONE] Reached target altitude {altitude:.1f}m – stabilizing...")
                state = FlightState.STABILIZE
                stabilize_start_time = time

        elif state == FlightState.STABILIZE:
            # Hover for a few seconds to let PID settle
            if time - stabilize_start_time > STABILIZE_TIME:
                print(f"[DRONE] Stabilized. Beginning inspection – SIDE_1 ({building_length}m)")
                state = FlightState.SIDE_1
                side_index = 0
                begin_side(side_distances[0], gps_pos)

        elif state in (FlightState.SIDE_1, FlightState.SIDE_2,
                       FlightState.SIDE_3, FlightState.SIDE_4):
            # -- Strafe right (roll disturbance) --
            roll_disturbance = -STRAFE_SPEED

            # -- Decompose displacement into forward / lateral components --
            dx = gps_pos[0] - state_start_pos[0]
            dy = gps_pos[1] - state_start_pos[1]
            forward_drift, lateral_distance = decompose_displacement(
                dx, dy, accumulated_yaw
            )

            # -- Forward-drift correction --
            # Positive pitch_disturbance = push drone backward (away from building)
            pitch_disturbance = K_FORWARD_CORRECTION * forward_drift

            # -- Yaw correction – maintain current heading --
            yaw_error = normalize_angle(accumulated_yaw - yaw)
            yaw_disturbance = 2.0 * yaw_error

            # Use lateral (rightward) distance as the true strafe progress
            distance_traveled = abs(lateral_distance)

            # -- Distance-based image capture --
            if (images_taken_this_side < IMAGES_PER_SIDE and
                    distance_traveled >= capture_distances[images_taken_this_side]):
                image_counter += 1
                images_taken_this_side += 1
                filename = os.path.join(image_dir, f"capture_{image_counter:04d}.jpg")
                camera.saveImage(filename, 80)  # quality = 80
                side_name = state
                print(f"[CAMERA] {side_name} – photo {images_taken_this_side}/{IMAGES_PER_SIDE} "
                      f"at {distance_traveled:.1f}m  ->  {filename}")

            # -- Check if side is complete --
            if distance_traveled >= current_side_distance:
                if state == FlightState.SIDE_1:
                    print(f"[DRONE] SIDE_1 complete ({distance_traveled:.1f}m). Turning left 90 deg...")
                    accumulated_yaw += math.pi / 2
                    target_yaw = accumulated_yaw
                    state = FlightState.TURN_1
                elif state == FlightState.SIDE_2:
                    print(f"[DRONE] SIDE_2 complete ({distance_traveled:.1f}m). Turning left 90 deg...")
                    accumulated_yaw += math.pi / 2
                    target_yaw = accumulated_yaw
                    state = FlightState.TURN_2
                elif state == FlightState.SIDE_3:
                    print(f"[DRONE] SIDE_3 complete ({distance_traveled:.1f}m). Turning left 90 deg...")
                    accumulated_yaw += math.pi / 2
                    target_yaw = accumulated_yaw
                    state = FlightState.TURN_3
                elif state == FlightState.SIDE_4:
                    print(f"[DRONE] SIDE_4 complete ({distance_traveled:.1f}m). Inspection finished – landing...")
                    state = FlightState.LAND

        elif state in (FlightState.TURN_1, FlightState.TURN_2, FlightState.TURN_3):
            # Turn left by applying yaw disturbance until we reach the target yaw
            yaw_error = normalize_angle(target_yaw - yaw)
            yaw_disturbance = 2.0 * yaw_error

            if abs(yaw_error) < ANGLE_TOLERANCE:
                if state == FlightState.TURN_1:
                    side_index = 1
                    print(f"[DRONE] Turn complete. SIDE_2 ({side_distances[side_index]}m)")
                    state = FlightState.SIDE_2
                elif state == FlightState.TURN_2:
                    side_index = 2
                    print(f"[DRONE] Turn complete. SIDE_3 ({side_distances[side_index]}m)")
                    state = FlightState.SIDE_3
                elif state == FlightState.TURN_3:
                    side_index = 3
                    print(f"[DRONE] Turn complete. SIDE_4 ({side_distances[side_index]}m)")
                    state = FlightState.SIDE_4
                begin_side(side_distances[side_index], gps_pos)

        elif state == FlightState.LAND:
            # Descend by lowering target altitude
            target_altitude = max(0.0, target_altitude - 0.005)
            if altitude < 0.3:
                print("[DRONE] Landed. Inspection complete!")
                state = FlightState.DONE

        elif state == FlightState.DONE:
            # Stop all motors
            for m in motors:
                m.setVelocity(0.0)
            break

        # ===================================================================
        #  PID MOTOR CONTROL (ported from official C controller)
        # ===================================================================
        roll_input  = K_ROLL_P * clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
        pitch_input = K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
        yaw_input   = yaw_disturbance

        clamped_alt_diff = clamp(target_altitude - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input   = K_VERTICAL_P * (clamped_alt_diff ** 3)

        fl = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
        fr = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
        rl = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
        rr = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

        motors[0].setVelocity(fl)
        motors[1].setVelocity(-fr)
        motors[2].setVelocity(-rl)
        motors[3].setVelocity(rr)

    print("[DRONE] Controller finished.")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
