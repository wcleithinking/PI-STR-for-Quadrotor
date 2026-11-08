// ===========================================================================
// PI-STR flight firmware -- Mode II (PI-STR inner/outer loop), Arduino Due.
//
// MOTOR OUTPUT POLICY (stage-A audit fix)
// ---------------------------------------
// The legacy file enabled `DEBUG`, and `DEBUG` silently forced `PWM_out[]` to
// PWM_MIN inside the controller while the ARM state machine still reported the
// vehicle as ARMED.  That is not a safe mode: the state machine, the LED and any
// logger all claim "armed" while the motors are pinned, so a half-broken build
// looks like a working one, and the only thing keeping the props still is a side
// effect of a debug switch.
//
// The two concerns are now separate switches:
//
//   MOTOR_ENABLE  1 = the ESCs may be driven with the controller output (flight).
//                 0 = HARD INTERLOCK: PWM_out is forced to PWM_MIN in the
//                     controller *and* ARM_flag is forced to 0 in the arm state
//                     machine, so the LED/log can never claim "armed".
//   MOTOR_TEST    1 = with MOTOR_ENABLE 0, keep running the full control chain
//                     (IMU -> estimator -> PID/STR -> logging) with every output
//                     pinned to PWM_MIN, so bench data can be collected.
//                 0 = do not run the control chain at all while interlocked.
//
// `DEBUG` is now purely a debugging-output switch (Serial, loop timing) and has
// no effect whatsoever on motor output.  See firmware/SAFETY_NOTES.md.
// ===========================================================================

// Debug output only (serial + timing instrumentation) -- no motor effect.
#define BaudRate  115200
#define DEBUG
#define Debug_single

// Log
#define LOG

// Motor output interlock -- see the block comment above.
#define MOTOR_ENABLE          0
#define MOTOR_TEST            1

// Frame Type
#define QuadP

// Indexes
#define IndexRoll             0
#define IndexPitch            1
#define IndexYaw              2
#define IndexAltitude         3

// §3 analytic initial guess, per channel, in the firmware's deg / PWM-unit domain.
// The values are `Params.tf_alpha` / `Params.tf_beta` from the Python stack
// (sim/python/pistr_sim/params.py, the numeric port of Load_SystemParameters.m):
// [roll, pitch, yaw, altitude] in deg/s per PWM unit.  Using the same numbers as the
// simulator means firmware and simulation start from the *same* model (§3 formula).
#define TF_BETA               47.45112887f
#define TF_ALPHA_ROLL       1112.27298381f
#define TF_ALPHA_PITCH      1126.85305913f
#define TF_ALPHA_YAW         114.71504431f
// §8 deliberate initial mismatch (theta0 = THETA0_PERCENT * [a1, a2, b0, b1]).
#define THETA0_PERCENT           0.75f

// Control Algorithms Settings
#define FEEDDESIRE
#define LOOP_period           10
#define PID_v1
#define STR_v1
#define STR_Index             IndexYaw
#define STR_period            40

// Constants Settings
#define Pulse_MIN_Original    1016
#define Pulse_MAX_Original    2016
#define Pulse_MIN             1000
#define Pulse_MAX             2000
#define PWM_MIN               1000
#define PWM_MAX               2000
#define Roll_MIN              -50
#define Roll_MAX              50
#define Pitch_MIN             -50
#define Pitch_MAX             50
#define Yaw_MIN               -90
#define Yaw_MAX               90
#define RadToDeg              (180/PI)
#define DegToRad              (PI/180)

// Pins Settings
#define channel1Pin           2
#define channel2Pin           3
#define channel3Pin           4
#define channel4Pin           5
#define motor0Pin             6
#define motor1Pin             7
#define motor2Pin             8
#define motor3Pin             9
#define CSPin                 10
#define ledPin                13
