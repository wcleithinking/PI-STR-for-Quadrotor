// ===========================================================================
// PI-STR flight firmware -- Mode I (single-loop STR), Arduino Due.
//
// MOTOR OUTPUT POLICY (stage-A audit fix)
// ---------------------------------------
// `DEBUG` used to force `PWM_out[]` to PWM_MIN inside the controller while the ARM
// state machine still reported ARMED, i.e. the "safe mode" was a side effect of a
// debug switch and the vehicle could claim to be armed with the outputs pinned.
// The two concerns are now separate:
//
//   MOTOR_ENABLE  1 = the ESCs may be driven with the controller output (flight).
//                 0 = HARD INTERLOCK: PWM_out forced to PWM_MIN in the controller
//                     and ARM_flag forced to 0 in the arm state machine.
//   MOTOR_TEST    1 = with MOTOR_ENABLE 0, keep running the control chain so bench
//                     data can be logged while every output stays at PWM_MIN.
//                 0 = do not run the control chain at all while interlocked.
//
// `DEBUG` is now purely a debugging-output switch and has no effect on the motors.
// See firmware/SAFETY_NOTES.md.
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

#define FEEDFORWARD
#define FEEDBACK

// Control Algorithms Settings
//#define PID_v1
#define STR_v1
#define STR_Index             IndexYaw
#define STR_period            100

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
