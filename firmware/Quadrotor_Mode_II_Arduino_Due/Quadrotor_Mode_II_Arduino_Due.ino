// Terms of use
/****************************************************************************/
/* 
 *  The Sofrware is provided "as is", without warranty of any kind, express or
 *  implied, including but not limited to the warranties or merchantability,
 *  fitness for a particular purpose and noninfringement. In no event shall the
 *  authors or copyright holders be liable for any claim, damages or other liability,
 *  whether in an action of contract, tort or otherwise, arising from, out of
 *  or in connection with the software or the use of other dealings in the software.
*/
/****************************************************************************/
// Safety note ---- Important!!!
/****************************************************************************/
/*
 * Please remove the propellers and stay away from the motors unless you are 
 * 100% certain of what you are doing.
 */
/****************************************************************************/

/****************************************************************************/
/*
   This project is maintained by Wenchao Lei, email: wcleithinking@gmail.com.
   Mode II: The PI-STR-based Inner-Outer Loop Structure
*/
/****************************************************************************/
/*
                    (CW)          |                 (CW)      (CCW)
   QuadP :           0            |   QuadX :         0         1
           (CCW)3   CoM   1(CCW)  |                       CoM
                     2            |                   3         2
                    (CW)          |                 (CCW)      (CW)
*/
/****************************************************************************/
#include "Config.h"
#include <Servo.h>
#include <IMU.h>
#include <SdFat.h>
// <STR.h> is now a deprecated one-line compatibility shim that pulls in the shared
// core (firmware/lib/str_core/str_core.h, pure C++17, no Arduino headers, host
// tested under tests/cpp).  The legacy STR class no longer exists.
#include <STR.h>
Servo     motor[4];
IMU       myIMU;
// PI-STR regulator, 2nd-order ARX model (CONTROL_SPEC §3), float instantiation.
str_core::Str2f mySTR;
/****************************************************************************/
// States of Vehicle, from Sensors and Filters
float Gyro_threemeasure[3]  = {0, 0, 0};
float Gyro_biasestimate[3]  = {0, 0, 0};
float Angle_oldestimate[3]  = {0, 0, 0};
float Angle_newestimate[3]  = {0, 0, 0};
float Angle_offlinebias[3]  = {0, 0, 0};
float Angle_eulerdesire[3]  = {0, 0, 0};
float Rate_threemeasure[3]  = {0, 0, 0};
float Rate_middledesire[3]  = {0, 0, 0};
// Motor Variables
bool ARM_flag = 0;
uint16_t RC[4] = {PWM_MIN, PWM_MIN, PWM_MIN, PWM_MIN};
int PWM_ref[4] = {PWM_MIN, PWM_MIN, PWM_MIN, PWM_MIN};
int PWM_out[4] = {PWM_MIN, PWM_MIN, PWM_MIN, PWM_MIN};
// Main Loop Setting
float dt = LOOP_period / 1000;
uint32_t loop_index  = 0;
uint32_t time_diff[4] = {0, 0, 0, 0}; // timer0, timer1, timer2, timer3 for loop, roll, pitch, yaw, respectively
uint8_t time_Index = 0;
volatile unsigned long time_previous[4], time_current[4];
volatile unsigned long time_start, time_end;
/****************************************************************************/
/****************************************************************************/
void setup() {
#ifdef DEBUG
  Serial.begin(BaudRate);
#endif
  Led_init();
  Led_low();
  Motor_init();
  Motor_update();
  Receiver_init();
  myIMU.init(3, 4, 3, 4, 1, dt);
#ifdef STR_v1
  // Initial parameter guess, §3 analytic form, from the *same* tf_alpha / tf_beta the
  // Python simulator uses (Config.h), scaled by THETA0_PERCENT = 0.75 (§8):
  //   a2 = exp(-tf_beta*h),  a1 = -(1+a2),
  //   b0 = (tf_alpha/tf_beta)*h - (tf_alpha/tf_beta^2)*(1-a2),
  //   b1 = (1/tf_beta)(1-a2)(tf_alpha/tf_beta)(1-a2) - a2*b0
  //
  // A-stage fix (initialisation only, not a control law): the legacy lines used +54/64
  // with atemp positive (a2 = exp(+2.16) = 8.67, an unstable non-minimum-phase model) and
  // multiplied by STR_period = 40 (milliseconds) where h = 0.04 s belongs.
  //   before: theta0 = [-9.6711, +8.6711, +47.5758, -411.2445]   (outside every §6.1 limit)
  //   after : theta0 = [-0.8624, +0.1124, +0.0400, +0.0216]      (yaw axis, STR_Index)
  const float h_est = (float)STR_period / 1000.0f;
  const float tf_alpha_axis = (STR_Index == IndexRoll)  ? TF_ALPHA_ROLL
                            : (STR_Index == IndexPitch) ? TF_ALPHA_PITCH
                                                        : TF_ALPHA_YAW;
  const float a2temp = exp(-TF_BETA * h_est);
  const float a1temp = -1 - a2temp;
  const float b0temp = tf_alpha_axis / TF_BETA * h_est
                     - tf_alpha_axis / (TF_BETA * TF_BETA) * (1 - a2temp);
  const float b1temp = (1 / TF_BETA) * (1 - a2temp) * (tf_alpha_axis / TF_BETA)
                       * (1 - a2temp) - a2temp * b0temp;
  const str_core::Str2f::Vec theta0 = {THETA0_PERCENT * a1temp,
                                       THETA0_PERCENT * a2temp,
                                       THETA0_PERCENT * b0temp,
                                       THETA0_PERCENT * b1temp};
  mySTR.init_estimator((float)0.99, (float)5, theta0);
  // Reference model: zeta = 0.7, wn = 5, mu = 0, h = STR_period in *seconds*.
  // A-stage fix: the legacy call passed `STR_period / 1000`, i.e. integer division = 0,
  // which made Am(z) = 1 - 2z^-1 + z^-2 and the reference gain t0 = Am(1)/B(1) = 0 -- the
  // inner loop had no reference feedforward at all.  Now h = 0.04 s, as in the paper's
  // simulation (the real-time behaviour is not verified on hardware).
  const float h_str = (float)STR_period / 1000.0f;
  mySTR.set_second_order_reference((float)0.7, (float)5, (float)0, h_str);
#endif
#ifdef LOG
  Log_init();
#endif
  Led_blink(3);
  ARM_flag = 0;
  Led_armstate(ARM_flag);
}

void loop() {
  // The ARM state machine lives in exactly one place now (Arm_update(), defined in
  // Driver_Motor.ino): it used to be duplicated here and inside Motor_control().
  while (ARM_flag == 0) {
    Arm_update();
#if MOTOR_ENABLE
    // Stay here until the two arm switches are raised.
#else
    // MOTOR_ENABLE == 0: the vehicle can never arm.  Fall through to the main body
    // so that (with MOTOR_TEST) the control chain and the SD logger still run while
    // every motor output stays at PWM_MIN.
    break;
#endif
  }
  time_current[0] = millis();
  if (time_current[0] - time_previous[0] >= LOOP_period ) {
#ifdef DEBUG
    time_diff[0] = time_current[0] - time_previous[0];
#endif
    time_previous[0] = time_current[0];
    myIMU.sample();
    myIMU.copygyro(Gyro_threemeasure);
    myIMU.attitude_filter(Angle_oldestimate, Gyro_biasestimate);
    Sensor_calibrate();
    Receiver_copy();
    Motor_control();
    Motor_update();
    Controller_updatedata();
#ifdef LOG
#ifdef DEBUG
    time_start = millis();
    Log_savedata();
    time_end = millis();
#else
    Log_savedata();
#endif
#endif
    if (loop_index < 350) loop_index++;
  }
#ifdef DEBUG
  Serial_debug();
#endif
}
/****************************************************************************/
/****************************************************************************/
void Sensor_calibrate() {
  if (loop_index < 100) {
  }
  else if ((loop_index >= 100) && (loop_index < 300)) {
    for (int i = 0; i < 3; i++) Angle_offlinebias[i]  += Angle_oldestimate[i];
  }
  else {
    if (loop_index == 300) {
      for (int i = 0; i < 3; i++) Angle_offlinebias[i] *= 0.005;
    }
    for (int i = 0; i < 3; i++) {
      Angle_newestimate[i]  = Angle_oldestimate[i] - Angle_offlinebias[i];
      Rate_threemeasure[i]  = Gyro_threemeasure[i] - Gyro_biasestimate[i];
    }
  }
}
/****************************************************************************/
/****************************************************************************/
