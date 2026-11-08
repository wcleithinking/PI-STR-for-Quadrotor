void Motor_init() {
  motor[0].attach(motor0Pin);
  motor[1].attach(motor1Pin);
  motor[2].attach(motor2Pin);
  motor[3].attach(motor3Pin);
}

void Motor_update() {
  for (int i = 0; i < 4; i++) motor[i].writeMicroseconds(PWM_out[i]);
}

// ---------------------------------------------------------------------------
// Single ARM state machine (stage-A audit fix P1).
//
// The state machine used to be written twice: the arming condition lived in the
// `while (ARM_flag == 0)` loop of the main sketch, and a second, slightly different
// pair of conditions lived in Motor_control().  There is now exactly one place.
//
// Behaviour is unchanged -- the conditions are the legacy ones, only merged:
//   * ARM   (only from the disarmed state) requires both the roll and the pitch
//     stick in the top corner; that is the condition that was in loop(), with the
//     side effects it had (clear the controller data, restart the timers).
//   * DISARM (from any state) uses the condition that was in Motor_control():
//     roll stick bottom + pitch stick top (the "kill" corner).
// The two conditions are mutually exclusive (roll high vs roll low), so evaluating
// both here is exactly the legacy net behaviour.
//
// MOTOR_ENABLE == 0 is a hard interlock: the machine never arms.  See Config.h.
// ---------------------------------------------------------------------------
void Arm_update() {
#if MOTOR_ENABLE
  Receiver_copy();
  if (ARM_flag == 0) {
    if (RC[IndexRoll] <= (PWM_MAX - 10) || RC[IndexPitch] <= (PWM_MAX - 10)) {
      ARM_flag = 0;
    }
    else {
      ARM_flag = 1;
      Controller_cleardata();
      for (int i = 0; i < 4; i++) time_previous[i] = millis();
    }
  }
  else {
    if (RC[IndexRoll] <= (PWM_MIN + 15) && RC[IndexPitch] >= (PWM_MAX - 15)) {
      ARM_flag = 0;
    }
  }
#else
  Receiver_copy();   // keep RC[] fresh for the logger / ground station
  ARM_flag = 0;
#endif
  Led_armstate(ARM_flag);
}

void Motor_control() {
  if (loop_index < 350) {
    for (int i = 0; i < 4; i++) PWM_out[i] = PWM_MIN;
  }
  else {
    Arm_update();
    if (ARM_flag == 0) {
      Controller_cleardata();
      for (int i = 0; i < 4; i++) PWM_out[i] = PWM_MIN;
#if !MOTOR_ENABLE && MOTOR_TEST
      // Bench mode: run the control chain with the outputs pinned low so that the
      // estimator / PID / STR and the SD log can be exercised on the bench without
      // spinning a motor.  PWM_out is pinned again afterwards because Controller_run()
      // writes the mixed values.
      Receiver_copy();
      Controller_run();
      for (int i = 0; i < 4; i++) PWM_out[i] = PWM_MIN;
#endif
    }
    else if (ARM_flag == 1) {
      Receiver_copy();
      Controller_run();
    }
  }
}
