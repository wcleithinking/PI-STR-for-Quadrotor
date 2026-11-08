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
// It used to be written twice -- the arming condition in the `while (ARM_flag == 0)`
// loop of the main sketch and a second, slightly different pair of conditions in
// Motor_control().  There is now exactly one place, with the legacy conditions
// merged and unchanged:
//   * ARM   (only from the disarmed state): both roll and pitch sticks in the top
//     corner; side effects preserved (restart the loop timers, Mode I has no
//     controller data to clear -- its integrators are reset while disarmed).
//   * DISARM (from any state): roll stick bottom + pitch stick top (the kill corner).
// The two conditions are mutually exclusive, so this is exactly the legacy behaviour.
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
#ifdef STR_v1
      time_STR_previous = millis();
#endif
      time_previous = millis();
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
  Led_armstate();
}

void Motor_control() {
  if (loop_index < 350) {
    for (int i = 0; i < 4; i++) PWM_out[i] = PWM_MIN;
  }
  else {
    Arm_update();
    if (ARM_flag == 0) {
      Reset_ITerm();
      for (int i = 0; i < 4; i++) PWM_out[i] = PWM_MIN;
#if !MOTOR_ENABLE && MOTOR_TEST
      // Bench mode: run the control chain with the outputs pinned low so the estimator
      // and the SD log can be exercised without spinning a motor.  PWM_out is pinned
      // again afterwards because Controller() writes the mixed values.
      Receiver_copy();
      Controller();
      for (int i = 0; i < 4; i++) PWM_out[i] = PWM_MIN;
#endif
    }
    else if (ARM_flag == 1) {
      Receiver_copy();
      Controller();
    }
  }
}

void Reset_ITerm() {
#if defined(PID_v1)
  for (int i = 0; i < 3; i++) {
    ITerm_outer[i] = 0;
  }
#elif defined (PID_v2)
  for (int i = 0; i < 3 ; i++) {
    ITerm_outer[i] = 0;
    ITerm_inner[i] = 0;
  }
#elif defined(PID_v3)
  for (int i = 0; i < 3; i++) {
    ITerm[i] = 0;
  }
#endif
}
