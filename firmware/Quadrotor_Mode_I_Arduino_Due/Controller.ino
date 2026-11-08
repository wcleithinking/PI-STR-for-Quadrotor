// Feed Control of Vehicle, from Controllers
float FeedForward[4]  = {0, 0, 0, PWM_MIN};
float FeedBack[4]     = {0, 0, 0, 0};
float FeedAll[4]      = {0, 0, 0, PWM_MIN};
float FeedBackMax[4]  = {50, 50, 100, PWM_MAX};

void Controller() {
#ifdef FEEDFORWARD
  // commands from RC
  Angle_desire[IndexRoll]  = constrain(map(RC[IndexRoll], Pulse_MIN, Pulse_MAX, Roll_MIN, Roll_MAX), Roll_MIN, Roll_MAX);
  Angle_desire[IndexPitch] = constrain(map(RC[IndexPitch], Pulse_MIN, Pulse_MAX, Pitch_MIN, Pitch_MAX), Pitch_MIN, Pitch_MAX);
  Angle_desire[IndexYaw]   = constrain(map(RC[IndexYaw], Pulse_MIN, Pulse_MAX, Yaw_MIN, Yaw_MAX), Yaw_MIN, Yaw_MAX);
  for (int i = 0; i < 3; i++) {
    if ((1490 <= RC[i]) && (RC[i] <= 1510)) {
      Angle_desire[i] = 0;
    }
  }
#endif
  // feedforward
  FeedForward[IndexRoll]      = 0;
  FeedForward[IndexPitch]     = 0;
  FeedForward[IndexYaw]       = 0;
  FeedForward[IndexAltitude]  = constrain(map(RC[IndexAltitude], Pulse_MIN, Pulse_MAX, PWM_MIN, PWM_MAX), PWM_MIN, PWM_MAX);
  // feedback
#ifdef STR_v1 // STR for the chosen angle decided by variable "STR_Index" and PID for the remaining two angles
  switch (STR_Index) {
    case 0: // STR for Roll and PID for Pitch and Yaw
      time_STR_current = millis();
      if (time_STR_current >= time_STR_previous + STR_period) {
#ifdef DEBUG
        time_STR_diff = time_STR_current - time_STR_previous;
        time_STR_previous = time_STR_current;
        time_STR_start = millis();
        FeedBack[IndexRoll]  = STR_controller(IndexRoll);
        time_STR_end = millis();
#else
        time_STR_previous = time_STR_current;
        FeedBack[IndexRoll]  = STR_controller(IndexRoll);
#endif
      }
      FeedBack[IndexPitch]  = PID_controller(IndexPitch);
      FeedBack[IndexYaw]   = PID_controller(IndexYaw);
      break;
    case 1: // STR for Pitch and PID for Roll and Yaw
      time_STR_current = millis();
      if (time_STR_current >= time_STR_previous + STR_period) {
#ifdef DEBUG
        time_STR_diff = time_STR_current - time_STR_previous;
        time_STR_previous = time_STR_current;
        time_STR_start = millis();
        FeedBack[IndexPitch]  = STR_controller(IndexPitch);
        time_STR_end = millis();
#else
        time_STR_previous = time_STR_current;
        FeedBack[IndexPitch]  = STR_controller(IndexPitch);
#endif
      }
      FeedBack[IndexRoll]  = PID_controller(IndexRoll);
      FeedBack[IndexYaw]   = PID_controller(IndexYaw);
      break;
    case 2: // STR for Yaw and PID for Roll and Pitch
      time_STR_current = millis();
      if (time_STR_current >= time_STR_previous + STR_period) {
#ifdef DEBUG
        time_STR_diff = time_STR_current - time_STR_previous;
        time_STR_previous = time_STR_current;
        time_STR_start = millis();
        FeedBack[IndexYaw]  = STR_controller(IndexYaw);
        time_STR_end = millis();
#else
        time_STR_previous = time_STR_current;
        FeedBack[IndexYaw]  = STR_controller(IndexYaw);
#endif
      }
      FeedBack[IndexRoll]   = PID_controller(IndexRoll);
      FeedBack[IndexPitch]  = PID_controller(IndexPitch);
      break;
  }
#else // PID for Roll, Pitch and Yaw
  FeedBack[IndexRoll]   = PID_controller(IndexRoll);
  FeedBack[IndexPitch]  = PID_controller(IndexPitch);
  FeedBack[IndexYaw]   = PID_controller(IndexYaw);
#endif
  // all = feedforward + feedback
  for (int i = 0; i < 4; i++) FeedAll[i] = FeedForward[i] + FeedBack[i];
#if defined(QuadP)
  PWM_ref[0] =  FeedAll[IndexPitch] - FeedAll[IndexYaw] + FeedAll[IndexAltitude];
  PWM_ref[1] = -FeedAll[IndexRoll]  + FeedAll[IndexYaw] + FeedAll[IndexAltitude];
  PWM_ref[2] = -FeedAll[IndexPitch] - FeedAll[IndexYaw] + FeedAll[IndexAltitude];
  PWM_ref[3] =  FeedAll[IndexRoll]  + FeedAll[IndexYaw] + FeedAll[IndexAltitude];
#elif defined(QuadX)
  PWM_ref[0] =  FeedAll[IndexRoll] + FeedAll[IndexPitch] + FeedAll[IndexYaw] + FeedAll[IndexAltitude];
  PWM_ref[1] = -FeedAll[IndexRoll] + FeedAll[IndexPitch] - FeedAll[IndexYaw] + FeedAll[IndexAltitude];
  PWM_ref[2] = -FeedAll[IndexRoll] - FeedAll[IndexPitch] + FeedAll[IndexYaw] + FeedAll[IndexAltitude];
  PWM_ref[3] =  FeedAll[IndexRoll] - FeedAll[IndexPitch] - FeedAll[IndexYaw] + FeedAll[IndexAltitude];
#endif
  // Motor output policy (see Config.h / SAFETY_NOTES.md).  `DEBUG` no longer has any
  // influence here: the only thing that can pin the outputs is MOTOR_ENABLE == 0, and
  // that interlock also forces ARM_flag = 0 so the state machine cannot lie.
#if MOTOR_ENABLE
  for (int i = 0; i < 4; i++) PWM_out[i] = constrain(PWM_ref[i], PWM_MIN, PWM_MAX);
#else
  for (int i = 0; i < 4; i++) PWM_out[i] = PWM_MIN;
#endif
}

/*
  PID Variables
*/
#if defined(PID_v1) // single loop
// outer loop
float Kp_outer = 1.8, Ki_outer = 0.15, Kd_outer = 4.8;
float eTerm_outer[3] = {0, 0, 0};
float PTerm_outer[3] = {0, 0, 0};
float ITerm_outer[3] = {0, 0, 0};
float DTerm_outer[3] = {0, 0, 0};
// inner loop
float eTerm_inner[3] = {0, 0, 0};
#elif defined(PID_v2) // inner-outer loop
// outer loop
float Kp_outer = 1.8, Ki_outer = 0.08, Kd_outer = 4.8;
float eTerm_outer[3] = {0, 0, 0};
float PTerm_outer[3] = {0, 0, 0};
float ITerm_outer[3] = {0, 0, 0};
float DTerm_outer[3] = {0, 0, 0};
// inner loop
float Kp_inner = 1.8, Ki_inner = 0.02, Kd_inner = 4.8;
float eTerm_inner[3] = {0, 0, 0};
float PTerm_inner[3] = {0, 0, 0};
float ITerm_inner[3] = {0, 0, 0};
float DTerm_inner[3] = {0, 0, 0};
float eTerm_inner_old[3] = {0, 0, 0};
// innerinner loop
float eTerm_innerinner[3] = {0, 0, 0};
#elif defined(PID_v3) // digital form (not well tuned)
float K[3]  = {2, 2, 2};
float Ti[3] = {36.3, 36.3, 36.3};
float Td[3] = {9.075, 9.075, 9.075};
float N[3]  = {8, 8, 8};
float eTerm[3]  = {0, 0, 0};
float PTerm[3] = {0, 0, 0};
float ITerm[3] = {0, 0, 0};
float DTerm[3] = {0, 0, 0};
float eTerm_old[3] = {0, 0, 0};
float DTerm_old[3] = {0, 0, 0};
#endif

float PID_controller(uint8_t IndexAngle) {
  uint8_t Index = IndexAngle;
  // FOUND DURING THE STAGE-A AUDIT (needs an owner decision, NOT fixed silently):
  // Mode I's Config.h has `//#define PID_v1` commented out and no PID_v2/v3 either, so
  // the #if/#elif chain below declares PID_out and never assigns it -> the roll and
  // pitch feedback (STR_Index is the yaw axis) was an *uninitialized* value.  That is
  // undefined behaviour, not "no feedback".  Initialising it to 0 makes the outcome
  // deterministic -- zero feedback on the two non-STR axes -- which is the safe reading
  // of "no PID variant selected", but the ship-ready fix is to enable the intended PID
  // variant in Config.h.  Reported in firmware/SAFETY_NOTES.md; do not enable a variant
  // here without a flight test.
  float PID_out = 0;
#if defined(PID_v1)
  eTerm_outer[Index]  = Angle_desire[Index] - Angle_measure[Index] * RadToDeg;
  eTerm_inner[Index]  = Rate_desire[Index]  - Rate_measure[Index]  * RadToDeg;
  PTerm_outer[Index]  = Kp_outer * eTerm_outer[Index];
  ITerm_outer[Index] += Ki_outer * eTerm_outer[Index] * dt;
  // Anti-windup (stage-A audit fix): the output is clamped after the sum, so the
  // integrator must be clamped too, otherwise it keeps winding up while the actuator
  // is saturated.
  ITerm_outer[Index] = constrain(ITerm_outer[Index], -FeedBackMax[Index], FeedBackMax[Index]);
  DTerm_outer[Index]  = Kd_outer * eTerm_inner[Index];
  PID_out = PTerm_outer[Index] + ITerm_outer[Index] + DTerm_outer[Index];
#elif defined(PID_v2)
  eTerm_outer[Index]  = Angle_desire[Index] - Angle_measure[Index] * RadToDeg;
  eTerm_inner[Index]  = Rate_desire[Index]  - Rate_measure[Index]  * RadToDeg;
  PTerm_outer[Index]  = Kp_outer * eTerm_outer[Index];
  ITerm_outer[Index] += Ki_outer * eTerm_outer[Index] * dt;
  ITerm_outer[Index] = constrain(ITerm_outer[Index], -FeedBackMax[Index], FeedBackMax[Index]);
  DTerm_outer[Index]  = Kd_outer * eTerm_inner[Index];
  Rate_desire[Index]  = PTerm_outer[Index] + ITerm_outer[Index] + DTerm_outer[Index];
  Rate_desire[Index]  = constrain(Rate_desire[Index], -FeedBackMax[Index], FeedBackMax[Index]);
  eTerm_inner[Index]  = Rate_desire[Index]  - Rate_measure[Index]  * RadToDeg;
  eTerm_innerinner[Index] = (eTerm_inner[Index] - eTerm_inner_old[Index]) / dt;
  PTerm_inner[Index]  = Kp_inner * eTerm_inner[Index];
  ITerm_inner[Index] += Ki_inner * eTerm_inner[Index] * dt;
  ITerm_inner[Index] = constrain(ITerm_inner[Index], -FeedBackMax[Index], FeedBackMax[Index]);
  DTerm_inner[Index]  = Kd_inner * eTerm_innerinner[Index];
  PID_out = PTerm_inner[Index] + ITerm_inner[Index] + DTerm_inner[Index];
#elif defined(PID_v3)
  eTerm[Index]  = Angle_desire[Index] - Angle_measure[Index] * RadToDeg;
  PTerm[Index]  = K[Index] * eTerm[Index];
  DTerm[Index]  = K[Index] * (exp(-dt * N[Index] / Td[Index]) * DTerm_old[Index] + N[Index] * (eTerm[Index] - eTerm_old[Index]));
  PID_out       = PTerm[Index] + ITerm[Index] + DTerm[Index];
  ITerm[Index]  = K[Index] * (ITerm[Index] + (dt / Ti[Index]) * eTerm[Index]);
  ITerm[Index]  = constrain(ITerm[Index], -FeedBackMax[Index], FeedBackMax[Index]);
#endif
  PID_out = constrain(PID_out, -FeedBackMax[Index], FeedBackMax[Index]);
  return PID_out;
}

#ifdef STR_v1
/*
   STR Variables
   ---------------------------------------------------------------------------
   The regulator itself now lives in firmware/lib/str_core (pure C++17, no Arduino
   headers, host tested by tests/cpp).  What remains here is exactly the legacy glue:
   the per-axis reference polynomials (Am 3rd order, Ao 2nd order), the initial
   parameter guess and the PWM -> u_actual recovery.

   Removed on purpose: the hand-expanded 5x5 adjugate and the 6x6 RLS that used to
   live in this file.  CONTROL_SPEC §5 forbids the adjugate (it is where the
   historical indexing bug lived) and asks for Gaussian elimination with partial
   pivoting, which is what str_core implements for both model orders.
*/
uint8_t STR_flag = 0;
// Inputs and Outputs
// y_desire[] is *never* written by this sketch; the legacy behaviour is that the STR
// axis is regulated to 0 deg.  Preserved verbatim (reported, not "fixed": changing it
// would change flight behaviour).
float y_desire[3][4] = {{0, 0, 0, 0},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0}};
// Last realised differential command per axis (debug / logging only).
float u_actual[3][4] = {{0, 0, 0, 0},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0}};
// Parameters: reference model coefficients, verbatim from the legacy sketch.
float am1[3] = { -1.4431986644, -1.4431986644, -1.92701};
float am2[3] = {  0.7708989340,  0.7708989340,  1.29750};
float am3[3] = { -0.1380692400, -0.1380692400, -0.30120};
float ao1[3] = {0.8, 0.8, 0.8}, ao2[3] = {0.16, 0.16, 0.16};
// Estimator settings (lambda per axis; Pmax = 100 as before).
float lambda[3] = {0.98, 0.98, 0.98};

// Configure the shared core for the STR axis.  Called once from setup(); kept in this
// file because the per-axis constants above are declared here (Arduino concatenates
// the .ino files in directory order and generates the prototypes).
void STR_init() {
  const str_core::Str3f::Vec theta0 = {a1[STR_Index], a2[STR_Index], a3[STR_Index],
                                       b0[STR_Index], b1[STR_Index], b2[STR_Index]};
  mySTR.init_estimator(lambda[STR_Index], 100.0f, theta0);
  const std::array<float, str_core::Str3f::kNumA> am = {
      am1[STR_Index], am2[STR_Index], am3[STR_Index]};
  const std::array<float, str_core::Str3f::kNumR> ao = {ao1[STR_Index],
                                                        ao2[STR_Index]};
  mySTR.set_reference(am, ao);
}

float STR_controller(uint8_t IndexAngle) {
  uint8_t Index = IndexAngle;
  const float y_measure_now = Angle_measure[Index] * RadToDeg;

  // A-stage fix (behaviour change, documented in SAFETY_NOTES.md): the reference
  // channel used to be dead -- y_desire[] was never written, so the STR axis regulated the
  // measured angle to 0 deg instead of following the stick.  It is now fed from
  // Angle_desire[].  The history shift is owned by str_core; y_desire[] is kept as a
  // debug/inspection mirror only.
  for (int i = 3; i >= 1; --i) y_desire[Index][i] = y_desire[Index][i - 1];
  y_desire[Index][0] = Angle_desire[Index];

  // Legacy tick order for Mode I: the control law runs first (using last tick's
  // realised command), then the estimator, then the history shift -- all of which
  // the core owns now.  The realised command is committed from Controller_updatedata()
  // once Motor_update() has produced PWM_out[].
  mySTR.begin_tick(y_desire[Index][0], y_measure_now);
  const float u_computed = mySTR.compute_control(FeedBackMax[Index]);
  mySTR.update_estimator();

  // Publish the current estimates for the SD log / debugger (columns unchanged).
  const str_core::Str3f::Vec& theta_hat = mySTR.theta_hat();
  a1[Index] = theta_hat[0];
  a2[Index] = theta_hat[1];
  a3[Index] = theta_hat[2];
  b0[Index] = theta_hat[3];
  b1[Index] = theta_hat[4];
  b2[Index] = theta_hat[5];

  u_actual[Index][0] = u_computed;
  STR_flag = Index + 1;
  return u_computed;
}
#endif

void Controller_updatedata() {
#if defined(PID_v2)
  for (int i = 0; i < 3; i++) {
    eTerm_inner_old[i] = eTerm_inner[i];
  }
#elif defined(PID_v3)
  for (int i = 0; i < 3; i++) {
    eTerm_old[i] = eTerm[i];
    DTerm_old[i] = DTerm[i];
  }
#endif
#ifdef STR_v1
  if (STR_flag != 0) {
    uint8_t Index = STR_Index;
    // Realised differential command recovered from the saturated motor PWMs.  This is
    // the value the identifier and the control law must see (CONTROL_SPEC §5/§6.4);
    // the legacy code fed the pre-saturation design value into the -ao terms, which is
    // the one deliberate behaviour change of stage A (documented in the report).
    float u_realized = 0;
#if defined(QuadP)
    switch (Index) {
      case 0:
        u_realized = (-PWM_out[1] + PWM_out[3]) / 2;
        break;
      case 1:
        u_realized = ( PWM_out[0] - PWM_out[2]) / 2;
        break;
      case 2:
        u_realized = (-PWM_out[0] + PWM_out[1] - PWM_out[2] + PWM_out[3]) / 4;
        break;
      default:
        u_realized = 0;
        break;
    }
#elif defined(QuadX)
    switch (Index) {
      case 0:
        u_realized = (PWM_out[0] - PWM_out[1] - PWM_out[2] + PWM_out[3]) / 4;
        break;
      case 1:
        u_realized = (PWM_out[0] + PWM_out[1] - PWM_out[2] - PWM_out[3]) / 4;
        break;
      case 2:
        u_realized = (PWM_out[0] - PWM_out[1] + PWM_out[2] - PWM_out[3]) / 4;
        break;
      default:
        u_realized = 0;
        break;
    }
#endif
    mySTR.commit_actual(u_realized);
    // Keep the debug array coherent (histories themselves are owned by str_core).
    u_actual[Index][3] = u_actual[Index][2];
    u_actual[Index][2] = u_actual[Index][1];
    u_actual[Index][1] = u_realized;
    STR_flag = 0;
  }
#endif
}
