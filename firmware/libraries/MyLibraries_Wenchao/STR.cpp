// ===========================================================================
// DEPRECATED -- intentionally empty translation unit.
//
// The legacy 2nd-order STR implementation that used to be compiled here (class STR,
// 3x3 adjugate, no guards, the `s0 = xx[2]; s1 = xx[3];` out-of-range bug) has been
// replaced by firmware/lib/str_core/str_core.h, which is header-only and has no
// Arduino dependency.  STR.h now forwards to that header.
//
// This file is kept (rather than deleted) so that the Arduino/PlatformIO library
// layout -- and any build that lists STR.cpp explicitly -- keeps working.  There is
// nothing to compile and nothing to link.
//
// Rationale and the list of fixes: docs/CONTROL_SPEC.md §4-§6, the stage-A report, and
// firmware/SAFETY_NOTES.md.
// ===========================================================================

#include "STR.h"
