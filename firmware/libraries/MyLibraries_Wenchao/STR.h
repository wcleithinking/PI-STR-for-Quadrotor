// ===========================================================================
// DEPRECATED -- do not add code here.
//
// The PI-STR regulator used to live in this file (class STR, 2nd-order ARX model,
// 3x3 adjugate, explicit (I - K phi') covariance update, no numerical guards).  It
// had the historical out-of-range index bug (`s0 = xx[2]; s1 = xx[3];` on a
// 3-element solution vector) and no protection against a vanishing B(1), a singular
// Diophantine system or a NaN sample, all of which put garbage straight into the
// motor mixer.
//
// The implementation now lives in
//     firmware/lib/str_core/str_core.h   (pure C++17, no Arduino headers)
//     firmware/lib/str_core/str_core.cpp (float/double x order 2/3 instantiations)
// and is exercised on the host by tests/cpp (ctest).  See
// firmware/lib/str_core/README.md and docs/CONTROL_SPEC.md §4-§6.
//
// This header is kept only as a compatibility include so that the sketches and any
// external Arduino project that does `#include <STR.h>` keep building: it simply
// forwards to the new core.  The class `STR` no longer exists -- use
//     str_core::Str2f   (float,  order 2 -- Mode II)
//     str_core::Str3f   (float,  order 3 -- Mode I)
//     str_core::Str2d / Str3d     (double, for the simulation/golden vectors)
// ===========================================================================

#ifndef _STR_H_
#define _STR_H_

// Relative quoted include: resolves to firmware/lib/str_core/str_core.h from this
// file's own directory, which keeps both the Arduino IDE layout
// (libraries/MyLibraries_Wenchao/) and the PlatformIO `lib_extra_dirs` layout working
// without adding anything to the include path.
#include "../../lib/str_core/str_core.h"

#endif  // _STR_H_
