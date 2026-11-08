// Explicit instantiations of the str_core templates.
//
// This translation unit exists so that both model orders and both scalar types
// are compiled (and therefore warning-checked) even if the firmware sketches only
// instantiate a subset.  Keep the include list minimal: str_core must stay free
// of Arduino / AVR headers.

#include "str_core.h"

namespace str_core {

template class SelfTuningRegulator<float, 2>;
template class SelfTuningRegulator<double, 2>;
template class SelfTuningRegulator<float, 3>;
template class SelfTuningRegulator<double, 3>;

// The three mandated §5/§6 constants, spelled out so a change is visible in a
// diff rather than buried in a template default.
static_assert(Str2f::kNumParams == 4, "Mode II ARX model has 4 parameters");
static_assert(Str3f::kNumParams == 6, "Mode I ARX model has 6 parameters");
static_assert(Str2f::kSolveSize == 3, "Mode II Diophantine is 3x3");
static_assert(Str3f::kSolveSize == 5, "Mode I Diophantine is 5x5");

}  // namespace str_core
