#!/usr/bin/env python3
"""Gate agx_insert_waits region-pass tracing behind AGXWAITS_DEBUG.

The committed pass writes several lines per shader compile to stderr
unconditionally. That tracing is load-bearing evidence (it is how the pass
was proven inert at 84fcd220de1) but it must not run in a timed build.
"""
import sys

PATH = "src/asahi/compiler/agx_insert_waits.c"
ANCHOR = '#include "agx_debug.h"\n'

HELPER = ANCHOR + '''#include <stdlib.h>

/* The region pass traces every frame decision. That tracing is load-bearing
 * evidence - it is how the pass was proven inert at 84fcd220de1 - but it
 * writes several lines per shader compile, so it must not run in a timed
 * build. Opt in with AGXWAITS_DEBUG=1.
 */
static bool
agxwaits_verbose(void)
{
   static int enabled = -1;

   if (enabled < 0)
      enabled = getenv("AGXWAITS_DEBUG") != NULL;

   return enabled;
}

#define AGXWAITS_LOG(...)                                                    \\
   do {                                                                      \\
      if (agxwaits_verbose())                                                \\
         fprintf(stderr, __VA_ARGS__);                                       \\
   } while (0)
'''

src = open(PATH).read()

if "AGXWAITS_LOG" in src:
    print("already gated; no change")
    sys.exit(0)

assert ANCHOR in src, "include anchor missing"
before = src.count("fprintf(stderr")

# Substitute the call sites BEFORE inserting the helper, or the macro's own
# fprintf body gets rewritten into infinite self-reference.
# The one call split across two lines keeps its layout; the rest are one-line.
src = src.replace('fprintf(stderr,\n                       "',
                  'AGXWAITS_LOG(\n                       "')
src = src.replace("fprintf(stderr, ", "AGXWAITS_LOG(")
assert "fprintf(stderr" not in src, "unconverted trace site remains"

src = src.replace(ANCHOR, HELPER, 1)
after = src.count("fprintf(stderr")
assert after == 1, "expected only the macro body to keep fprintf, got %d" % after

open(PATH, "w").write(src)
print("gated %d trace sites; fprintf(stderr occurrences %d -> %d"
      % (before, before, after))
