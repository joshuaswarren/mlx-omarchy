/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "agx_nir.h"

/*
 * AGX transcendental lowering with fp32 rounding contracts.
 *
 * The hardware gives an approximate rcp (off by 1 ulp), an approximate log2
 * (absolute error only, so relative error explodes near 1) and a sin unit that
 * takes an argument in quadrants. Everything here is built from the exact fp32
 * fma, so the sequences are marked no-fast-math to keep nir_opt_algebraic from
 * reassociating or contracting them.
 */

/*
 * Correctly rounded 1/x. One Newton-Raphson step on the hardware rcp:
 *
 *    u_2 = u + u(1 - xu) = fma(fma(-x, u, 1), u, u)
 *
 * exhaustively verified upstream with a modified math_bruteforce. If u is
 * infinite (x zero or flushed), the refinement is NaN, so keep u.
 */
static nir_def *
rcp_rn(nir_builder *b, nir_def *x)
{
   nir_def *u = nir_frcp(b, x);
   nir_def *one = nir_imm_float(b, 1.0);
   nir_def *u_2 = nir_ffma(b, nir_ffma(b, nir_fneg(b, x), u, one), u, u);
   return nir_bcsel(b, nir_fisnan(b, u_2), u, u_2);
}

/*
 * Correctly rounded a/b (Markstein). With y = RN(1/b) and q = RN(a y), the
 * residual r = a - b q is exact in an fma and q + r y rounds to RN(a/b) for
 * normal operands. If the refinement produces NaN (b zero, q infinite), q
 * already holds the IEEE result for that case. A zero q is also final (a zero
 * or the quotient flushed) and keeps its sign, which the refinement would
 * lose.
 */
static nir_def *
div_rn(nir_builder *b, nir_def *a, nir_def *d)
{
   nir_def *y = rcp_rn(b, d);
   nir_def *q = nir_fmul(b, a, y);
   nir_def *r = nir_ffma(b, nir_fneg(b, d), q, a);
   nir_def *q_2 = nir_ffma(b, r, y, q);
   nir_def *keep = nir_ior(b, nir_fisnan(b, q_2), nir_feq_imm(b, q, 0.0));
   return nir_bcsel(b, keep, q, q_2);
}

static bool
lower_fdiv(nir_builder *b, nir_alu_instr *alu, void *data)
{
   bool fdiv_only = *(bool *)data;

   if (alu->op != nir_op_fdiv && (alu->op != nir_op_frcp || fdiv_only))
      return false;

   if (alu->def.bit_size == 64)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   b->fp_math_ctrl = nir_fp_no_fast_math;

   nir_def *res;
   if (alu->op == nir_op_frcp) {
      if (alu->def.bit_size != 32)
         return false;

      res = rcp_rn(b, nir_ssa_for_alu_src(b, alu, 0));
   } else {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *d = nir_ssa_for_alu_src(b, alu, 1);

      if (alu->def.bit_size == 32)
         res = div_rn(b, a, d);
      else
         res = nir_fmul(b, a, nir_frcp(b, d));
   }

   nir_def_replace(&alu->def, res);
   return true;
}

bool
agx_nir_lower_fdiv(nir_shader *s)
{
   bool fdiv_only = false;
   return nir_shader_alu_pass(s, lower_fdiv, nir_metadata_control_flow,
                              &fdiv_only);
}

/*
 * Passes that run after agx_preprocess_nir still emit fdiv: nir_lower_int64
 * (f2u64/f2i64 divide by 2^32, and lower_fmod turns the frem into a second
 * fdiv) and agx_nir_lower_interpolation (perspective divide). Lower only fdiv
 * here so the reciprocals built by the first pass and by the log lowering are
 * not refined a second time.
 */
bool
agx_nir_lower_fdiv_late(nir_shader *s)
{
   bool fdiv_only = true;
   return nir_shader_alu_pass(s, lower_fdiv, nir_metadata_control_flow,
                              &fdiv_only);
}

/*
 * Natural log and log2 as double-float (fp32 pair) arithmetic.
 *
 * x = 2^e m with m in [sqrt(1/2), sqrt(2)). With s = (m - 1) / (m + 1),
 *
 *    log(m) = 2s + 2s^3 (1/3 + s^2/5 + s^4/7 + ...)
 *
 * s is kept as a pair (s, s_lo) from one correctly rounded division plus the
 * fma residual, the cubic term is kept as a pair, and the terms are summed
 * with Fast2Sum so the value handed to the final rounding carries about 40
 * bits. That makes the exact cases (powers of two, log(3)) come out correctly
 * rounded and everything else faithful, without any table.
 */
struct log_parts {
   nir_def *e;    /* exponent as float */
   nir_def *s;    /* 2s = leading term, as a pair (s, s_lo) */
   nir_def *s_lo;
   nir_def *t_hi; /* 2 s^3 P(s^2), as a pair */
   nir_def *t_lo;
};

static nir_def *
fast2sum_lo(nir_builder *b, nir_def *a, nir_def *bb, nir_def *sum)
{
   /* Requires |a| >= |bb| or a == 0 */
   return nir_fsub(b, bb, nir_fsub(b, sum, a));
}

static struct log_parts
log_core(nir_builder *b, nir_def *x)
{
   nir_def *bits = x;
   nir_def *e = nir_iadd_imm(b, nir_ushr_imm(b, bits, 23), -127);
   nir_def *m = nir_ior_imm(b, nir_iand_imm(b, bits, 0x7fffff), 0x3f800000);

   /* Fold m into [sqrt(1/2), sqrt(2)) so the series converges fast and
    * e ln2 never cancels more than one bit against log(m).
    */
   nir_def *big = nir_fgt_imm(b, m, 1.41421353816986083984375);
   m = nir_bcsel(b, big, nir_fmul_imm(b, m, 0.5), m);
   e = nir_iadd(b, e, nir_b2i32(b, big));

   nir_def *one = nir_imm_float(b, 1.0);
   nir_def *two = nir_imm_float(b, 2.0);
   nir_def *f = nir_fsub(b, m, one); /* exact */
   nir_def *d = nir_fadd(b, two, f);
   nir_def *d_lo = fast2sum_lo(b, two, f, d);

   /* s = f / (d + d_lo) as a pair */
   nir_def *y = rcp_rn(b, d);
   nir_def *s = nir_fmul(b, f, y);
   nir_def *r = nir_ffma(b, nir_fneg(b, s), d, f);
   s = nir_ffma(b, r, y, s);
   r = nir_ffma(b, nir_fneg(b, s), d, f);
   nir_def *s_lo =
      nir_fmul(b, nir_fsub(b, r, nir_fmul(b, s, d_lo)), y);

   /* u = s^3 as a pair, including the 3 s^2 s_lo cross term */
   nir_def *w = nir_fmul(b, s, s);
   nir_def *w_lo = nir_ffma(b, s, s, nir_fneg(b, w));
   nir_def *u = nir_fmul(b, s, w);
   nir_def *u_lo = nir_fadd(b, nir_ffma(b, s, w, nir_fneg(b, u)),
                            nir_fmul(b, s, w_lo));
   u_lo = nir_ffma(b, nir_fmul_imm(b, s_lo, 3.0), w, u_lo);

   /* P(w) = 1/3 + w Q(w) as a pair, 1/3 split hi/lo */
   nir_def *q = nir_imm_float(b, 1.0 / 13.0);
   q = nir_ffma(b, q, w, nir_imm_float(b, 1.0 / 11.0));
   q = nir_ffma(b, q, w, nir_imm_float(b, 1.0 / 9.0));
   q = nir_ffma(b, q, w, nir_imm_float(b, 1.0 / 7.0));
   q = nir_ffma(b, q, w, nir_imm_float(b, 1.0 / 5.0));
   nir_def *third = nir_imm_float(b, 0.3333333432674408);
   nir_def *v = nir_ffma(b, w, q, nir_imm_float(b, -9.934107758624577e-09));
   nir_def *p = nir_fadd(b, third, v);
   nir_def *p_lo = fast2sum_lo(b, third, v, p);

   nir_def *th = nir_fmul(b, u, p);
   nir_def *tl = nir_fadd(b, nir_fadd(b, nir_ffma(b, u, p, nir_fneg(b, th)),
                                      nir_fmul(b, u, p_lo)),
                          nir_fmul(b, u_lo, p));

   return (struct log_parts){
      .e = nir_i2f32(b, e),
      .s = s,
      .s_lo = s_lo,
      .t_hi = nir_fmul(b, two, th),
      .t_lo = nir_fmul(b, two, tl),
   };
}

/* Positive, finite, not flushed: the only inputs the series handles. */
static nir_def *
is_positive_normal(nir_builder *b, nir_def *x)
{
   return nir_ult_imm(b, nir_iadd_imm(b, x, -0x00800000),
                      0x7f7fffff - 0x00800000 + 1);
}

/*
 * Everything else: +-0 and flushed subnormals give -inf, negatives and NaN
 * give NaN, +inf gives +inf. Same answer for log and log2.
 */
static nir_def *
log_special(nir_builder *b, nir_def *x)
{
   nir_def *zero = nir_ult_imm(b, nir_iand_imm(b, x, 0x7fffffff), 0x00800000);
   nir_def *nan = nir_ugt_imm(b, x, 0x7f800000);
   return nir_bcsel(b, zero, nir_imm_float(b, -INFINITY),
                    nir_bcsel(b, nan, nir_imm_float(b, NAN),
                              nir_imm_float(b, INFINITY)));
}

static nir_def *
soft_log(nir_builder *b, nir_def *x)
{
   struct log_parts l = log_core(b, x);

   /* ln2 split so e * LN2_HI is exact for |e| < 2^8 */
   nir_def *a = nir_fmul_imm(b, l.e, 0.693145751953125);
   nir_def *bb = nir_fmul_imm(b, l.s, 2.0);
   nir_def *h = nir_fadd(b, a, bb);
   nir_def *lo = fast2sum_lo(b, a, bb, h);
   nir_def *h2 = nir_fadd(b, h, l.t_hi);
   nir_def *lo2 = fast2sum_lo(b, h, l.t_hi, h2);

   nir_def *small = nir_fadd(b, lo, lo2);
   small = nir_fadd(b, small, l.t_lo);
   small = nir_fadd(b, small, nir_fmul_imm(b, l.s_lo, 2.0));
   small = nir_ffma(b, l.e, nir_imm_float(b, 1.428606765330187e-06), small);
   return nir_fadd(b, h2, small);
}

static nir_def *
soft_log2(nir_builder *b, nir_def *x)
{
   struct log_parts l = log_core(b, x);

   /* (2s + 2s_lo + t) / ln2 with 1/ln2 as a pair */
   nir_def *il2 = nir_imm_float(b, 1.4426950216293335);
   nir_def *il2_lo = nir_imm_float(b, 1.925963033500011e-08);
   nir_def *bb = nir_fmul_imm(b, l.s, 2.0);
   nir_def *ph = nir_fmul(b, bb, il2);
   nir_def *pl = nir_ffma(b, bb, il2, nir_fneg(b, ph));
   pl = nir_ffma(b, bb, il2_lo, pl);
   nir_def *th = nir_fmul(b, l.t_hi, il2);
   nir_def *tl = nir_ffma(b, l.t_hi, il2, nir_fneg(b, th));
   nir_def *rest = nir_fadd(b, nir_fmul_imm(b, l.s_lo, 2.0), l.t_lo);
   pl = nir_fadd(b, nir_ffma(b, rest, il2, pl), tl);

   nir_def *h = nir_fadd(b, l.e, ph);
   nir_def *lo = fast2sum_lo(b, l.e, ph, h);
   nir_def *h2 = nir_fadd(b, h, th);
   nir_def *lo2 = fast2sum_lo(b, h, th, h2);
   return nir_fadd(b, h2, nir_fadd(b, nir_fadd(b, lo, lo2), pl));
}

static const uint32_t LN2_F32_BITS = 0x3f317218;

static bool
src_is_ln2(nir_alu_instr *alu, unsigned i)
{
   nir_scalar s = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), i);
   return nir_scalar_is_const(s) && nir_scalar_as_uint(s) == LN2_F32_BITS;
}

static bool
src_is_flog2(nir_alu_instr *alu, unsigned i)
{
   nir_scalar s = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), i);
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_flog2;
}

/* nir_flog builds fmul(flog2(x), ln2); catch that before the flog2 goes. */
static bool
lower_log(nir_builder *b, nir_alu_instr *alu, void *_)
{
   if (alu->op != nir_op_fmul || alu->def.bit_size != 32 ||
       alu->def.num_components != 1)
      return false;

   unsigned li;
   if (src_is_flog2(alu, 0) && src_is_ln2(alu, 1))
      li = 0;
   else if (src_is_flog2(alu, 1) && src_is_ln2(alu, 0))
      li = 1;
   else
      return false;

   nir_scalar log = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), li);
   nir_scalar xs = nir_scalar_chase_alu_src(log, 0);

   b->cursor = nir_before_instr(&alu->instr);
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_def *x = nir_channel(b, xs.def, xs.comp);
   nir_def *res = nir_bcsel(b, is_positive_normal(b, x), soft_log(b, x),
                            log_special(b, x));
   nir_def_replace(&alu->def, res);

   if (nir_def_is_unused(log.def))
      nir_instr_remove(nir_def_instr(log.def));

   return true;
}

static bool
lower_log2(nir_builder *b, nir_alu_instr *alu, void *_)
{
   if (alu->op != nir_op_flog2 || alu->def.bit_size != 32)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *res = nir_bcsel(b, is_positive_normal(b, x), soft_log2(b, x),
                            log_special(b, x));
   nir_def_replace(&alu->def, res);
   return true;
}

bool
agx_nir_lower_log(nir_shader *s)
{
   bool progress =
      nir_shader_alu_pass(s, lower_log, nir_metadata_control_flow, NULL);
   progress |=
      nir_shader_alu_pass(s, lower_log2, nir_metadata_control_flow, NULL);
   return progress;
}

/*
 * Sine and cosine. Range reduction produces k = round(x 2/pi) and the fraction
 * fq = x 2/pi - k in [-1/2, 1/2] quadrants; the quadrant then picks a Taylor
 * polynomial for sin(pi/2 fq) or cos(pi/2 fq) and a sign. The hardware sin
 * unit (sin_pt_1/sin_pt_2, nir_op_fsin_agx) is not used: it is a few ulp off,
 * which is more than the reduction now loses.
 *
 * |x| < 2^22: Cody-Waite in radians with pi/2 in three fp32 pieces. The first
 * fma is exact, the remaining rounding is tracked as a low part so fq is
 * within an ulp of the true reduced argument.
 *
 * |x| >= 2^22: Payne-Hanek. Multiply the 24-bit mantissa by a 128-bit window
 * of 2/pi aligned to the exponent; the integer part mod 4 is the quadrant and
 * the top 64 fraction bits give fq. fp32 inputs never reduce below ~2^-30 of a
 * quadrant, so 64 fraction bits leave more than 24 significant bits.
 */
static const double TWO_OVER_PI = 0.6366197466850281;   /* fp32(2/pi) */
static const double TWO_OVER_PI_LO = 2 / M_PI - 0.6366197466850281;
static const double PIO2_1 = 1.5707963705062866;
static const double PIO2_2 = -4.371138828673793e-08;
static const double PIO2_3 = -1.7151245100058819e-15;

/* Bits of 2/pi, 32 per word, preceded by three zero words. */
static const uint32_t two_over_pi_bits[10] = {
   0, 0, 0, 0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0,
   0xdb629599, 0x3c439041, 0xfe5163ab,
};

struct reduced {
   nir_def *k;     /* int32, only the low two bits matter */
   nir_def *fq;    /* about [-1/2, 1/2] in quadrants, as a pair */
   nir_def *fq_lo;
};

static struct reduced
reduce_cody_waite(nir_builder *b, nir_def *x)
{
   nir_def *tp = nir_imm_float(b, TWO_OVER_PI);
   nir_def *c2 = nir_imm_float(b, PIO2_2);
   nir_def *kf = nir_fround_even(b, nir_fmul(b, x, tp));
   nir_def *nk = nir_fneg(b, kf);

   /* r1 = x - k C1 is exact. k C2 is a TwoProduct and r1 - k C2 a TwoSum so
    * nothing is lost even when r1 and k C2 nearly cancel.
    */
   nir_def *r1 = nir_ffma(b, nk, nir_imm_float(b, PIO2_1), x);
   nir_def *p = nir_fmul(b, kf, c2);
   nir_def *p_lo = nir_ffma(b, kf, c2, nir_fneg(b, p));
   nir_def *r2 = nir_fsub(b, r1, p);
   nir_def *bb = nir_fsub(b, r2, r1);
   nir_def *t = nir_fadd(b, nir_fsub(b, r1, nir_fsub(b, r2, bb)),
                         nir_fsub(b, nir_fneg(b, p), bb));
   nir_def *r_lo =
      nir_ffma(b, nk, nir_imm_float(b, PIO2_3), nir_fsub(b, t, p_lo));

   nir_def *fq = nir_fmul(b, r2, tp);
   nir_def *fq_lo = nir_ffma(b, r2, tp, nir_fneg(b, fq));
   fq_lo = nir_ffma(b, r_lo, tp, fq_lo);
   fq_lo = nir_ffma(b, r2, nir_imm_float(b, TWO_OVER_PI_LO), fq_lo);
   nir_def *sum = nir_fadd(b, fq, fq_lo);

   return (struct reduced){
      .k = nir_f2i32(b, kf),
      .fq = sum,
      .fq_lo = fast2sum_lo(b, fq, fq_lo, sum),
   };
}

static struct reduced
reduce_payne_hanek(nir_builder *b, nir_def *ax)
{
   nir_def *e8 = nir_ushr_imm(b, ax, 23);
   nir_def *m32 =
      nir_ishl_imm(b, nir_ior_imm(b, nir_iand_imm(b, ax, 0x7fffff), 0x800000),
                   8);

   /* Word/bit offset of the 2/pi window for this exponent */
   nir_def *off = nir_iadd_imm(b, e8, -126);
   nir_def *idx = nir_ushr_imm(b, off, 5);
   nir_def *sh = nir_iand_imm(b, off, 31);
   nir_def *sh_r = nir_isub(b, nir_imm_int(b, 31), sh);

   nir_def *tab[ARRAY_SIZE(two_over_pi_bits)];
   for (unsigned i = 0; i < ARRAY_SIZE(tab); ++i)
      tab[i] = nir_imm_int(b, two_over_pi_bits[i]);

   /* Windows n = 1..4 at weights 2^0, 2^-32, 2^-64, 2^-96 */
   nir_def *lo[5], *hi[5];
   for (unsigned n = 1; n <= 4; ++n) {
      nir_def *cand[5], *cand2[5];
      for (unsigned j = 0; j < 5; ++j) {
         cand[j] = tab[j + n];
         cand2[j] = tab[j + n + 1];
      }
      nir_def *w0 = nir_select_from_ssa_def_array(b, cand, 5, idx);
      nir_def *w1 = nir_select_from_ssa_def_array(b, cand2, 5, idx);
      nir_def *win = nir_ior(b, nir_ishl(b, w0, sh),
                             nir_ushr_imm(b, nir_ushr(b, w1, sh_r), 1));
      lo[n] = nir_imul(b, m32, win);
      hi[n] = nir_umul_high(b, m32, win);
   }

   /* Carry-propagate the 96 fraction bits, keep the top 64 */
   nir_def *f2 = nir_iadd(b, lo[3], hi[4]);
   nir_def *c2 = nir_b2i32(b, nir_ult(b, f2, hi[4]));
   nir_def *f1a = nir_iadd(b, lo[2], hi[3]);
   nir_def *c1a = nir_b2i32(b, nir_ult(b, f1a, hi[3]));
   nir_def *f1 = nir_iadd(b, f1a, c2);
   nir_def *c1b = nir_b2i32(b, nir_ult(b, f1, f1a));
   nir_def *k = nir_iadd(b, nir_iadd(b, lo[1], hi[2]), nir_iadd(b, c1a, c1b));

   /* Round to nearest quadrant so fq lands in [-1/2, 1/2] */
   nir_def *half = nir_ushr_imm(b, f1, 31);
   k = nir_iadd(b, k, half);
   nir_def *frac = nir_pack_64_2x32_split(b, f2, f1);
   nir_def *is_half = nir_ine_imm(b, half, 0);
   frac = nir_bcsel(b, is_half, nir_ineg(b, frac), frac);
   nir_def *mag = nir_fmul_imm(b, nir_u2f32(b, frac), 0x1p-64);

   return (struct reduced){
      .k = k,
      .fq = nir_bcsel(b, is_half, nir_fneg(b, mag), mag),
      .fq_lo = nir_imm_float(b, 0.0),
   };
}

static bool
sincos_filter(const nir_instr *instr, UNUSED const void *_)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return (alu->op == nir_op_fsin || alu->op == nir_op_fcos) &&
          alu->def.bit_size <= 32;
}

static nir_def *
lower_sincos(nir_builder *b, nir_instr *instr, UNUSED void *_)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_def *x = nir_mov_alu(b, alu->src[0], 1);
   unsigned bit_size = x->bit_size;
   if (bit_size == 16)
      x = nir_f2f32(b, x);

   nir_def *phase = nir_fmul_imm(b, x, 0.6366197466850281);
   if (alu->op == nir_op_fcos)
      phase = nir_fadd_imm(b, phase, 1.0);
   phase = nir_fmul_imm(b, nir_ffract(b, nir_fmul_imm(b, phase, 0.25)), 4.0);
   nir_def *native = nir_fsin_agx(b, phase);
   return bit_size == 16 ? nir_f2f16(b, native) : native;

   nir_def *ax = nir_fabs(b, x);

   nir_push_if(b, nir_fge_imm(b, ax, 0x1p22));
   struct reduced ph = reduce_payne_hanek(b, ax);

   /* Payne-Hanek reduced |x|; sin is odd, so flip k and fq for negatives */
   nir_def *neg_x = nir_flt_imm(b, x, 0.0);
   ph.k = nir_bcsel(b, neg_x, nir_ineg(b, ph.k), ph.k);
   ph.fq = nir_bcsel(b, neg_x, nir_fneg(b, ph.fq), ph.fq);

   nir_push_else(b, NULL);
   struct reduced cw = reduce_cody_waite(b, x);
   nir_pop_if(b, NULL);

   nir_def *k = nir_if_phi(b, ph.k, cw.k);
   nir_def *fq = nir_if_phi(b, ph.fq, cw.fq);
   nir_def *fq_lo = nir_if_phi(b, ph.fq_lo, cw.fq_lo);

   /* cos(x) = sin(x + pi/2): one more quadrant */
   if (alu->op == nir_op_fcos)
      k = nir_iadd_imm(b, k, 1);

   /* Taylor series in quadrants, |fq| < 0.71 (k can be off by one),
    * truncation below 2^-36. The leading sin term and the quadratic cos term
    * are kept as pairs and the low part of fq feeds the derivative, so both
    * land within an ulp.
    */
   nir_def *t = fq;
   nir_def *t_lo = fq_lo;
   nir_def *w = nir_fmul(b, t, t);
   nir_def *w_lo = nir_ffma(b, t, t, nir_fneg(b, w));

   static const double sin_coef[] = {
      -0.6459640860557556,     0.07969262450933456,   -0.004681753925979137,
      0.00016044118092395365, -3.598843250074424e-06, 5.6921727775716136e-08,
   };
   static const double cos_coef[] = {
      -1.2337005138397217,     0.25366950035095215,   -0.020863480865955353,
      0.0009192602592520416, -2.520204179745633e-05, 4.710874748070637e-07,
      -6.386603246255618e-09,
   };
   static const double cos_coef0_lo = -3.629644851343983e-08;

   nir_def *pio2 = nir_imm_float(b, PIO2_1);
   nir_def *p = nir_imm_float(b, sin_coef[5]);
   for (int i = 4; i >= 0; --i)
      p = nir_ffma(b, p, w, nir_imm_float(b, sin_coef[i]));

   nir_def *hi = nir_fmul(b, t, pio2);
   nir_def *lo = nir_ffma(b, t, pio2, nir_fneg(b, hi));
   lo = nir_ffma(b, t, nir_imm_float(b, PIO2_2), lo);
   lo = nir_ffma(b, t_lo, pio2, lo);
   nir_def *sn = nir_fadd(b, hi, nir_ffma(b, nir_fmul(b, t, w), p, lo));

   nir_def *d2 = nir_imm_float(b, cos_coef[0]);
   nir_def *r = nir_imm_float(b, cos_coef[6]);
   for (int i = 5; i >= 1; --i)
      r = nir_ffma(b, r, w, nir_imm_float(b, cos_coef[i]));

   nir_def *h = nir_fmul(b, w, d2);
   nir_def *l = nir_fadd(b, nir_fadd(b, nir_ffma(b, w, d2, nir_fneg(b, h)),
                                     nir_fmul_imm(b, w, cos_coef0_lo)),
                         nir_fmul(b, w_lo, d2));
   l = nir_ffma(b, nir_fmul_imm(b, t_lo, 2.0 * cos_coef[0]), t, l);
   nir_def *one = nir_imm_float(b, 1.0);
   nir_def *s1 = nir_fadd(b, one, h);
   nir_def *e = fast2sum_lo(b, one, h, s1);
   nir_def *cs = nir_fadd(
      b, s1, nir_fadd(b, nir_fadd(b, e, l), nir_fmul(b, nir_fmul(b, w, w), r)));

   /* Odd quadrants take the cosine, quadrants 2 and 3 negate */
   nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, k, 1), 0);
   nir_def *neg = nir_ine_imm(b, nir_iand_imm(b, k, 2), 0);
   nir_def *v = nir_bcsel(b, odd, cs, sn);
   v = nir_bcsel(b, neg, nir_fneg(b, v), v);

   /* Inf and NaN fall into the Payne-Hanek branch with a bogus exponent */
   nir_def *nan = nir_imm_float(b, NAN);
   v = nir_bcsel(b, nir_flt_imm(b, ax, INFINITY), v, nan);

   /* sin(+-0) = +-0 exactly (the reduction returns +0 for -0) */
   if (alu->op == nir_op_fsin)
      v = nir_bcsel(b, nir_feq_imm(b, x, 0.0), x, v);

   if (bit_size == 16)
      v = nir_f2f16(b, v);

   return v;
}

bool
agx_nir_lower_sincos(nir_shader *s)
{
   return nir_shader_lower_instructions(s, sincos_filter, lower_sincos, NULL);
}
