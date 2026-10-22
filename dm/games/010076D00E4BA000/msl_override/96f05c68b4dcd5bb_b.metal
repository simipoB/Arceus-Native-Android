// Hand-written replacement for 96f05c68b4dcd5bb (Risk of Rain 2 1.4.1, the TranslucentImage asset's UI/QuickDumbBlur pixel shader, var19.3), compiled in place of its translation (dmsh_build.py OVERRIDES).
// The blur behind the translucent HUD panels runs as 4–5 full-screen 1280x720 passes a frame. Its Switch code is a 128-tap loop the compiler unrolled with a predicate per tap; translated, that is 13,266
// lines with 129 texture samples, and Metal spilled 1,760 bytes a frame for it: 95% of all fragment time in gameplay (Metal System Trace on an M4, 10 s: 823 ms of every second), the game GPU-bound at
// about 27 fps. Dropping its draws took gameplay to a steady 60 fps (GPU about 5.5 ms a frame). What the translation computes, read off it:
//   N = c3[129].x as int bits; out = (sample(uv) + sum over i = 1..min(N, 128) of sample(uv + c3[129].y * c3[i].xy * c3[0].xy)) / (N + 1)
// (tap i taken while i - 1 < N; coordinates that are NaN read at 0; only .x / .y of c3[0..128] are read). The Switch code adds in half precision, this loop in float.
// The declarations are the translation's, so the argument buffer layout and the reflection (taken from the translation) stay the same; fp_c2 is declared but unused, as in effect it was there.
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;


struct _10
{
    uint _m0;
    uint _m1[8];
    float4 _m2;
    float4 _m3;
    int _m4;
    float _m5[73];
    int4 _m6;
    int _m7;
};

struct _16
{
    float4 _m0[4096];
};

struct spvDescriptorSetBuffer0
{
    constant _16* fp_c3 [[id(0)]];
    constant _16* fp_c2 [[id(1)]];
    texture2d<float> fp_t_cb196610_1056805A [[id(2)]];
    sampler fp_t_cb196610_1056805ASmplr [[id(3)]];
};

struct main0_out
{
    float4 m_32 [[color(0)]];
};

struct main0_in
{
    float4 m_29 [[user(locn0)]];
};

fragment main0_out main0(main0_in in [[stage_in]], constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]])
{
    main0_out out = {};
    constant float4 *c = (*spvDescriptorSet0.fp_c3)._m0;
    texture2d<float> t = spvDescriptorSet0.fp_t_cb196610_1056805A;
    sampler s = spvDescriptorSet0.fp_t_cb196610_1056805ASmplr;
    float2 uv = in.m_29.xy;
    int n = as_type<int>(c[129].x);
    float scale = c[129].y;
    float4 acc = t.sample(s, select(uv, float2(0.0), isnan(uv)));
    int taps = min(n, 128);
    for (int i = 1; i <= taps; i++)
    {
        float2 p = fma(scale * c[i].xy, c[0].xy, uv);
        acc += t.sample(s, select(p, float2(0.0), isnan(p)));
    }
    out.m_32 = acc * (1.0 / (float(n) + 1.0));
    return out;
}
