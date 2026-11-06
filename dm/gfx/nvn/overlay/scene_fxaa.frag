#version 450
// FXAA on the game's 3D scene before its interface is drawn over it (nvn_vk.c fx_scene, DM_VK_FXAA=2): the scene copied aside is read here and written back in place.
// As ovl.frag's fxaa (after Timothy Lottes' FXAA 3.11, the low-quality preset); p.hdr 1: a floating-point scene, its luma compressed (l / (1 + l)) so the edge thresholds hold
layout(push_constant) uniform P { vec2 px; float hdr; float pad; } p;
layout(binding = 0) uniform sampler2D src;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
float luma(vec3 c) { float l = dot(c, vec3(0.299, 0.587, 0.114)); return p.hdr > 0.5 ? l / (1.0 + l) : l; }
void main() {
    vec2 t = uv, px = p.px;
    vec4 m4 = texture(src, t); vec3 m = m4.rgb;
    float nw = luma(texture(src, t + vec2(-1.0, -1.0) * px).rgb), ne = luma(texture(src, t + vec2(1.0, -1.0) * px).rgb);
    float sw = luma(texture(src, t + vec2(-1.0, 1.0) * px).rgb), se = luma(texture(src, t + vec2(1.0, 1.0) * px).rgb), lm = luma(m);
    float lo = min(lm, min(min(nw, ne), min(sw, se))), hi = max(lm, max(max(nw, ne), max(sw, se)));
    if (hi - lo < max(0.0312, hi * 0.125)) { o = m4; return; }   // no edge: the pixel as it is
    vec2 d = vec2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
    float red = max((nw + ne + sw + se) * 0.03125, 0.0078125);
    d = clamp(d / (min(abs(d.x), abs(d.y)) + red), -8.0, 8.0) * px;
    vec3 a = 0.5 * (texture(src, t + d * (1.0 / 3.0 - 0.5)).rgb + texture(src, t + d * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(src, t - d * 0.5).rgb + texture(src, t + d * 0.5).rgb);
    float lb = luma(b);
    o = vec4(lb < lo || lb > hi ? a : b, m4.a);   // the scene's alpha kept (some games keep a mask there)
}
