#version 450
// shape.x: 0 disc, 1 ring, 2 rounded rectangle, 3 a disc's top half; shape.y: ring / border width in uv units; shape.z: corner radius (rectangles); shape.w: rectangle aspect (w / h)
// glyph.x: atlas cell (-1 none), glyph.y: glyph scale (fraction of the control), glyph.z: glyph alpha, glyph.w: 0 white glyph .. 1 black (the game's own button prompts)
// shape.x 4: the game's picture (present_window), from frame over the source rectangle glyph (u0, v0, u1, v1), opaque; shape.y 1: through FXAA (DM_VK_FXAA=1); shape.z 1: sharpened (AMD CAS, DM_VK_SHARPEN=1)
layout(push_constant) uniform P { vec4 rect; vec4 color; vec4 shape; vec4 glyph; vec2 target; vec4 rot; } p;
layout(binding = 0) uniform sampler2D font;
layout(binding = 1) uniform sampler2D frame;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
// FXAA (after Timothy Lottes' FXAA 3.11, the low-quality preset): the edge direction from the luma of the 4 diagonal neighbours, blurred along it, kept within the local luma range
vec3 fxaa(vec2 t) {
    vec2 px = 1.0 / vec2(textureSize(frame, 0)); const vec3 L = vec3(0.299, 0.587, 0.114);
    vec3 m = texture(frame, t).rgb;
    float nw = dot(texture(frame, t + vec2(-1.0, -1.0) * px).rgb, L), ne = dot(texture(frame, t + vec2(1.0, -1.0) * px).rgb, L);
    float sw = dot(texture(frame, t + vec2(-1.0, 1.0) * px).rgb, L), se = dot(texture(frame, t + vec2(1.0, 1.0) * px).rgb, L), lm = dot(m, L);
    float lo = min(lm, min(min(nw, ne), min(sw, se))), hi = max(lm, max(max(nw, ne), max(sw, se)));
    if (hi - lo < max(0.0312, hi * 0.125)) return m;   // no edge: the pixel as it is
    vec2 d = vec2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
    float red = max((nw + ne + sw + se) * 0.03125, 0.0078125);
    d = clamp(d / (min(abs(d.x), abs(d.y)) + red), -8.0, 8.0) * px;
    vec3 a = 0.5 * (texture(frame, t + d * (1.0 / 3.0 - 0.5)).rgb + texture(frame, t + d * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(frame, t - d * 0.5).rgb + texture(frame, t + d * 0.5).rgb);
    float lb = dot(b, L);
    return lb < lo || lb > hi ? a : b;
}
// Contrast-adaptive sharpening (after AMD FidelityFX CAS, MIT): the cross of neighbours weighted negatively by how much room the local min / max leave, so flat areas and
// edges already at full contrast are left alone; sampled a source texel apart, around the scaled position
vec3 cas(vec2 t) {
    vec2 px = 1.0 / vec2(textureSize(frame, 0));
    vec3 a = texture(frame, t + vec2(-1.0, -1.0) * px).rgb, b = texture(frame, t + vec2(0.0, -1.0) * px).rgb, c = texture(frame, t + vec2(1.0, -1.0) * px).rgb;
    vec3 d = texture(frame, t + vec2(-1.0, 0.0) * px).rgb, e = texture(frame, t).rgb, f = texture(frame, t + vec2(1.0, 0.0) * px).rgb;
    vec3 g = texture(frame, t + vec2(-1.0, 1.0) * px).rgb, h = texture(frame, t + vec2(0.0, 1.0) * px).rgb, i = texture(frame, t + vec2(1.0, 1.0) * px).rgb;
    vec3 mn = min(min(min(d, e), min(f, b)), h), mx = max(max(max(d, e), max(f, b)), h);
    mn += min(mn, min(min(a, c), min(g, i))); mx += max(mx, max(max(a, c), max(g, i)));
    vec3 amp = sqrt(clamp(min(mn, 2.0 - mx) / max(mx, vec3(1e-5)), 0.0, 1.0));
    vec3 w = amp * (-1.0 / 6.5);   // sharpness between CAS's 0 (-1/8) and 1 (-1/5)
    return clamp((b * w + d * w + f * w + h * w + e) / (1.0 + 4.0 * w), 0.0, 1.0);
}
void main() {
    if (p.shape.x > 3.5) { vec2 t = mix(p.glyph.xy, p.glyph.zw, uv * 0.5 + 0.5); o = vec4(p.shape.y > 0.5 ? fxaa(t) : p.shape.z > 0.5 ? cas(t) : texture(frame, t).rgb, 1.0); return; }
    float aa = 2.0 / min(p.rect.z, p.rect.w) * 2.0;
    float a;
    if (p.shape.x < 0.5) { float d = length(uv); a = 1.0 - smoothstep(1.0 - aa, 1.0, d); }
    else if (p.shape.x < 1.5) { float d = length(uv); a = (1.0 - smoothstep(1.0 - aa, 1.0, d)) * smoothstep(1.0 - p.shape.y - aa, 1.0 - p.shape.y, d); }
    else if (p.shape.x < 2.5) { vec2 q = abs(uv) * vec2(p.shape.w, 1.0); vec2 b = vec2(p.shape.w, 1.0) - p.shape.z; float d = length(max(q - b, 0.0)) - p.shape.z; a = 1.0 - smoothstep(-aa, 0.0, d); }
    else { float d = length(uv); a = (1.0 - smoothstep(1.0 - aa, 1.0, d)) * (1.0 - smoothstep(-aa, 0.0, uv.y)); }   // 3: the disc's top half
    vec4 col = vec4(p.color.rgb, p.color.a * a);
    if (p.glyph.x >= 0.0) {   // the label: a 16x16 cell of a 16-cell-wide atlas, centred, bilinear then sharpened
        vec2 g = (uv / p.glyph.y) * vec2(p.shape.x > 1.5 ? p.shape.w : 1.0, 1.0) * 0.5 + 0.5;
        if (all(greaterThanEqual(g, vec2(0.0))) && all(lessThanEqual(g, vec2(1.0)))) {
            vec2 t = (vec2(mod(p.glyph.x, 16.0), floor(p.glyph.x / 16.0)) + g) / vec2(16.0, 1.0);
            float s = texture(font, t).r; float ga = smoothstep(0.35, 0.65, s) * p.glyph.z;
            col.rgb = mix(col.rgb, vec3(1.0 - p.glyph.w), ga); col.a = max(col.a, ga);
        }
    }
    o = vec4(col.rgb * col.a, col.a);   // premultiplied
}
