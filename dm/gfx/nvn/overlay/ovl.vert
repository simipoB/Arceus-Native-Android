#version 450
// Touch overlay (nvn_vk.c ovl_draw): one quad per control, from its rectangle in pixels and the target size in the push constants. rot (cos, sin): the swapchain's
// pre-rotation, applied to the clip position (identity: 1, 0)
layout(push_constant) uniform P { vec4 rect; vec4 color; vec4 shape; vec4 glyph; vec2 target; vec4 rot; } p;
layout(location = 0) out vec2 uv;
void main() {
    vec2 c = vec2((gl_VertexIndex == 1 || gl_VertexIndex == 2 || gl_VertexIndex == 4) ? 1.0 : 0.0, (gl_VertexIndex == 2 || gl_VertexIndex == 4 || gl_VertexIndex == 5) ? 1.0 : 0.0);
    vec2 px = p.rect.xy + c * p.rect.zw;
    uv = c * 2.0 - 1.0;
    gl_Position = vec4(mat2(p.rot.x, p.rot.y, -p.rot.y, p.rot.x) * (px / p.target * 2.0 - 1.0), 0.0, 1.0);
}
