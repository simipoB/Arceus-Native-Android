#version 450
// FXAA inside the game's frame (nvn_vk.c fx_scene): one triangle over the whole scene image
layout(location = 0) out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
