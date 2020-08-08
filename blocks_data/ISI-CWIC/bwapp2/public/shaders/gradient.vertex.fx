precision mediump float;

// Attributes
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;

// Uniforms
uniform mat4 worldViewProjection;

// Varying
varying vec4 vPosition;
varying vec3 vNormal;

void main() {

    vec4 p = vec4( position, 1. );
    vPosition = p;
    vNormal = normal;
    gl_Position = worldViewProjection * p;

}