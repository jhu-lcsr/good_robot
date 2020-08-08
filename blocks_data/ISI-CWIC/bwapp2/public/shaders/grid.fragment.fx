precision highp float;

varying vec2 vUV;

uniform sampler2D textureSampler;

void main(void) {
  float divisions = 10.0;
  float thickness = 0.01;

  float x = step(fract(vUV.x / (1.0 / divisions)), thickness);
  float y = step(fract(vUV.y / (1.0 / divisions)), thickness);
  float c = x + y;

  if(c < 0.5) discard;

  gl_FragColor = vec4(c, c, c, 1.0);
}
