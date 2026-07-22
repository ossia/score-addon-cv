/*{
  "DESCRIPTION": "Binarize an image. cv.jit.threshold equivalent (adaptive mean-C thresholding on luminance), plus an extra Global fixed-level mode that cv.jit does not have. POLARITY: cv.jit.threshold calls OpenCV adaptiveThreshold(..., ADAPTIVE_THRESH_MEAN_C, mode, 1+2*radius, C) where its 'mode' attribute selects THRESH_BINARY_INV when 0 (its DEFAULT) and THRESH_BINARY when 1. So cv.jit's out-of-the-box behaviour is INVERTED: white (255) where src <= localmean - C, black where src > localmean - C. The 'invert' toggle here IS cv.jit's 'mode': invert=false (default) == cv.jit mode 0 == THRESH_BINARY_INV; invert=true == cv.jit mode 1 == THRESH_BINARY. Before this fix the adaptive branch produced THRESH_BINARY at invert=false, so every patch ported 1:1 from cv.jit came out negated. MAPPING: cv.jit's 'threshold' attribute (default 5) is on the 0..255 char scale; 'C' here is normalised, so C = cv.jit_threshold / 255 (the 0.02 default is cv.jit's 5/255 = 0.0196). The range is now [-1, 1] so the full cv.jit [-255, 255] span is reachable; it used to be [-0.5, 0.5], reaching only about half of it. cv.jit's 'radius' (default 1) maps to 'radius' here: the neighbourhood is the (2*radius+1)^2 box that OpenCV calls blockSize = 1 + 2*radius. NOTE this port still defaults radius to 4, not cv.jit's 1 -- set radius to 1 for a 1:1 port. MODE: mode 1 (Adaptive) is the cv.jit-equivalent path and is now the DEFAULT; mode 0 (Global, fixed 'level') is an ossia-only extra with no cv.jit counterpart. The numbering is unchanged so existing patches that set mode explicitly keep their meaning. Global mode deliberately keeps the conventional non-inverted sense (white where luma >= level) and 'invert' flips it; only the adaptive branch follows cv.jit's inverted convention.",
  "CREDIT": "ossia score",
  "ISFVSN": "2",
  "CATEGORIES": ["Computer Vision", "Image Processing"],
  "INPUTS": [
    { "NAME": "inputImage", "TYPE": "image" },
    {
      "NAME": "mode",
      "TYPE": "long",
      "DEFAULT": 1,
      "VALUES": [0, 1],
      "LABELS": ["Global (fixed level)", "Adaptive mean - C (cv.jit)"]
    },
    { "NAME": "level",  "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0, "MAX": 1.0 },
    { "NAME": "radius", "TYPE": "float", "DEFAULT": 4.0,  "MIN": 1.0, "MAX": 32.0 },
    { "NAME": "C",      "TYPE": "float", "DEFAULT": 0.02, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "invert", "TYPE": "bool",  "DEFAULT": false }
  ]
}*/

float luminance(vec4 c)
{
  return dot(c.rgb, vec3(0.299, 0.587, 0.114));
}

void main()
{
  vec2 uv = isf_FragNormCoord;
  vec2 texel = vec2(1.0) / RENDERSIZE;

  float v = luminance(IMG_NORM_PIXEL(inputImage, uv));
  float bin;

  if(mode == 0)
  {
    // Global fixed level (no cv.jit counterpart). Conventional sense: white above level.
    bin = step(level, v);
  }
  else
  {
    // Local mean over the (2*radius+1)^2 box == OpenCV blockSize 1+2*radius, then - C.
    int r = int(radius);
    float sum = 0.0;
    float n = 0.0;
    for(int dy = -r; dy <= r; ++dy)
    {
      for(int dx = -r; dx <= r; ++dx)
      {
        sum += luminance(IMG_NORM_PIXEL(inputImage, uv + vec2(float(dx), float(dy)) * texel));
        n += 1.0;
      }
    }
    float thr = (sum / n) - C;

    // cv.jit's default is THRESH_BINARY_INV: white where src <= thr, black where src > thr.
    // step(thr, v) is 1 when v >= thr (THRESH_BINARY), so take its complement here.
    bin = 1.0 - step(thr, v);
  }

  // `invert` == cv.jit.threshold's `mode` attribute: false (0) keeps the base polarity
  // above, true (1) selects the opposite one (THRESH_BINARY for the adaptive branch).
  if(invert)
    bin = 1.0 - bin;

  gl_FragColor = vec4(vec3(bin), 1.0);
}
