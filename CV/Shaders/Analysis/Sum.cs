/*{
  "DESCRIPTION": "Sum of image luminance reduced on the GPU into an SSBO, read back to the host via BufferToArray as [sum, count, sumHi]. Covers BOTH cv.jit.sum and cv.jit.mass -- they are different quantities and the SSBO deliberately carries the raw one so the host can derive either without ambiguity.\n\nWIRE VALUE: the SSBO holds the RAW 64-bit luminance total in cv.jit's 0..255 char units, S = sum + sumHi*2^32, where each pixel contributes round(luma*255). Nothing is divided on the GPU.\n\nDERIVING THE TWO cv.jit OBJECTS:\n  cv.jit.sum  (raw 0..255-unit total, char input)  =  S           <-- NO division\n  cv.jit.mass (cv.jit divides the char sum by 255) =  S / 255.0\nThis shader previously divided by 255 in its own documentation, i.e. it described only cv.jit.mass while being named after cv.jit.sum; one of the two objects was therefore always wrong. Divide (or not) on the host according to which object you are reproducing. 'count' (index 1) is the number of accumulated pixels, useful for a mean = S / (255*count).\n\nOVERFLOW: the total is a 64-bit (sumHi:sum) pair so it never overflows uint32 even on 8K/tiled/supersampled frames (a bare uint32 'sum' would wrap above ~16.8M px). The wire layout is UNCHANGED by this documentation fix: sum at word 0, count at 1, sumHi at 2.\n\nPipeline: <image> -> this(inputTex) -> [buf outlet] -> BufferToArray -> value. The output image is a luma passthrough so the node is also visually inspectable.",
  "CREDIT": "ossia score",
  "ISFVSN": "2.0",
  "MODE": "COMPUTE_SHADER",
  "CATEGORIES": ["Computer Vision", "Analysis"],
  "RESOURCES": [
    { "NAME": "inputTex", "TYPE": "texture" },
    {
      "NAME": "outputImage",
      "TYPE": "image",
      "ACCESS": "write_only",
      "FORMAT": "rgba8",
      "WIDTH": "$WIDTH_inputTex",
      "HEIGHT": "$HEIGHT_inputTex"
    },
    {
      "NAME": "buf",
      "TYPE": "storage",
      "ACCESS": "read_write",
      "LAYOUT": [
        { "NAME": "sum",   "TYPE": "uint" },
        { "NAME": "count", "TYPE": "uint" },
        { "NAME": "sumHi", "TYPE": "uint" }
      ]
    }
  ],
  "PASSES": [
    { "LOCAL_SIZE": [1, 1, 1],   "EXECUTION_MODEL": { "TYPE": "MANUAL", "WORKGROUPS": [1, 1, 1] } },
    { "LOCAL_SIZE": [16, 16, 1], "EXECUTION_MODEL": { "TYPE": "2D_IMAGE" } }
  ]
}*/

// The SSBO persists across frames, so pass 0 (a single invocation) clears the accumulator
// before pass 1 (one invocation per output pixel) atomically accumulates luminance into it.
// Luminance in [0,1] is scaled to a fixed-point integer so atomicAdd (uint) can be used.
//
// OVERFLOW (64-bit hi/lo accumulation):
//   Each pixel adds up to LUMA_SCALE (255). A bare uint32 'sum' overflows at
//   255 * Npx > 2^32, i.e. above ~16.8M px (fails at 8K / tiled / supersampled).
//   We therefore carry a 64-bit total as a (sumHi:sum) pair. We atomicAdd into the low
//   word and capture the PREVIOUS value it returns; if the add wrapped past 2^32
//   (old + add < old, detected via the returned previous low word) we atomicAdd 1 into
//   sumHi. The host reconstructs total = sumHi * 2^32 + sum. With the hi word the safe
//   ceiling is effectively unbounded for any realistic frame (>10^21 px).
//   BACK-COMPAT: 'sum' (low) stays at word index 0 and 'count' at index 1, so a host
//   that only reads the low word still decodes correctly below the 16.8M-px ceiling.

// LUMA_SCALE is cv.jit's char scale. Each pixel contributes round(luma * 255), so the
// (sumHi:sum) total is exactly cv.jit.sum's value for an 8-bit input. cv.jit.mass is the
// SAME accumulation divided by 255 -- that division happens on the HOST, not here, so both
// objects are derivable from one buffer. Do not "simplify" by dividing on the GPU: that
// would silently turn this node into cv.jit.mass only, which is the bug this comment
// (and the DESCRIPTION above) exists to prevent recurring.
const float LUMA_SCALE = 255.0;

void main()
{
    if(PASSINDEX == 0)
    {
        buf.sum = 0u;
        buf.count = 0u;
        buf.sumHi = 0u;
        return;
    }

    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outputImage);
    if(pos.x >= size.x || pos.y >= size.y)
        return;

    vec2 uv = (vec2(pos) + 0.5) / vec2(size);
    vec3 rgb = texture(inputTex, uv).rgb;
    float luma = dot(rgb, vec3(0.299, 0.587, 0.114));

    uint add = uint(luma * LUMA_SCALE + 0.5);
    uint prev = atomicAdd(buf.sum, add);
    // Carry detection: unsigned wrap iff (prev + add) wrapped past 2^32, i.e. result < prev.
    if(prev + add < prev)
        atomicAdd(buf.sumHi, 1u);
    atomicAdd(buf.count, 1u);

    imageStore(outputImage, pos, vec4(vec3(luma), 1.0));
}
