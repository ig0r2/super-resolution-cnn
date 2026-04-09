"""
Converts a FastEDSR model into a series of GLSL shaders
that can be used with mpv's --glsl-shader option for real-time upscaling.


mpv passes images through shaders one pixel at a time. Because GLSL has no
concept of convolution, we unroll every kernel tap manually and express the
4-channel groups as mat4 × vec4 multiplications.

We use texture (fragment) shaders that have 4 channels - RGBA.
That means that output channels of convolution have to be split into multiple shaders (4 channels per shader)
We will call those 4 channels - output group / part
So a 32 channel layer produces 8 shaders/textures (parts).
When the number of input groups is too large to fit in a single shader (GPU texture-unit limit),
the input is split into "chunks" and the partial sums are accumulated by each next chunk.



//!DESC - shader description - mpv uses it as a shader name in stats
//!HOOK MAIN - run this shader during the MAIN rendering stage, using the MAIN frame as the coordinate reference
//!BIND [texture] - binds texture so shader can access it
//!SAVE [texture] - texture name that this shader will save to
//!WIDTH //!HEIGHT - width and height of the texture
//!COMPONENTS 4 - shader will be RGBA
//!WHEN OUTPUT.w MAIN.w / 1.2 > OUTPUT.h MAIN.h / 1.2 > * - shader will be used only when OUTPUT image will be 1.2x MAIN image
       (OUTPUT.w/MAIN.w > 1.2) AND (OUTPUT.h/MAIN.h > 1.2)            * is AND
"""

import subprocess
from math import ceil
from pathlib import Path
import numpy as np

from utils.checkpoints import load_model_from_checkpoint


def print_model_layers(model):
    print("Model layers")
    for name, module in model.named_modules():
        print(f"{name:30} | {module.__class__.__name__:15}")
    print("\nModel parameters")
    for name, param in model.named_parameters():
        print(f"{name:40} |  shape: {tuple(param.shape)} | params: {param.numel()}")


def get_weights_from_model(model):
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


"""
Convolution 4x4x3x3
each pixel computes 4 outputs from 4 input channels using 9 surrounding pixels

#define go_0(x_off, y_off) [texture]_texOff(vec2(x_off, y_off))
        [texture]_texOff(vec2(x_off, y_off)) - read pixel at offset from current pixel


======================================
Convolution for a single output group (texture) and input group (texture)
output group - 4 channels
input group - 4 channels
kernel - 3x3 pixels


torch weights: [out_channels, in_channels, kernel_y, kernel_x] = 4x4x3x3
w00 = weights[:, :, 0, 0] = 4x4 matrix for offset (-1, -1)
w01 = weights[:, :, 0, 1] = 4x4 matrix for offset (-1, 0)
w01 = weights[:, :, 0, 2] = 4x4 matrix for offset (-1, 1)
....
pixel indexes within       pixel offsets from center
kernel (for weights)       used for <tex>_texOff(x_off, y_off)       
(0,0) (0,1) (0,2)          (-1,-1)  (-1,0)  (-1,1)
(1,0) (1,1) (1,2)    <->   (0,-1)   (0,0)   (0,1)
(2,0) (2,1) (2,2)          (1,-1)   (1,0)   (1,1)
######################################################
in_pixel is vec4 of 4 channel output from corresponding texture
##############
vec4 result = (0.0, 0.0, 0.0, 0.0)
result += mat4(w00) * in_pixel(-1,-1);
result += mat4(w01) * in_pixel(-1,0);
result += mat4(w02) * in_pixel(-1,1);

result += mat4(w10) * in_pixel(0,-1);
result += mat4(w11) * in_pixel(0,0);
result += mat4(w12) * in_pixel(0,1);

result += mat4(w20) * in_pixel(1,-1);
result += mat4(w21) * in_pixel(1,0);
result += mat4(w22) * in_pixel(1,1);

result += vec4(bias);
############################################

This will be repeated for each input group in that chunk
"""


def get_mat4_weights(W, out_group, in_group, x, y):
    # output_ch, input_ch, kernel_h, kernel_w = W.shape
    mat = W[4 * out_group:4 * out_group + 4, 4 * in_group:4 * in_group + 4, y, x]

    if len(mat) < 16:  # pad to 4x4
        padded = np.zeros((4, 4), dtype=mat.dtype)
        padded[:mat.shape[0], :mat.shape[1]] = mat
        mat = padded

    # in GLSL mat4 is column-major so transpose is needed
    weights_str = ", ".join([f"{w:.8f}" for w in mat.T.flatten()])

    return f"mat4({weights_str})"


def get_vec4_bias(b, out_group):
    bias = b[4 * out_group: 4 * out_group + 4]
    while len(bias) < 4:  # pad to 4
        bias = np.append(bias, 0.0)
    bias_str = ", ".join([f"{val:.8f}" for val in bias])
    return f"    result += vec4({bias_str});"


def get_conv_line(W, out_group, in_group, x, y):
    return f"    result += {get_mat4_weights(W, out_group, in_group, x, y)} * get_{in_group}({x - 1},{y - 1});"


# For single output group (4 channels) and input group (4 channels) => (9 pixels)
def get_conv_single_group_lines(W, out_group, in_group):
    code = ""
    for x in range(3):
        for y in range(3):
            code += get_conv_line(W, out_group, in_group, x, y) + "\n"
    return code


# Shader for selected output group of conv and a selected chunk of inputs
def get_conv_shader(W, b, out_group, name, prev_name, relu, skip_name, in_start, in_end, is_first_chunk,
                    is_last_chunk, prev_in_start, when):
    prev_chunk_name = f"conv_{name}_p{out_group}_c{prev_in_start}"
    save_name = f"conv_{name}_p{out_group}" if is_last_chunk else f"conv_{name}_p{out_group}_c{in_start}"

    code = f"""//!DESC {name} Conv3x3 part {out_group} chunk {in_start}-{in_end}
//!HOOK MAIN\n"""

    # bind previous inputs (textures)
    for i in range(in_start, in_end):
        code += f"//!BIND conv_{prev_name}_p{i}\n"
    # bind skip connection textures if last chunk
    if skip_name is not None and is_last_chunk:
        code += f"//!BIND conv_{skip_name}_p{out_group}\n"
    # bind previous chunk
    if not is_first_chunk:
        code += f"//!BIND {prev_chunk_name}\n"

    code += f"""//!SAVE {save_name}
//!WIDTH conv_{prev_name}_p{in_start}.w
//!HEIGHT conv_{prev_name}_p{in_start}.h
//!COMPONENTS 4
//!WHEN {when}\n"""

    # define functions for inputs (textures)
    for i in range(in_start, in_end):
        code += f"#define get_{i}(x_off, y_off) conv_{prev_name}_p{i}_texOff(vec2(x_off, y_off))\n"

    code += "vec4 hook() {\n"
    code += "    vec4 result = vec4(0.0);\n"

    # convolution for each input group (texture)
    for i in range(in_start, in_end):
        code += get_conv_single_group_lines(W, out_group, i)

    # Add result from previous chunk
    if not is_first_chunk:
        code += f"    result += {prev_chunk_name}_texOff(vec2(0.0, 0.0));\n"

    if is_last_chunk:
        # Add bias for this output group
        code += get_vec4_bias(b, out_group) + "\n"
        # Add skip connection
        if skip_name is not None:
            code += f"    result += conv_{skip_name}_p{out_group}_texOff(vec2(0.0, 0.0));\n"
        # Add relu
        if relu: code += "    result = max(result, 0.0);\n"

    code += "    return result;\n}\n"
    return code


# First conv - reads from MAIN
def get_conv_shader_MAIN(W, b, out_group, name, when):
    code = f"""//!DESC {name} Conv3x3 part {out_group}
//!HOOK MAIN
//!BIND MAIN
//!SAVE conv_{name}_p{out_group}
//!WIDTH MAIN.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
//!WHEN {when}
#define get_0(x_off, y_off) MAIN_texOff(vec2(x_off, y_off))
vec4 hook() {{
    vec4 result = vec4(0.0);
"""
    code += get_conv_single_group_lines(W, out_group, 0)
    code += get_vec4_bias(b, out_group) + "\n"
    code += "    return result;\n}\n"
    return code


def get_conv3x3(W, name, prev_name, relu=False, skip_name=None, chunk_size=8, when=None):
    weight = W[f'{name}.weight']
    bias = W[f'{name}.bias']

    name = name.replace(".", "_")
    if prev_name is not None:
        prev_name = prev_name.replace(".", "_")
    if skip_name is not None:
        skip_name = skip_name.replace(".", "_")

    output_ch, input_ch, kernel_h, kernel_w = weight.shape

    num_shaders = int(ceil(output_ch / 4))  # num_ouput_groups

    if when is None:
        when = "OUTPUT.w MAIN.w / 1.2 > OUTPUT.h MAIN.h / 1.2 > *"

    code = ""
    for i in range(num_shaders):
        if prev_name is None:
            code += get_conv_shader_MAIN(weight, bias, i, name, when)
        else:
            num_in_groups = int(ceil(input_ch / 4))
            num_chunks = int(ceil(num_in_groups / chunk_size))

            for chunk_idx in range(num_chunks):
                in_start = chunk_idx * chunk_size
                in_end = min(in_start + chunk_size, num_in_groups)
                is_first = chunk_idx == 0
                is_last = chunk_idx == num_chunks - 1
                prev_in_start = (chunk_idx - 1) * chunk_size

                code += get_conv_shader(weight, bias, i, name, prev_name, relu, skip_name, in_start, in_end,
                                        is_first, is_last, prev_in_start, when)

    return code


"""
############################################
            Pixel shuffle
#############################################

Convolution before pixel shuffle always outputs 12 channels for 2x upscale
first 4 channels (1st texture) contain 4 pixels for R channel for 2x2 grid that goes instead of a single LR pixel
next 4 channels (2nd texture) contain 4 pixels for G channel for 2x2 grid
last 4 channels (3rd texure) contain 4 pixels for B channel for 2x2 grid

1 shader hooks onto single output pixel
First we get position of that pixel within upscaled image
        ivec2 pos = ivec2(gl_FragCoord.xy);         glFlagCoord returns coords where (0,0) is bottom-left
        
Then we need to determine which pixel is that within 2x2 grid - vec2 sub
        ivec2 sub = ivec2(pos) % 2;

If top left (0,0) then we get channel 0 from each texture
If top right (0,1) then we get channel 1 from each texture
If bottom left (1,0) then we get channel 2 from each texture
If bottom right (1,1) we get channel 3 from each texture
    =>      int idx = sub.x + (sub.y * 2);
    
We need to calculate position within LR image (texture)
        source_pos = floor(pos / 2.0) + 0.5
and normalize to [0-1]
        tex_coord = source_pos / conv_{prev_name}_p0_size
        
We get our result               vec3 res = vec3(p0[idx], p1[idx], p2[idx]);
and add to upscaled image       vec3 final_output = MAIN_tex(MAIN_pos).rgb + res;
        
############################################
"""


def get_pixel_shuffle_x2(prev_name, when):
    prev_name = prev_name.replace(".", "_")

    code = f"""//!DESC PixelShuffle x2
//!HOOK MAIN
//!BIND MAIN
{"\n".join(f"//!BIND conv_{prev_name}_p{i}" for i in range(3))}
//!SAVE MAIN
//!WIDTH conv_{prev_name}_p0.w 2 *
//!HEIGHT conv_{prev_name}_p0.h 2 *
//!COMPONENTS 4
//!WHEN {when}
vec4 hook() {{
    ivec2 pos = ivec2(gl_FragCoord.xy);

    ivec2 sub = ivec2(pos) & 1;
    int idx = sub.x + (sub.y << 1);

    vec2 source_pos = floor(pos / 2.0) + 0.5;
    vec2 tex_coord = source_pos / conv_{prev_name}_p0_size;

    vec4 p0 = conv_{prev_name}_p0_tex(tex_coord);
    vec4 p1 = conv_{prev_name}_p1_tex(tex_coord);
    vec4 p2 = conv_{prev_name}_p2_tex(tex_coord);

    vec3 res = vec3(p0[idx], p1[idx], p2[idx]);
    vec3 final_output = MAIN_tex(MAIN_pos).rgb + res;

    return vec4(clamp(final_output, 0.0, 1.0), 1.0);
}}
"""
    return code


def get_pixel_shuffle_x3(prev_name, when):
    prev_name = prev_name.replace(".", "_")

    # 3x3 * 3 channels = 27 output channels => ceil(27/4) = 7 textures (p0..p6)

    code = f"""//!DESC PixelShuffle x3
//!HOOK MAIN
//!BIND MAIN
{"\n".join(f"//!BIND conv_{prev_name}_p{i}" for i in range(7))}
//!SAVE MAIN
//!WIDTH conv_{prev_name}_p0.w 3 *
//!HEIGHT conv_{prev_name}_p0.h 3 *
//!COMPONENTS 4
//!WHEN {when}
vec4 hook() {{
    ivec2 pos = ivec2(gl_FragCoord.xy);

    ivec2 sub = ivec2(pos) % 3;
    int idx = sub.x + (sub.y * 3);  // 0..8

    vec2 source_pos = floor(pos / 3.0) + 0.5;
    vec2 tex_coord = source_pos / conv_{prev_name}_p0_size;

    // treat all 27 channels as a flat array across textures
    // and just index by (channel / 4) for texture, (channel % 4) for element

    vec4 p0 = conv_{prev_name}_p0_tex(tex_coord);
    vec4 p1 = conv_{prev_name}_p1_tex(tex_coord);
    vec4 p2 = conv_{prev_name}_p2_tex(tex_coord);
    vec4 p3 = conv_{prev_name}_p3_tex(tex_coord);
    vec4 p4 = conv_{prev_name}_p4_tex(tex_coord);
    vec4 p5 = conv_{prev_name}_p5_tex(tex_coord);
    vec4 p6 = conv_{prev_name}_p6_tex(tex_coord);

    // Flat channel index for each color: R=idx, G=idx+9, B=idx+18
    // texture = flat_ch / 4,  element = flat_ch % 4
    float r = (idx < 4) ? p0[idx] : (idx < 8) ? p1[idx-4] : p2[0];

    int g_ch = idx + 9;
    float g = (g_ch < 12) ? p2[g_ch-8] : (g_ch < 16) ? p3[g_ch-12] : (g_ch < 20) ? p4[g_ch-16] : p5[g_ch-20];

    int b_ch = idx + 18;
    float b = (b_ch < 20) ? p4[b_ch-16] : (b_ch < 24) ? p5[b_ch-20] : (b_ch < 28) ? p6[b_ch-24] : 0.0;

    vec3 res = vec3(r, g, b);
    vec3 final_output = MAIN_tex(MAIN_pos).rgb + res;

    return vec4(clamp(final_output, 0.0, 1.0), 1.0);
}}
"""
    return code


def get_pixel_shuffle_x4(prev_name, when):
    prev_name = prev_name.replace(".", "_")

    # 4x4 * 3 channels = 48 output channels => ceil(48/4) = 12 textures (p0..p11)

    code = f"""//!DESC PixelShuffle x4
//!HOOK MAIN
//!BIND MAIN
{"\n".join(f"//!BIND conv_{prev_name}_p{i}" for i in range(12))}
//!SAVE MAIN
//!WIDTH conv_{prev_name}_p0.w 4 *
//!HEIGHT conv_{prev_name}_p0.h 4 *
//!COMPONENTS 4
//!WHEN {when}
vec4 hook() {{
    ivec2 pos = ivec2(gl_FragCoord.xy);

    ivec2 sub = ivec2(pos) % 4;
    int idx = sub.x + (sub.y * 4);  // 0..15

    vec2 source_pos = floor(pos / 4.0) + 0.5;
    vec2 tex_coord = source_pos / conv_{prev_name}_p0_size;

    vec4 p0  = conv_{prev_name}_p0_tex(tex_coord);
    vec4 p1  = conv_{prev_name}_p1_tex(tex_coord);
    vec4 p2  = conv_{prev_name}_p2_tex(tex_coord);
    vec4 p3  = conv_{prev_name}_p3_tex(tex_coord);
    vec4 p4  = conv_{prev_name}_p4_tex(tex_coord);
    vec4 p5  = conv_{prev_name}_p5_tex(tex_coord);
    vec4 p6  = conv_{prev_name}_p6_tex(tex_coord);
    vec4 p7  = conv_{prev_name}_p7_tex(tex_coord);
    vec4 p8  = conv_{prev_name}_p8_tex(tex_coord);
    vec4 p9  = conv_{prev_name}_p9_tex(tex_coord);
    vec4 p10 = conv_{prev_name}_p10_tex(tex_coord);
    vec4 p11 = conv_{prev_name}_p11_tex(tex_coord);

    // All 48 channels flat: R=idx (0-15), G=idx+16 (16-31), B=idx+32 (32-47)
    // texture = flat_ch / 4,  element = flat_ch % 4
    // R: idx 0-15 -> p0-p3
    float r = (idx < 4)  ? p0[idx]    : (idx < 8)  ? p1[idx-4]
            : (idx < 12) ? p2[idx-8]  : p3[idx-12];

    // G: idx+16, range 16-31 -> p4-p7
    int g_ch = idx + 16;
    float g = (g_ch < 20) ? p4[g_ch-16] : (g_ch < 24) ? p5[g_ch-20]
            : (g_ch < 28) ? p6[g_ch-24] : p7[g_ch-28];

    // B: idx+32, range 32-47 -> p8-p11
    int b_ch = idx + 32;
    float b = (b_ch < 36) ? p8[b_ch-32]  : (b_ch < 40) ? p9[b_ch-36]
            : (b_ch < 44) ? p10[b_ch-40] : p11[b_ch-44];

    vec3 res = vec3(r, g, b);
    vec3 final_output = MAIN_tex(MAIN_pos).rgb + res;

    return vec4(clamp(final_output, 0.0, 1.0), 1.0);
}}
"""
    return code


# Get upscale block for 2x scale (shuffle conv + pixel shuffle)
def get_upscale_block_x2(W, prev_name):
    final_conv_name = f"upscale_block_2.0"
    when = "OUTPUT.w MAIN.w / 1.2 > OUTPUT.h MAIN.h / 1.2 > * OUTPUT.w MAIN.w / 2.2 < OUTPUT.h MAIN.h / 2.2 < * *"
    return (get_conv3x3(W, name=final_conv_name, prev_name=prev_name, when=when) +
            get_pixel_shuffle_x2(prev_name=final_conv_name, when=when))


def get_upscale_block_x3(W, prev_name):
    final_conv_name = f"upscale_block_3.0"
    when = "OUTPUT.w MAIN.w / 2.2 >= OUTPUT.h MAIN.h / 2.2 >= * OUTPUT.w MAIN.w / 3.2 < OUTPUT.h MAIN.h / 3.2 < * *"
    return (get_conv3x3(W, name=final_conv_name, prev_name=prev_name, when=when) +
            get_pixel_shuffle_x3(prev_name=final_conv_name, when=when))


def get_upscale_block_x4(W, prev_name):
    final_conv_name = f"upscale_block_4.0"
    when = "OUTPUT.w MAIN.w / 3.2 >= OUTPUT.h MAIN.h / 3.2 >= *"
    return (get_conv3x3(W, name=final_conv_name, prev_name=prev_name, when=when) +
            get_pixel_shuffle_x4(prev_name=final_conv_name, when=when))


##############################################

if __name__ == "__main__":
    checkpoint_path = Path("checkpoints/SR_FastEDSR_jpeg_4_64.pth")

    output_path_2 = Path(f"exports/glsl/{checkpoint_path.stem}.glsl")
    output_path_3 = Path(f"C:/Users/User/Tools/mpv/shaders/{checkpoint_path.stem}.glsl")
    output_path = Path("C:/Users/User/Tools/mpv/shaders/current.glsl")

    output_path_2.parent.mkdir(parents=True, exist_ok=True)

    model, model_config = load_model_from_checkpoint(checkpoint_path, "cpu")

    print_model_layers(model)
    W = get_weights_from_model(model)

    code = ""

    # First conv layer
    code += get_conv3x3(W, name="net.0", prev_name=None)

    nb = model_config['params']['num_blocks']
    last_layer_name = "net.0"  # Keep track of the actual name to link to

    # Loop through the blocks
    for i in range(1, nb + 1):
        layer0 = f"net.{i}.layers.0"
        layer2 = f"net.{i}.layers.2"

        code += get_conv3x3(W, name=layer0, prev_name=last_layer_name, relu=True)
        code += get_conv3x3(W, name=layer2, prev_name=layer0, skip_name=last_layer_name)
        # Update the tail pointer
        last_layer_name = layer2

    # Upscale blocks
    code += get_upscale_block_x2(W, prev_name=last_layer_name)
    code += get_upscale_block_x3(W, prev_name=last_layer_name)
    code += get_upscale_block_x4(W, prev_name=last_layer_name)

    output_path.write_text(code)
    output_path_2.write_text(code)
    output_path_3.write_text(code)

    # exit()

    subprocess.run(["powershell", "-Command",
                    'rm C:\\Users\\User\\AppData\\Local\\mpv\\cache\\*; mpv.exe --msg-level=vo=debug,gpu=debug C:\\Users\\User\\PycharmProjects\\Pytorch\\super-resolution-cnn\\videoinput\\F1Bahr-240p50.mp4 '])
