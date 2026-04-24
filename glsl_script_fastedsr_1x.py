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
                    is_last_chunk, prev_in_start, when, is_output_layer):
    prev_chunk_name = f"conv_{name}_p{out_group}_c{prev_in_start}"
    if is_output_layer:
        save_name = "MAIN"
    else:
        save_name = f"conv_{name}_p{out_group}" if is_last_chunk else f"conv_{name}_p{out_group}_c{in_start}"

    code = f"""//!DESC {name} Conv3x3 part {out_group} chunk {in_start}-{in_end}
//!HOOK MAIN\n"""

    if is_output_layer:
        code += f"//!BIND MAIN\n"
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
//!COMPONENTS 4\n"""

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

    if is_output_layer:
        code += f"""

        vec3 res = result.rgb;
        vec3 final_output = MAIN_tex(MAIN_pos).rgb + res;
    
        return vec4(clamp(final_output, 0.0, 1.0), 1.0);
        }}
    """
    else:
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
#define get_0(x_off, y_off) MAIN_texOff(vec2(x_off, y_off))
vec4 hook() {{
    vec4 result = vec4(0.0);
"""
    code += get_conv_single_group_lines(W, out_group, 0)
    code += get_vec4_bias(b, out_group) + "\n"
    code += "    return result;\n}\n"
    return code


def get_conv3x3(W, name, prev_name, relu=False, skip_name=None, chunk_size=8, when=None, is_output_layer=False):
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
                                        is_first, is_last, prev_in_start, when, is_output_layer)

    return code


##############################################

if __name__ == "__main__":
    checkpoint_path = Path("checkpoints/SR_FastEDSR_1x_jpeg_4_64_s.pth")

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
    code += get_conv3x3(W, name=f"net.{nb + 1}", prev_name=last_layer_name, is_output_layer=True)

    output_path.write_text(code)
    output_path_2.write_text(code)
    output_path_3.write_text(code)

    exit()

    subprocess.run(["powershell", "-Command",
                    'rm C:\\Users\\User\\AppData\\Local\\mpv\\cache\\*; mpv.exe --msg-level=vo=debug,gpu=debug C:\\Users\\User\\PycharmProjects\\Pytorch\\super-resolution-cnn\\videoinput\\F1Bahr-480p50.mp4 '])
