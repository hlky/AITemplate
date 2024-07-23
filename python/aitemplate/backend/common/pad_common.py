import jinja2

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

template <typename T>
__global__ void pad_kernel_constant_1d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} pad_left,
    {{index_type}} pad_right,
    T pad_value
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N + pad_left + pad_right;
    if (idx < total_elements) {
        if (idx < pad_left || idx >= N + pad_left) {
            output[idx] = pad_value;
        } else {
            output[idx] = input[idx - pad_left];
        }
    }
}

template <typename T>
__global__ void pad_kernel_reflect_1d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N + pad_left + pad_right;
    if (idx < total_elements) {
        if (idx < pad_left) {
            output[idx] = input[pad_left - idx - 1];
        } else if (idx >= N + pad_left) {
            output[idx] = input[2 * N + pad_left - idx - 1];
        } else {
            output[idx] = input[idx - pad_left];
        }
    }
}

template <typename T>
__global__ void pad_kernel_replicate_1d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N + pad_left + pad_right;
    if (idx < total_elements) {
        if (idx < pad_left) {
            output[idx] = input[0];
        } else if (idx >= N + pad_left) {
            output[idx] = input[N - 1];
        } else {
            output[idx] = input[idx - pad_left];
        }
    }
}

template <typename T>
__global__ void pad_kernel_circular_1d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N + pad_left + pad_right;
    if (idx < total_elements) {
        if (idx < pad_left) {
            output[idx] = input[(idx - pad_left + N) % N];
        } else if (idx >= N + pad_left) {
            output[idx] = input[(idx - pad_left) % N];
        } else {
            output[idx] = input[idx - pad_left];
        }
    }
}


template <typename T>
__global__ void pad_kernel_constant_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} pad_left,
    {{index_type}} pad_right,
    T pad_value
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_left + pad_right);
    if (idx < total_elements) {
        {{index_type}} h = idx % (H + pad_left + pad_right);
        {{index_type}} n = idx / (H + pad_left + pad_right);

        if (h < pad_left || h >= H + pad_left) {
            output[idx] = pad_value;
        } else {
            {{index_type}} in_h = h - pad_left;
            {{index_type}} in_index = n * H + in_h;
            output[idx] = input[in_index];
        }
    }
}

template <typename T>
__global__ void pad_kernel_reflect_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_left + pad_right);
    if (idx < total_elements) {
        {{index_type}} h = idx % (H + pad_left + pad_right);
        {{index_type}} n = idx / (H + pad_left + pad_right);

        {{index_type}} in_h = h < pad_left ? pad_left - h : (h >= H + pad_left ? 2 * H + pad_left - h - 2 : h - pad_left);
        {{index_type}} in_index = n * H + in_h;
        output[idx] = input[in_index];
    }
}

template <typename T>
__global__ void pad_kernel_replicate_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_left + pad_right);
    if (idx < total_elements) {
        {{index_type}} h = idx % (H + pad_left + pad_right);
        {{index_type}} n = idx / (H + pad_left + pad_right);

        {{index_type}} in_h = std::min<{{index_type}}>(std::max<{{index_type}}>(h - pad_left, 0), H - 1);
        {{index_type}} in_index = n * H + in_h;
        output[idx] = input[in_index];
    }
}

template <typename T>
__global__ void pad_kernel_circular_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_left + pad_right);
    if (idx < total_elements) {
        {{index_type}} h = idx % (H + pad_left + pad_right);
        {{index_type}} n = idx / (H + pad_left + pad_right);

        {{index_type}} in_h = (h - pad_left + H) % H;
        {{index_type}} in_index = n * H + in_h;
        output[idx] = input[in_index];
    }
}


template <typename T>
__global__ void pad_kernel_constant_3d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right,
    T pad_value
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
    if (idx < total_elements) {
        {{index_type}} w = idx % (W + pad_left + pad_right);
        {{index_type}} h = (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
        {{index_type}} n = idx / ((H + pad_top + pad_bottom) * (W + pad_left + pad_right));

        if (h < pad_top || h >= H + pad_top || w < pad_left || w >= W + pad_left) {
            output[idx] = pad_value;
        } else {
            {{index_type}} in_h = h - pad_top;
            {{index_type}} in_w = w - pad_left;
            {{index_type}} in_index = n * H * W + in_h * W + in_w;
            output[idx] = input[in_index];
        }
    }
}

template <typename T>
__global__ void pad_kernel_reflect_3d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
    if (idx < total_elements) {
        {{index_type}} w = idx % (W + pad_left + pad_right);
        {{index_type}} h = (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
        {{index_type}} n = idx / ((H + pad_top + pad_bottom) * (W + pad_left + pad_right));

        {{index_type}} in_h = h < pad_top ? pad_top - h : (h >= H + pad_top ? 2 * H + pad_top - h - 2 : h - pad_top);
        {{index_type}} in_w = w < pad_left ? pad_left - w : (w >= W + pad_left ? 2 * W + pad_left - w - 2 : w - pad_left);
        {{index_type}} in_index = n * H * W + in_h * W + in_w;
        output[idx] = input[in_index];
    }
}

template <typename T>
__global__ void pad_kernel_replicate_3d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
    if (idx < total_elements) {
        {{index_type}} w = idx % (W + pad_left + pad_right);
        {{index_type}} h = (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
        {{index_type}} n = idx / ((H + pad_top + pad_bottom) * (W + pad_left + pad_right));

        {{index_type}} in_h = std::min<{{index_type}}>(std::max<{{index_type}}>(h - pad_top, 0), H - 1);
        {{index_type}} in_w = std::min<{{index_type}}>(std::max<{{index_type}}>(w - pad_left, 0), W - 1);
        {{index_type}} in_index = n * H * W + in_h * W + in_w;
        output[idx] = input[in_index];
    }
}

template <typename T>
__global__ void pad_kernel_circular_3d(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
    if (idx < total_elements) {
        {{index_type}} w = idx % (W + pad_left + pad_right);
        {{index_type}} h = (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
        {{index_type}} n = idx / ((H + pad_top + pad_bottom) * (W + pad_left + pad_right));

        {{index_type}} in_h = (h - pad_top + H) % H;
        {{index_type}} in_w = (w - pad_left + W) % W;
        {{index_type}} in_index = n * H * W + in_h * W + in_w;
        output[idx] = input[in_index];
    }
}

template <typename T>
__global__ void pad_kernel_constant_4d_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} C,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right,
    T pad_value
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
    if (idx < total_elements) {
        {{index_type}} c = idx % C;
        {{index_type}} w = (idx / C) % (W + pad_left + pad_right);
        {{index_type}} h = (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
        {{index_type}} n = idx / (C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right));

        if (h < pad_top || h >= H + pad_top || w < pad_left || w >= W + pad_left) {
            output[idx] = pad_value;
        } else {
            {{index_type}} in_h = h - pad_top;
            {{index_type}} in_w = w - pad_left;
            {{index_type}} in_index = n * H * W * C + in_h * W * C + in_w * C + c;
            output[idx] = input[in_index];
        }
    }
}


template <typename T>
__global__ void pad_kernel_reflect_4d_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} C,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
    if (idx < total_elements) {
        {{index_type}} c = idx % C;
        {{index_type}} w = (idx / C) % (W + pad_left + pad_right);
        {{index_type}} h = (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
        {{index_type}} n = idx / (C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right));

        {{index_type}} in_h = h < pad_top ? pad_top - h : (h >= H + pad_top ? 2 * H + pad_top - h - 2 : h - pad_top);
        {{index_type}} in_w = w < pad_left ? pad_left - w : (w >= W + pad_left ? 2 * W + pad_left - w - 2 : w - pad_left);
        {{index_type}} in_index = n * H * W * C + in_h * W * C + in_w * C + c;
        output[idx] = input[in_index];
    }
}

template <typename T>
__global__ void pad_kernel_replicate_4d_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} C,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
    if (idx < total_elements) {
        {{index_type}} c = idx % C;
        {{index_type}} w = (idx / C) % (W + pad_left + pad_right);
        {{index_type}} h = (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
        {{index_type}} n = idx / (C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right));

        {{index_type}} in_h = std::min<{{index_type}}>(std::max<{{index_type}}>(h - pad_top, 0), H - 1);
        {{index_type}} in_w = std::min<{{index_type}}>(std::max<{{index_type}}>(w - pad_left, 0), W - 1);
        {{index_type}} in_index = n * H * W * C + in_h * W * C + in_w * C + c;
        output[idx] = input[in_index];
    }
}

template <typename T>
__global__ void pad_kernel_circular_4d_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} C,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
    if (idx < total_elements) {
        {{index_type}} c = idx % C;
        {{index_type}} w = (idx / C) % (W + pad_left + pad_right);
        {{index_type}} h = (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
        {{index_type}} n = idx / (C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right));

        {{index_type}} in_h = (h - pad_top + H) % H;
        {{index_type}} in_w = (w - pad_left + W) % W;
        {{index_type}} in_index = n * H * W * C + in_h * W * C + in_w * C + c;
        output[idx] = input[in_index];
    }
}


}  // namespace
void {{function_name}}(
    const void* in_ptr,
    void* out_ptr,
    {{index_type}} N,
    {{index_type}} H,
    {{index_type}} W,
    {{index_type}} C,
    {{index_type}} pad_top,
    {{index_type}} pad_bottom,
    {{index_type}} pad_left,
    {{index_type}} pad_right,
    {{elem_input_type}} pad_value,
    {{index_type}} rank,
    const char* mode,
    {{prefix}}Stream_t stream
) {
    {{index_type}} total_elements;
    {{index_type}} threads_per_block = 256;
    {{index_type}} num_blocks;

    if (rank == 1) {
        total_elements = N + pad_left + pad_right;
        num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        if (strcmp(mode, "constant") == 0) {
            pad_kernel_constant_1d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N,
                pad_left, pad_right,
                pad_value
            );
        } else if (strcmp(mode, "reflect") == 0) {
            pad_kernel_reflect_1d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N,
                pad_left, pad_right
            );
        } else if (strcmp(mode, "replicate") == 0) {
            pad_kernel_replicate_1d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N,
                pad_left, pad_right
            );
        } else if (strcmp(mode, "circular") == 0) {
            pad_kernel_circular_1d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N,
                pad_left, pad_right
            );
        }
    } else if (rank == 2) {
        total_elements = N * (H + pad_left + pad_right);
        num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        if (strcmp(mode, "constant") == 0) {
            pad_kernel_constant_2d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H,
                pad_left, pad_right,
                pad_value
            );
        } else if (strcmp(mode, "reflect") == 0) {
            pad_kernel_reflect_2d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H,
                pad_left, pad_right
            );
        } else if (strcmp(mode, "replicate") == 0) {
            pad_kernel_replicate_2d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H,
                pad_left, pad_right
            );
        } else if (strcmp(mode, "circular") == 0) {
            pad_kernel_circular_2d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H,
                pad_left, pad_right
            );
        }
    } else if (rank == 3) {
        total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
        num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        if (strcmp(mode, "constant") == 0) {
            pad_kernel_constant_3d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H, W,
                pad_top, pad_bottom,
                pad_left, pad_right,
                pad_value
            );
        } else if (strcmp(mode, "reflect") == 0) {
            pad_kernel_reflect_3d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H, W,
                pad_top, pad_bottom,
                pad_left, pad_right
            );
        } else if (strcmp(mode, "replicate") == 0) {
            pad_kernel_replicate_3d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H, W,
                pad_top, pad_bottom,
                pad_left, pad_right
            );
        } else if (strcmp(mode, "circular") == 0) {
            pad_kernel_circular_3d<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H, W,
                pad_top, pad_bottom,
                pad_left, pad_right
            );
        }
    } else if (rank == 4) {
        total_elements = N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
        num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        if (strcmp(mode, "constant") == 0) {
            pad_kernel_constant_4d_nhwc<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H, W, C,
                pad_top, pad_bottom,
                pad_left, pad_right,
                pad_value
            );
        } else if (strcmp(mode, "reflect") == 0) {
            pad_kernel_reflect_4d_nhwc<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H, W, C,
                pad_top, pad_bottom,
                pad_left, pad_right
            );
        } else if (strcmp(mode, "replicate") == 0) {
            pad_kernel_replicate_4d_nhwc<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H, W, C,
                pad_top, pad_bottom,
                pad_left, pad_right
            );
        } else if (strcmp(mode, "circular") == 0) {
            pad_kernel_circular_4d_nhwc<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const {{elem_input_type}}*>(in_ptr),
                static_cast<{{elem_output_type}}*>(out_ptr),
                N, H, W, C,
                pad_top, pad_bottom,
                pad_left, pad_right
            );
        }
    }
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    const void*,
    void*,
    {{index_type}},
    {{index_type}},
    {{index_type}},
    {{index_type}},
    {{index_type}},
    {{index_type}},
    {{index_type}},
    {{index_type}},
    {{elem_input_type}},
    {{index_type}},
    const char*,
    {{prefix}}Stream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{N}},
{{indent}}    {{H}},
{{indent}}    {{W}},
{{indent}}    {{C}},
{{indent}}    {{pad_top}},
{{indent}}    {{pad_bottom}},
{{indent}}    {{pad_left}},
{{indent}}    {{pad_right}},
{{indent}}    {{pad_value}},
{{indent}}    {{rank}},
{{indent}}    "{{mode}}",
{{indent}}    stream
{{indent}});
"""
)


def gen_function_decl(func_attrs, backend_spec):
    """Function declaration generation

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        It describes the operation attributes
    backend_spec : custom class
        It specifies the corresponding backend dtypes of pytorch dtypes for many operations

    Returns
    -------
    str
        Rendered function declaration stmt
    """
    x = func_attrs["inputs"][0]
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        elem_input_type=backend_spec.dtype_to_backend_type(x._attrs["dtype"]),
    )


def gen_function_call(func_attrs, backend_spec, indent="  "):
    """Function call generation

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        It describes the operation attributes
    indent : str, optional
        Indent for template, by default "  "

    Returns
    -------
    str
        Rendered function call
    """
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]
    rank = len(x_shape)

    if rank == 1:
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            in_ptr=x._attrs["name"],
            out_ptr=y._attrs["name"],
            N=x_shape[0]._attrs["name"],
            H=0,
            W=0,
            C=0,
            pad_top=0,
            pad_bottom=0,
            pad_left=func_attrs["pad"][0],
            pad_right=func_attrs["pad"][1],
            pad_value=func_attrs["value"],
            rank=rank,
            mode=func_attrs["mode"],
            indent=indent,
        )
    elif rank == 2:
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            in_ptr=x._attrs["name"],
            out_ptr=y._attrs["name"],
            N=x_shape[0]._attrs["name"],
            H=x_shape[1]._attrs["name"],
            W=0,
            C=0,
            pad_top=0,
            pad_bottom=0,
            pad_left=func_attrs["pad"][0],
            pad_right=func_attrs["pad"][1],
            pad_value=func_attrs["value"],
            rank=rank,
            mode=func_attrs["mode"],
            indent=indent,
        )
    elif rank == 3:
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            in_ptr=x._attrs["name"],
            out_ptr=y._attrs["name"],
            N=x_shape[0]._attrs["name"],
            H=x_shape[1]._attrs["name"],
            W=x_shape[2]._attrs["name"],
            C=0,
            pad_top=func_attrs["pad"][2],
            pad_bottom=func_attrs["pad"][3],
            pad_left=func_attrs["pad"][0],
            pad_right=func_attrs["pad"][1],
            pad_value=func_attrs["value"],
            rank=rank,
            mode=func_attrs["mode"],
            indent=indent,
        )
    elif rank == 4:
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            in_ptr=x._attrs["name"],
            out_ptr=y._attrs["name"],
            N=x_shape[0]._attrs["name"],
            H=x_shape[1]._attrs["name"],
            W=x_shape[2]._attrs["name"],
            C=x_shape[3]._attrs["name"],
            pad_top=func_attrs["pad"][2],
            pad_bottom=func_attrs["pad"][3],
            pad_left=func_attrs["pad"][0],
            pad_right=func_attrs["pad"][1],
            pad_value=func_attrs["value"],
            rank=rank,
            mode=func_attrs["mode"],
            indent=indent,
        )
    else:
        raise NotImplementedError(f"unsupported rank {rank}")
