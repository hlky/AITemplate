import jinja2

SRC_TEMPLATE = jinja2.Template(
    """
#include <ait/device.h>
#include <ops/pad.h>

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
    ait::DeviceStream stream
) {
    InvokePad<{{index_type}}, {{elem_input_type}}, {{elem_output_type}}>(
        in_ptr,
        out_ptr,
        N,
        H,
        W,
        C,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        pad_value,
        rank,
        mode,
        stream
    );
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
    ait::DeviceStream
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
