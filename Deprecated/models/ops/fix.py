from pathlib import Path
import re

cu = Path("src/cuda/ms_deform_attn_cuda.cu")
text = cu.read_text()

text = text.replace(
    "AT_DISPATCH_FLOATING_TYPES(value.type(), \"ms_deform_attn_forward_cuda\"",
    "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), \"ms_deform_attn_forward_cuda\"",
)
text = text.replace(
    "AT_DISPATCH_FLOATING_TYPES(value.type(), \"ms_deform_attn_backward_cuda\"",
    "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), \"ms_deform_attn_backward_cuda\"",
)

cu.write_text(text)

print("patched only AT_DISPATCH value.type() -> value.scalar_type()")