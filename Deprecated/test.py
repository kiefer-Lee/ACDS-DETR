import sys
sys.path.insert(0, ".")
from models.deformable_attention import HAS_MS_DEFORM_ATTN
print("HAS_MS_DEFORM_ATTN =", HAS_MS_DEFORM_ATTN)