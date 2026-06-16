import torch


def explicit_adjacent_rope(x, angles):
    """
    Ground truth RoPE for adjacent pairs.

    x = [a, b, c, d]
    angles = [A, B]

    Output:
    [a*cosA - b*sinA,
     a*sinA + b*cosA,
     c*cosB - d*sinB,
     c*sinB + d*cosB]
    """
    out = torch.empty_like(x)

    for i, angle in enumerate(angles):
        x0 = x[2 * i]
        x1 = x[2 * i + 1]

        c = torch.cos(angle)
        s = torch.sin(angle)

        out[2 * i] = x0 * c - x1 * s
        out[2 * i + 1] = x0 * s + x1 * c

    return out


def buggy_rotate_half(x):
    """
    This is the implementation from your pasted code.

    For x = [a, b, c, d]
    returns [-c, -d, a, b]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def fixed_adjacent_rotate_half(x):
    """
    Correct rotate_half for adjacent pair RoPE.

    For x = [a, b, c, d]
    returns [-b, a, -d, c]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


def rope_using_rotate_half(x, angles, rotate_half_fn):
    """
    Uses RoPE formula:

    x_rotated = x*cos + rotate_half(x)*sin

    For adjacent-pair RoPE, cos/sin must be:
    [cosA, cosA, cosB, cosB]
    [sinA, sinA, sinB, sinB]
    """
    cos = torch.cos(angles).repeat_interleave(2)
    sin = torch.sin(angles).repeat_interleave(2)

    return x * cos + rotate_half_fn(x) * sin


# ---------------- TEST DATA ----------------

x = torch.tensor([0.8, 0.3, -0.5, 0.2])

# Pair 0 angle = 1.0
# Pair 1 angle = 0.01
angles = torch.tensor([1.0, 0.01])

expected = explicit_adjacent_rope(x, angles)

buggy = rope_using_rotate_half(x, angles, buggy_rotate_half)
fixed = rope_using_rotate_half(x, angles, fixed_adjacent_rotate_half)

print("Input x:")
print(x)

print("\nExpected explicit adjacent RoPE:")
print(expected)

print("\nBuggy implementation output:")
print(buggy)

print("\nFixed implementation output:")
print(fixed)

print("\nBuggy matches expected?")
print(torch.allclose(buggy, expected, atol=1e-6))

print("\nFixed matches expected?")
print(torch.allclose(fixed, expected, atol=1e-6))

assert not torch.allclose(buggy, expected, atol=1e-6), "Buggy version unexpectedly matched"
assert torch.allclose(fixed, expected, atol=1e-6), "Fixed version did not match"

print("\nTest result: original rotate_half is inconsistent with adjacent-pair RoPE.")