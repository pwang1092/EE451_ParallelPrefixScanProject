"""
inspect_bin.py: Print stats and sample values from a .bin input or reference file.

Usage:
    python3 inspect_bin.py <file.bin> [--type input|ref]

    --type input : file contains [a | b] concatenated (2 * B*L*D floats)
    --type ref   : file contains [x] only             (1 * B*L*D floats)

    ie:
    python3 inspect_bin.py ../SyntheticData/inputs/input_B1_L1024_D16.bin --type input
    python3 inspect_bin.py SequentialData/ref_B1_L1024_D16.bin --type ref
"""

import sys
import numpy as np

def parse_filename(path):
    """Extract B, L, D from filename like input_B1_L1024_D16.bin"""
    import re
    m = re.search(r'B(\d+)_L(\d+)_D(\d+)', path)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def stats(arr, name):
    print(f"  {name}: min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}")
    print(f"         first 8 values: {arr.flat[:8]}")

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

path = sys.argv[1]
ftype = "input"
if "--type" in sys.argv:
    ftype = sys.argv[sys.argv.index("--type") + 1]

B, L, D = parse_filename(path)
if B is None:
    print("Could not parse B/L/D from filename.")
    sys.exit(1)

n = B * L * D
data = np.fromfile(path, dtype=np.float32)
print(f"\nFile:   {path}")
print(f"B={B}  L={L}  D={D}  n={n}  file_size={len(data)} floats\n")

if ftype == "input":
    if len(data) != 2 * n:
        print(f"Expected {2*n} floats, got {len(data)}")
        sys.exit(1)
    a = data[:n].reshape(B, L, D)
    b = data[n:].reshape(B, L, D)
    stats(a, "a (transition scalars)")
    stats(b, "b (input projections) ")
elif ftype == "ref":
    if len(data) != n:
        print(f"Expected {n} floats, got {len(data)}")
        sys.exit(1)
    x = data.reshape(B, L, D)
    stats(x, "x (hidden state output)")
    print(f"  last value (x[L-1, D-1]): {x[0, -1, -1]:.6f}")
