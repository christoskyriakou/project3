import numpy as np
import struct
import os

# READ FVECs (float32 vectors)
def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    if a.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    d = a[0]
    a = a.reshape(-1, d + 1)
    return a[:, 1:].astype(np.float32)

# READ BVECs (uint8 vectors)
def read_bvecs(path):
    with open(path, "rb") as f:
        content = f.read()
    offset = 0
    vecs = []
    while offset < len(content):
        dim = struct.unpack_from('i', content, offset)[0]
        offset += 4
        vec = np.frombuffer(content, dtype=np.uint8, count=dim, offset=offset)
        offset += dim
        vecs.append(vec)
    return np.vstack(vecs).astype(np.float32)

# READ IVECs (ground truth)
def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    a = a.reshape(-1, d + 1)
    return a[:, 1:]

# READ MNIST RAW IDX AND CONVERT TO BVECs
def read_mnist_idx(path):
    """Returns MNIST raw (n, 784) uint8"""
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Invalid MNIST IDX file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)
    return data

def convert_mnist_to_bvecs(idx_path, out_path):
    """Convert raw MNIST idx to .bvecs file."""
    print(f"[MNIST] Converting {idx_path} -> {out_path}")
    X = read_mnist_idx(idx_path)
    n, d = X.shape
    with open(out_path, "wb") as f:
        for i in range(n):
            f.write(struct.pack('i', d))  # dimension
            f.write(X[i].tobytes())       # vector
    print("[MNIST] Conversion done.")
    return out_path

# UNIFIED LOADER
def load_dataset(path):
    """
    Smart loader: accepts both base names and full filenames.
    Supports:
      - *.fvecs
      - *.bvecs
      - *.ivecs
      - *.idx3-ubyte (auto converts to bvecs)
    """
    # If user passed exact file
    if os.path.isfile(path):

        # ---- FVECs ----
        if path.endswith(".fvecs"):
            print("[INFO] Loading SIFT (.fvecs)")
            return read_fvecs(path), "sift"

        # ---- BVECs ----
        if path.endswith(".bvecs"):
            print("[INFO] Loading MNIST (.bvecs)")
            return read_bvecs(path), "mnist"

        # ---- IVECs ----
        if path.endswith(".ivecs"):
            print("[INFO] Loading GT (.ivecs)")
            return read_ivecs(path), "gt"

        # ---- IDX (MNIST) ----
        if path.endswith(".idx3-ubyte"):
            print("[INFO] Converting MNIST idx → .bvecs")
            out_bvecs = path.replace(".idx3-ubyte", ".bvecs")
            convert_mnist_to_bvecs(path, out_bvecs)
            return read_bvecs(out_bvecs), "mnist"

        raise ValueError(f"Unknown file type: {path}")

    # If user passed base (no extension)
    base = path
    fpath = base + ".fvecs"
    bpath = base + ".bvecs"
    idx_path = base + ".idx3-ubyte"
    ivec_path = base + ".ivecs"

    if os.path.exists(fpath):
        print("[INFO] Loading SIFT (.fvecs)")
        return read_fvecs(fpath), "sift"

    if os.path.exists(bpath):
        print("[INFO] Loading MNIST (.bvecs)")
        return read_bvecs(bpath), "mnist"

    if os.path.exists(idx_path):
        print("[INFO] Converting MNIST idx → .bvecs")
        out_bvecs = base + ".bvecs"
        convert_mnist_to_bvecs(idx_path, out_bvecs)
        return read_bvecs(out_bvecs), "mnist"

    if os.path.exists(ivec_path):
        print("[INFO] Loading GT (.ivecs)")
        return read_ivecs(ivec_path), "gt"

    raise ValueError(f"Unknown dataset type: {path}")
