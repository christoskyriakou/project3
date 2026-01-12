import numpy as np
import subprocess
import kahip

# BUILD KNN GRAPH USING C IVFFLAT (C BINARY: ivfflat_knn)
def build_knn_graph_with_ivfflat(X, k, data_path, data_type, nprobe=15):
    """
    X: numpy array (n, d)
    k: number of neighbors
    data_path: original MNIST/SIFT file (idx3-ubyte, fvecs, bvecs)
    data_type: 'mnist' or 'sift'
    nprobe: number of probes

    This function:
    1. Calls ./ivfflat/ivfflat_knn using subprocess
    2. Reads knn_out.bin (written by C code)
    3. Returns adjacency list graph
    """

    out_path = "knn_out.bin"
    cmd = [
        "./ivfflat/ivfflat_knn",
        data_type,
        data_path,
        str(k),
        str(nprobe),
        out_path
    ]

    print("[IVFFLAT] Running C IVFFlat binary...")

    subprocess.run(cmd, check=True)

    n = X.shape[0]
    print(f"[IVFFLAT] Reading knn_out.bin ({n} x {k}) ...")

    neighbors = np.fromfile(out_path, dtype=np.int32).reshape(n, k)

    graph = [list(row) for row in neighbors]

    print("[IVFFLAT] KNN graph build complete.")
    return graph


# MAKE UNDIRECTED + WEIGHTED GRAPH
def make_undirected_weighted(graph):
    n = len(graph)
    undir = [{} for _ in range(n)]

    for i in range(n):
        for j in graph[i]:
            w = 2 if i in graph[j] else 1
            undir[i][j] = w
            undir[j][i] = w

    return undir


# CONVERT UNDIRECTED GRAPH -> CSR FORMAT
def graph_to_csr(undir):
    n = len(undir)
    xadj = [0]
    adjncy = []
    adjcwgt = []

    for i in range(n):
        for j, w in undir[i].items():
            adjncy.append(j)
            adjcwgt.append(w)
        xadj.append(len(adjncy))

    return xadj, adjncy, adjcwgt


# RUN KaHIP PARTITIONING
def run_kahip(ugraph, m, imbalance, mode):
    xadj, adjncy, adjcwgt = graph_to_csr(ugraph)
    n = len(ugraph)

    print(f"[KaHIP] Graph: {n} vertices, {len(adjncy)} edges")
    print(f"[KaHIP] Partitioning into {m} parts (imbalance={imbalance})...")

    vwgt = [1] * n

    edgecut, parts = kahip.kaffpa(
        vwgt,
        xadj,
        adjcwgt,
        adjncy,
        int(m),
        float(imbalance),
        True,
        0,
        int(mode)
    )

    print(f"[KaHIP] Edgecut: {edgecut}")

    return np.array(parts, dtype=np.int32)
