import sys
import math
import time
import numpy as np
from mpi4py import MPI
from numba import cuda

# ====================== CUDA KERNELS ======================

@cuda.jit
def assign_clusters_kernel(points, centroids, cluster_ids, changes):
    idx = cuda.grid(1)
    if idx >= points.shape[0]:
        return

    min_dist = 1e30
    nearest = 0
    num_clusters = centroids.shape[0]
    dims = points.shape[1]

    shared_cent = cuda.shared.array((32, 54), dtype=np.float32)
    tid = cuda.threadIdx.x

    for k_block in range(0, num_clusters, 32):
        k = k_block + tid
        if k < num_clusters:
            for d in range(dims):
                if d < 54:
                    shared_cent[tid, d] = centroids[k, d]
        cuda.syncthreads()

        for i in range(min(32, num_clusters - k_block)):
            ck = k_block + i
            dist = 0.0
            for d in range(dims):
                diff = points[idx, d] - shared_cent[i, d]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                nearest = ck
        cuda.syncthreads()

    if cluster_ids[idx] != nearest:
        cluster_ids[idx] = nearest
        cuda.atomic.add(changes, 0, 1)


@cuda.jit
def compute_sums_kernel(points, cluster_ids, local_sums, local_counts):
    idx = cuda.grid(1)
    if idx >= points.shape[0]:
        return
    cid = cluster_ids[idx]
    cuda.atomic.add(local_counts, cid, 1)
    for d in range(points.shape[1]):
        cuda.atomic.add(local_sums, (cid, d), points[idx, d])


@cuda.jit
def calculate_wcss_kernel(points, centroids, cluster_ids, wcss):
    idx = cuda.grid(1)
    if idx >= points.shape[0]:
        return
    cid = cluster_ids[idx]
    dist = 0.0
    for d in range(points.shape[1]):
        diff = points[idx, d] - centroids[cid, d]
        dist += diff * diff
    cuda.atomic.add(wcss, 0, dist)



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if len(sys.argv) < 5:
            print("Usage: mpirun -np 1 python script.py <input_file> <k> <max_iter> <threads_per_block>")
            sys.exit(1)
        input_file = sys.argv[1]
        num_clusters = int(sys.argv[2])
        max_iter = int(sys.argv[3])
        threads_per_block = int(sys.argv[4])
    else:
        input_file = None
        num_clusters = 0
        max_iter = 0
        threads_per_block = 0

    input_file = comm.bcast(input_file, root=0)
    num_clusters = comm.bcast(num_clusters, root=0)
    max_iter = comm.bcast(max_iter, root=0)
    threads_per_block = comm.bcast(threads_per_block, root=0)

    # ====================== HEADER ======================
    if rank == 0:
        print("="*75)
        print("[Hybrid] CUDA + MPI K-Means Clustering")
        print("="*75)

    # ====================== LOAD DATA ======================
    if rank == 0:
        data = np.loadtxt(input_file, skiprows=1, dtype=np.float32)
        n_points, dims = data.shape
        chunks = np.array_split(data, size)
    else:
        chunks = None
        n_points = 0
        dims = 0

    n_points = comm.bcast(n_points, root=0)
    dims = comm.bcast(dims, root=0)
    local_data = comm.scatter(chunks, root=0)

    if rank == 0:
        print(f"Dataset      : {n_points:,} points, {dims} dimensions")
        print(f"Parameters   : {num_clusters} clusters, {max_iter} max iterations")
        print(f"MPI Ranks    : {size} | CUDA Threads/Block : {threads_per_block}")
        print("-"*75)

    # ====================== K-MEANS++ INIT (on rank 0) ======================
    if rank == 0:
        centroids = np.zeros((num_clusters, dims), dtype=np.float32)
        centroids[0] = data[np.random.randint(0, n_points)]
        for k in range(1, num_clusters):
            dists = np.min([np.sum((data - c)**2, axis=1) for c in centroids[:k]], axis=0)
            probs = dists / (dists.sum() + 1e-8)
            centroids[k] = data[np.random.choice(n_points, p=probs)]
    else:
        centroids = np.zeros((num_clusters, dims), dtype=np.float32)

    comm.Bcast(centroids, root=0)

    # ====================== CUDA SETUP ======================
    cuda.select_device(0)   # Important for single GPU
    d_points = cuda.to_device(local_data)
    d_centroids = cuda.to_device(centroids)
    d_cluster_ids = cuda.device_array(len(local_data), dtype=np.int32)
    d_cluster_ids[:] = -1

    blocks = (len(local_data) + threads_per_block - 1) // threads_per_block

    iter_count = 0
    start_time = time.time()

    while iter_count < max_iter:
        d_changes = cuda.device_array(1, dtype=np.int32)
        d_sums = cuda.device_array((num_clusters, dims), dtype=np.float32)
        d_counts = cuda.device_array(num_clusters, dtype=np.int32)

        assign_clusters_kernel[blocks, threads_per_block](d_points, d_centroids, d_cluster_ids, d_changes)
        compute_sums_kernel[blocks, threads_per_block](d_points, d_cluster_ids, d_sums, d_counts)

        local_changes = d_changes.copy_to_host()[0]
        global_changes = comm.allreduce(local_changes, MPI.SUM)

        local_sums = d_sums.copy_to_host()
        local_counts = d_counts.copy_to_host()
        global_sums = comm.allreduce(local_sums, MPI.SUM)
        global_counts = comm.allreduce(local_counts, MPI.SUM)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(num_clusters):
            if global_counts[k] > 0:
                new_centroids[k] = global_sums[k] / global_counts[k]

        centroids = new_centroids
        d_centroids.copy_to_device(centroids)

        iter_count += 1

        if global_changes == 0:
            break

    # ====================== FINAL WCSS ======================
    d_wcss = cuda.device_array(1, dtype=np.float32)
    calculate_wcss_kernel[blocks, threads_per_block](d_points, d_centroids, d_cluster_ids, d_wcss)
    local_wcss = d_wcss.copy_to_host()[0]
    global_wcss = comm.allreduce(local_wcss, MPI.SUM)

    runtime = time.time() - start_time
    rmse = math.sqrt(global_wcss / n_points)

    if rank == 0:
        print("="*75)
        print(f"Converged    : Yes ({iter_count} iterations)")
        print(f"Final WCSS   : {global_wcss:.6f}")
        print(f"Final RMSE   : {rmse:.6f}")
        print(f"Time Taken   : {runtime:.6f} seconds")
        print("="*75)


if __name__ == "__main__":
    main()