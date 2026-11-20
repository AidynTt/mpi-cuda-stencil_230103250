import os
import sys
import json
import math
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64


TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15

def get_gpu_uuid() -> str:
    """Get the active NVIDIA GPU UUID."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8"
    )
    return out.strip().splitlines()[0].strip()

def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Generate a unique seed from GitHub username and GPU."""
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)

def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(
            f"Expected exactly 4 MPI ranks, got {size}."
        )

    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)

    instance_seed = derive_student_seed(
        github_user, gpu_uuid
    )

    cart_comm = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False
    )

    coords = cart_comm.Get_coords(rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return (
        cart_comm, rank, coords,
        nbr_up, nbr_down,
        nbr_left, nbr_right,
        instance_seed
    )

# TASK 2: Initialize Subdomains
#I used ChatGPT


def initialize_subdomains(coords, instance_seed):
    ny, nx = 256, 256
    dx = dy = 1.0 / 513.0

    rows = coords[0] * ny + np.arange(258) - 1
    cols = coords[1] * nx + np.arange(258) - 1

    y = rows * dy
    x = cols * dx
    X, Y = np.meshgrid(x, y)

    kappa = 0.08 + 0.16 * (
        np.sin(np.pi * X)**2 *
        np.sin(np.pi * Y)**2
    )

    rng = np.random.default_rng(instance_seed)
    cx, cy = rng.uniform(0.3, 0.7, size=2)

    u = np.exp(
        -((X - cx)**2 + (Y - cy)**2) / 0.01
    )

    source = np.exp(
        -((X - 0.5)**2 + (Y - 0.5)**2) / 0.002
    )

    if coords[0] == 0:
        u[0, :] = 0.0
    if coords[0] == 1:
        u[-1, :] = 0.0
    if coords[1] == 0:
        u[:, 0] = 0.0
    if coords[1] == 1:
        u[:, -1] = 0.0

    return (
        cuda.to_device(u.astype(np.float64)),
        cuda.to_device(kappa.astype(np.float64)),
        cuda.to_device(source.astype(np.float64))
    )

@cuda.jit
def pack_halos_kernel(u, top, bottom, left, right):
    idx = cuda.grid(1)
    ny = u.shape[0] - 2
    nx = u.shape[1] - 2

    if idx < nx:
        top[idx] = u[1, idx + 1]
        bottom[idx] = u[ny, idx + 1]

    if idx < ny:
        left[idx] = u[idx + 1, 1]
        right[idx] = u[idx + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, top, bottom, left, right):
    idx = cuda.grid(1)
    ny = u.shape[0] - 2
    nx = u.shape[1] - 2

    if idx < nx:
        u[0, idx + 1] = top[idx]
        u[ny + 1, idx + 1] = bottom[idx]

    if idx < ny:
        u[idx + 1, 0] = left[idx]
        u[idx + 1, nx + 1] = right[idx]

#TASK 3: Asynchronous MPI Halo Exchange

def exchange_halos_async(
    cart_comm, neighbors, send_host, recv_host
):
    opposite = (1, 0, 3, 2)
    requests = []

    for direction, neighbor in enumerate(neighbors):
        if neighbor == MPI.PROC_NULL:
            recv_host[direction].fill(0.0)
            continue

        requests.append(
            cart_comm.Irecv(
                recv_host[direction],
                source=neighbor,
                tag=opposite[direction]
            )
        )

    for direction, neighbor in enumerate(neighbors):
        if neighbor == MPI.PROC_NULL:
            continue

        requests.append(
            cart_comm.Isend(
                send_host[direction],
                dest=neighbor,
                tag=direction
            )
        )

    return requests

#TASK 4: Shared-Memory Tiled Stencil
#I used ChatGpt


@cuda.jit
def tiled_stencil_kernel(
    u, u_next, kappa, source,
    dt, dx, dy,
    r_start, r_end, c_start, c_end
):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = cuda.blockIdx.y * 16 + ty + r_start
    col = cuda.blockIdx.x * 16 + tx + c_start

    sy = ty + 1
    sx = tx + 1

    smem = cuda.shared.array(
        shape=(18, 18), dtype=float64
    )

    # All threads initialize their shared-memory center
    valid = row < u.shape[0] and col < u.shape[1]

    if valid:
        smem[sy, sx] = u[row, col]
    else:
        smem[sy, sx] = 0.0

    if tx == 0:
        smem[sy, 0] = u[row, col - 1] if (
            row < u.shape[0] and 0 < col < u.shape[1]
        ) else 0.0

    if tx == 15:
        smem[sy, 17] = u[row, col + 1] if (
            row < u.shape[0] and col + 1 < u.shape[1]
        ) else 0.0

    if ty == 0:
        smem[0, sx] = u[row - 1, col] if (
            0 < row < u.shape[0] and col < u.shape[1]
        ) else 0.0

    if ty == 15:
        smem[17, sx] = u[row + 1, col] if (
            row + 1 < u.shape[0] and col < u.shape[1]
        ) else 0.0

    cuda.syncthreads()

    if (
        r_start <= row <= r_end
        and c_start <= col <= c_end
    ):
        center = smem[sy, sx]

        up = smem[sy - 1, sx]
        down = smem[sy + 1, sx]
        left = smem[sy, sx - 1]
        right = smem[sy, sx + 1]

        kc = kappa[row, col]
        ku = 0.5 * (kc + kappa[row - 1, col])
        kd = 0.5 * (kc + kappa[row + 1, col])
        kl = 0.5 * (kc + kappa[row, col - 1])
        kr = 0.5 * (kc + kappa[row, col + 1])

        diffusion = (
            (kd * (down - center)
             - ku * (center - up)) / (dy * dy)
            +
            (kr * (right - center)
             - kl * (center - left)) / (dx * dx)
        )

        u_next[row, col] = center + dt * (
            diffusion
            - 0.15 * center ** 3
            + source[row, col]
        )


# ============================================================
# Full 400-step solver
# AI assistance: ChatGPT (OpenAI)
# ============================================================

import time
import re

DIRECTIONS = ("top", "bottom", "left", "right")


def launch_stencil(u, out, kappa, source,
                   dt, dx, dy, box, stream):
    r0, r1, c0, c1 = box
    width = c1 - c0 + 1
    height = r1 - r0 + 1

    blocks = (
        (width + 15) // 16,
        (height + 15) // 16
    )

    tiled_stencil_kernel[blocks, (16, 16), stream](
        u, out, kappa, source,
        dt, dx, dy,
        r0, r1, c0, c1
    )


def step_overlapped(
    cart, neighbors, u, out, kappa, source,
    dt, dx, dy, buffers, stream_boundary,
    stream_interior, overlap=True
):
    sd, rd, sh, rh = buffers

    # Pack four GPU boundaries
    pack_halos_kernel[1, 256, stream_boundary](
        u, *sd
    )

    # GPU -> pinned host
    for i in range(4):
        sd[i].copy_to_host(
            sh[i], stream=stream_boundary
        )

    stream_boundary.synchronize()

    # Start non-blocking MPI communication
    requests = exchange_halos_async(
        cart, neighbors, sh, rh
    )

    # Deep interior is independent of incoming halos
    if overlap:
        launch_stencil(
            u, out, kappa, source,
            dt, dx, dy,
            (2, 255, 2, 255),
            stream_interior
        )

    MPI.Request.Waitall(requests)

    # Pinned host -> GPU
    for i in range(4):
        rd[i].copy_to_device(
            rh[i], stream=stream_boundary
        )

    unpack_halos_kernel[1, 256, stream_boundary](
        u, *rd
    )

    # In serialized mode, wait for communication first
    if not overlap:
        stream_boundary.synchronize()
        launch_stencil(
            u, out, kappa, source,
            dt, dx, dy,
            (2, 255, 2, 255),
            stream_interior
        )

    # Four non-overlapping perimeter regions
    boundary_boxes = [
        (1, 1, 1, 256),
        (256, 256, 1, 256),
        (2, 255, 1, 1),
        (2, 255, 256, 256)
    ]

    for box in boundary_boxes:
        launch_stencil(
            u, out, kappa, source,
            dt, dx, dy,
            box, stream_boundary
        )

    stream_boundary.synchronize()
    stream_interior.synchronize()

    return out, u


def distributed_metrics(cart, u, previous, dx, dy):
    current = u.copy_to_host()
    old = previous.copy_to_host()

    inside = current[1:257, 1:257]
    difference = (
        inside - old[1:257, 1:257]
    )

    local_energy = float(
        np.sum(inside) * dx * dy
    )
    local_residual_sq = float(
        np.sum(difference ** 2) * dx * dy
    )

    local_values = np.array(
        [local_energy, local_residual_sq],
        dtype=np.float64
    )
    global_values = np.zeros_like(local_values)

    cart.Allreduce(
        local_values,
        global_values,
        op=MPI.SUM
    )

    return (
        float(global_values[0]),
        float(np.sqrt(global_values[1]))
    )


def run_mode(cart, coords, neighbors, seed, mode):
    dx = dy = 1.0 / 513.0

    # Stable explicit time step
    dt = 0.1 * min(dx, dy) ** 2 / 0.24

    u, kappa, source = initialize_subdomains(
        coords, seed
    )
    out = cuda.device_array_like(u)

    sb = cuda.stream()
    si = cuda.stream()

    sd = [
        cuda.device_array(256, dtype=np.float64)
        for _ in range(4)
    ]
    rd = [
        cuda.device_array(256, dtype=np.float64)
        for _ in range(4)
    ]
    sh = [
        cuda.pinned_array(256, dtype=np.float64)
        for _ in range(4)
    ]
    rh = [
        cuda.pinned_array(256, dtype=np.float64)
        for _ in range(4)
    ]

    buffers = (sd, rd, sh, rh)

    rank = cart.Get_rank()
    trace = []

    energy0, _ = distributed_metrics(
        cart, u, u, dx, dy
    )
    trace.append({
        "step": 0,
        "energy": energy0,
        "residual": 0.0
    })

    if rank == 0:
        print(
            f"{mode} step 0: "
            f"energy={energy0:.12e}",
            flush=True
        )

    cuda.synchronize()
    cart.Barrier()
    start = time.perf_counter()

    for step in range(1, STEPS + 1):
        u, out = step_overlapped(
            cart, neighbors, u, out,
            kappa, source, dt, dx, dy,
            buffers, sb, si,
            overlap=(mode == "overlapped")
        )

        if step % 100 == 0:
            energy, residual = distributed_metrics(
                cart, u, out, dx, dy
            )
            trace.append({
                "step": step,
                "energy": energy,
                "residual": residual
            })

            if rank == 0:
                print(
                    f"{mode} step {step}: "
                    f"energy={energy:.12e}, "
                    f"residual={residual:.12e}",
                    flush=True
                )

    cuda.synchronize()
    cart.Barrier()

    local_ms = (time.perf_counter() - start) * 1000
    elapsed_ms = cart.allreduce(
        local_ms, op=MPI.MAX
    )

    if rank == 0:
        print(
            f"{mode} time: {elapsed_ms:.3f} ms",
            flush=True
        )

    return u.copy_to_host(), elapsed_ms, trace


def audit_ptx():
    asm = ""

    for signature in tiled_stencil_kernel.signatures:
        try:
            asm = tiled_stencil_kernel.inspect_asm(
                signature
            )
            if asm:
                break
        except Exception:
            pass

    counts = {
        instruction: len(
            re.findall(
                r"\b" + re.escape(instruction) + r"\b",
                asm
            )
        )
        for instruction in (
            "st.shared", "ld.shared", "bar.sync"
        )
    }

    return counts, all(v > 0 for v in counts.values())


def run_solver(github_user):
    (
        cart, rank, coords,
        up, down, left, right, seed
    ) = init_distributed_context(github_user)

    neighbors = (up, down, left, right)

    if rank == 0:
        print("MPI ranks: 4")
        print("Cartesian grid: 2 x 2")
        print("Time steps:", STEPS)
        print("Instance seed:", seed, flush=True)

    serial_u, serial_ms, serial_trace = run_mode(
        cart, coords, neighbors,
        seed, "serialized"
    )

    overlap_u, overlap_ms, overlap_trace = run_mode(
        cart, coords, neighbors,
        seed, "overlapped"
    )

    local_diff = float(np.max(
        np.abs(serial_u - overlap_u)
    ))

    parity = cart.allreduce(
        local_diff, op=MPI.MAX
    )

    ptx_counts, ptx_passed = audit_ptx()

    all_ptx_passed = cart.allreduce(
        ptx_passed, op=MPI.LAND
    )

    if rank == 0:
        speedup = (
            serial_ms / overlap_ms
            if overlap_ms > 0 else None
        )

        manifest = {
            "github_username": github_user,
            "gpu_uuid": get_gpu_uuid(),
            "instance_seed": seed,
            "mpi_ranks": 4,
            "cartesian_grid": [2, 2],
            "global_grid": [512, 512],
            "time_steps": STEPS,
            "serialized_ms": serial_ms,
            "overlapped_ms": overlap_ms,
            "speedup": speedup,
            "parity_max_diff": parity,
            "parity_passed": parity < 1e-12,
            "ptx_counts_rank0": ptx_counts,
            "ptx_audit_passed": bool(all_ptx_passed),
            "serialized_trace": serial_trace,
            "overlapped_trace": overlap_trace
        }

        with open(
            "submission_manifest.json", "w"
        ) as f:
            json.dump(manifest, f, indent=2)

        print("\n=== FINAL RESULTS ===")
        print("Serialized:", serial_ms, "ms")
        print("Overlapped:", overlap_ms, "ms")
        print("Speedup:", speedup)
        print("Parity max diff:", parity)
        print("Parity passed:", parity < 1e-12)
        print("PTX counts:", ptx_counts)
        print("PTX audit passed:", all_ptx_passed)
        print(
            "Manifest: submission_manifest.json",
            flush=True
        )

    cart.Barrier()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python distributed_cuda_solver.py "
            "<github_username>"
        )

    run_solver(sys.argv[1])
