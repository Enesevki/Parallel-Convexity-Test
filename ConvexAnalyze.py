import numpy as np
from numba import cuda
import math
import time
import matplotlib.pyplot as plt
import pandas as pd

from threading import Thread

import random

def cross_product(a, b, c):
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])

def worker(i, points, results):
    n = len(points)
    a = points[i]
    b = points[(i + 1) % n]
    c = points[(i + 2) % n]
    results[i] = cross_product(a, b, c)

def is_convex(results):
    has_positive = any(val > 0 for val in results)
    has_negative = any(val < 0 for val in results)
    return not (has_positive and has_negative)

# CUDA kernel (GPU)
@cuda.jit
def cross_product_kernel(xs, ys, results, num_points):
    i = cuda.grid(1)
    if i < num_points:
        a_idx = i
        b_idx = (i + 1) % num_points
        c_idx = (i + 2) % num_points
        ax, ay = xs[a_idx], ys[a_idx]
        bx, by = xs[b_idx], ys[b_idx]
        cx, cy = xs[c_idx], ys[c_idx]
        val = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        results[i] = val

# Düzgün n-gen (her köşesi poligonun köşesi!)
def generate_large_ngon(num_points=100_000, radius=100):
    xs = []
    ys = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        xs.append(radius * math.cos(angle))
        ys.append(radius * math.sin(angle))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# GPU konvekslik
def gpu_convexity_test(xs, ys, threads_per_block):
    n = len(xs)
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    d_xs = cuda.to_device(xs)
    d_ys = cuda.to_device(ys)
    d_results = cuda.device_array(n, dtype=np.float32)
    t1 = time.time()
    cross_product_kernel[blocks_per_grid, threads_per_block](d_xs, d_ys, d_results, n)
    cuda.synchronize()
    t2 = time.time()
    results = d_results.copy_to_host()
    return t2 - t1  # Sadece süreyi döndür

# CPU konvekslik (NumPy vektörize)
def cpu_convexity_test(xs, ys):
    n = len(xs)
    a_x = xs
    a_y = ys
    b_x = np.roll(xs, -1)
    b_y = np.roll(ys, -1)
    c_x = np.roll(xs, -2)
    c_y = np.roll(ys, -2)
    t1 = time.time()
    val = (b_x - a_x) * (c_y - b_y) - (b_y - a_y) * (c_x - b_x)
    t2 = time.time()
    return t2 - t1  # Sadece süreyi döndür

def cpu_gpu_benchmark_plot():
    threads_per_block = 256
    sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000]
    num_trials = 4
    data = []
    print(f"{'Boyut':>10} | {'CPU (sn)':>10} | {'GPU (sn)':>10}")
    print("-"*34)
    for size in sizes:
        # Isıtma
        xs, ys = generate_large_ngon(size)
        _ = gpu_convexity_test(xs, ys, threads_per_block)
        cpu_times = []
        gpu_times = []
        for trial in range(num_trials):
            xs, ys = generate_large_ngon(size)
            t_cpu = cpu_convexity_test(xs, ys)
            t_gpu = gpu_convexity_test(xs, ys, threads_per_block)
            cpu_times.append(t_cpu)
            gpu_times.append(t_gpu)
        avg_cpu = np.mean(cpu_times)
        avg_gpu = np.mean(gpu_times)
        data.append({"Boyut": size, "CPU": avg_cpu, "GPU": avg_gpu})
        print(f"{size:>10} | {avg_cpu:>10.5f} | {avg_gpu:>10.5f}")
    df = pd.DataFrame(data)
    plt.figure(figsize=(10,6))
    plt.plot(df["Boyut"], df["CPU"], marker='o', label="CPU (NumPy)")
    plt.plot(df["Boyut"], df["GPU"], marker='o', label="GPU (CUDA)")
    plt.xlabel("Poligon Köşe Sayısı")
    plt.ylabel("Konvekslik Hesaplama Süresi (sn)")
    plt.title("Konvekslik Testi: CPU vs GPU Performans Karşılaştırması")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    print("\n--- Tablo ---")
    print(df)
    # df.to_csv("cpu_gpu_benchmark.csv", index=False)

def main():
    # Örnek olarak rastgele poligon (kullanıcıdan da alınabilir)
    n = int(input("Poligonun köşe sayısı: "))
    points = [(random.randint(0,100), random.randint(0,100)) for _ in range(n)]

    print("Noktalar:", points)
    results = [0]*n
    threads = []
    for i in range(n):
        t = Thread(target=worker, args=(i, points, results))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    print("Çapraz çarpımlar:", results)
    if is_convex(results):
        print("Poligon KONVEKS.")
    else:
        print("Poligon KONKAV.")

if __name__ == "__main__":
    main()
    print("CPU & GPU Benchmark Analizi Başlatılıyor...")
    cpu_gpu_benchmark_plot()
