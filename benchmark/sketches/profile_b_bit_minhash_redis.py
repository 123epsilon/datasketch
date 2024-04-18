"""
Benchmarking the performance and accuracy of b-bit MinHash.
"""
import time, logging, tracemalloc
from numpy import random
import matplotlib
import subprocess

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.lsh import MinHashLSH
from datasketch.hashfunc import *
import os

import redis

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x: ("a-%d-%d" % (x, x)).encode("utf-8")


def gen_minhash(size, seed, num_perm, num_bits):
    m = MinHash(num_perm=num_perm)
    s = set()
    random.seed(seed)
    for i in range(size):
        v = int_bytes(random.randint(1, size))
        m.update(v)
        s.add(v)

    b = bBitMinHash(m, num_bits)
    return (b, s)


def insert(num, size, seed, num_perm, num_bits):
    # init redis
    port = 6879
    subprocess.Popen(
                [f"/usr/bin/redis-server -p {port}",], 
                close_fds=True
            )

    lsh = MinHashLSH(
        num_perm=num_perm,
        storage_config={
            "type": "redis",
            "redis": {"host": "localhost", "port": port},
        }
    )

    for i in range(num):
        key = str(i)
        b = gen_minhash(size, seed, num_perm, num_bits)
        lsh.insert(i, b)

    del lsh
    # get size
    redis_conn = redis.Redis(host = "localhost", port = port)
    redis_size = redis_conn.info()['used_memory']

    # close redis
    redis_conn.shutdown()

    # del dump.rdb if it exists
    if os.path.exists('dump.rdb'):
        os.remove('dump.rdb')

    return redis_size / 1e6 # get in MB


def run_acc(size, num_perm, num_bits):
    num = 100_000
    logging.info("MinHash using %d permutation functions" % num_perm)
    mem_mb = insert(num, size, 1, num_perm, num_bits)
    return mem_mb

num_perms = range(10, 256, 20)
num_bits = [8, 16, 32]
bit_colors = colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]
output = "b_bit_minhash_lsh_profile.png"

logging.info("> Running profiling")
size = 5000
redis_sizes = {}
for b in num_bits:
    mems = [run_acc(size, n, b) for n in num_perms]
    redis_sizes[b] = mems

logging.info("> Plotting result")
fig, ax = plt.subplots(figsize=(10, 4))

for i, b in enumerate(num_bits):
    ax.plot(num_perms, redis_sizes[b], marker="+", color=bit_colors[i], label=f"{b} bits")
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Size of Redis Index (MB)")
ax.set_title("bBitMinHash LSH Index Size (MB)")
ax.grid()
ax.legend()

plt.tight_layout()
fig.savefig(output)
logging.info("Plot saved to %s" % output)
