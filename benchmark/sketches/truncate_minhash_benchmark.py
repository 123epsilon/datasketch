'''
Benchmarking the performance and accuracy of MinHash with different hash sizes.
'''
import time, logging
from numpy import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasketch.minhash import MinHash
from datasketch.hashfunc import *

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x : ("a-%d-%d" % (x, x)).encode('utf-8')

def get_hashfunc(num_bits):
	if num_bits == 8:
		return sha1_hash8
	elif num_bits == 16:
		return sha1_hash16
	elif num_bits == 32:
		return sha1_hash32
	elif num_bits == 64:
		return sha1_hash64
	else:
		return sha1_hash128

def run_perf(card, num_perm, num_bits):
	hashfunc = get_hashfunc(num_bits)
	dur = 0
	n_trials = 5
	for i in range(n_trials):
		m = MinHash(num_perm=num_perm, hashfunc=hashfunc)
		logging.info("MinHash using %d permutation functions" % num_perm)
		start = time.perf_counter()
		for i in range(card):
			m.update(int_bytes(i))
		duration = time.perf_counter() - start
		dur += duration
		logging.info("Digested %d hashes in %.4f sec" % (card, duration))
	return dur / n_trials


def _run_acc(size, seed, num_perm, num_bits):
	hashfunc = get_hashfunc(num_bits)
	m = MinHash(num_perm=num_perm, hashfunc=hashfunc)
	s = set()
	random.seed(seed)
	for i in range(size):
		v = int_bytes(random.randint(1, size))
		m.update(v)
		s.add(v)
	return (m, s)

def run_acc(size, num_perm, num_bits):
	logging.info("MinHash using %d permutation functions" % num_perm)
	m1, s1 = _run_acc(size, 1, num_perm, num_bits)
	m2, s2 = _run_acc(size, 4, num_perm, num_bits)
	j = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
	j_e = m1.jaccard(m2)
	err = abs(j - j_e)
	return err

num_perms = range(10, 256, 20)
num_bits = [8, 16, 32, 64]
bit_colors = ['b', 'g', 'r', 'y']
output = "minhash_precision_benchmark.png"

logging.info("> Running performance tests")
card = 5000
perf_times = {}
for b in num_bits:
	run_times = [run_perf(card, n, b) for n in num_perms]
	perf_times[b] = run_times


logging.info("> Running accuracy tests")
size = 5000
errors = {}
for b in num_bits:
	errs = [run_acc(size, n, b) for n in num_perms]
	errors[b] = errs

logging.info("> Plotting result")
fig, axe = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
ax = axe[1]
for i, b in enumerate(num_bits):
	ax.plot(num_perms, perf_times[b], marker='+', color=bit_colors[i], label=f"{b} bits")
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Running time (sec)")
ax.set_title("MinHash performance")
ax.grid()
ax.legend()
ax = axe[0]
for i, b in enumerate(num_bits):
	ax.plot(num_perms, errors[b], marker='+', color=bit_colors[i], label=f"{b} bits")
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Absolute error in Jaccard estimation")
ax.set_title("MinHash accuracy")
ax.grid()
ax.legend()

print(errors[32])
print(errors[64])

plt.tight_layout()
fig.savefig(output)
logging.info("Plot saved to %s" % output)
