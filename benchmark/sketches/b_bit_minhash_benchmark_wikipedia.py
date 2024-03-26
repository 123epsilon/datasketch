"""
Benchmarking the performance and accuracy of b-bit MinHash.
"""
import time, logging
from numpy import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.hashfunc import *
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

# Produce some bytes
# int_bytes = lambda x: ("a-%d-%d" % (x, x)).encode("utf-8")

wiki_data = load_dataset("wikipedia", "20220301.simple")
N_DOCS = len(wiki_data['train'])
N_TRIALS = 30

def run_perf(card, num_perm, num_bits):
	dur = 0
	for i in range(N_TRIALS):
		doc = wiki_data['train'][random.randint(0, N_DOCS)]['text']
		m = MinHash(num_perm=num_perm)
		logging.info("MinHash using %d permutation functions" % num_perm)
		start = time.perf_counter()
		# get real document data, but upper bound the size to evaluate runtime performance
		s = set(doc.split()[:card])
		for d in s:
			m.update(d.encode("utf8"))

		b = bBitMinHash(m, num_bits)
		duration = time.perf_counter() - start
		dur += duration
		logging.info("Digested %d hashes in %.4f sec" % (card, duration))
	return dur / N_TRIALS


# takes a document string as first argument
def _run_acc(doc, num_perm, num_bits):
	m = MinHash(num_perm=num_perm)
	s = set(doc.split())
	for d in s:
		m.update(d.encode("utf8"))
	b = bBitMinHash(m, num_bits)
	return (b, s)


def run_acc(num_perm, num_bits):
	logging.info("MinHash using %d permutation functions" % num_perm)
	avg_err = 0
	avg_jaccard = 0
	for i in range(N_TRIALS):
		random.seed(i+1)
		i1 = random.randint(0, N_DOCS)

		overlap = random.uniform()
		doc1 = wiki_data['train'][i1]['text']
		# generate a random overlapping region of text given a start point
		# this isn't perfect since we could be cutting off the start/ending words in the region
		# but that won't affect the estimation too much
		overlap_size = int(len(doc1)*overlap)
		overlap_start = random.randint(0, len(doc1)-overlap_size)
		doc2 = wiki_data['train'][i1]['text'][overlap_start:overlap_start+overlap_size]
		m1, s1 = _run_acc(doc1, num_perm, num_bits)
		m2, s2 = _run_acc(doc2, num_perm, num_bits)
		j = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
		avg_jaccard += j
		j_e = m1.jaccard(m2)
		err = abs(j - j_e)
		logging.info(f"Jaccard Similarity for identical document with {overlap*100:.2f}% overlap: {wiki_data['train'][i1]['title']}= {j} / estimate={j_e}")
		avg_err += err
	avg_err /= N_TRIALS
	avg_jaccard /= N_TRIALS
	logging.info(f"Average True Jaccard Sim: {avg_jaccard}")
	return avg_err


num_perms = range(10, 512, 20)
num_bits = [1, 2, 3, 4, 8, 12, 16, 32]
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
output = "b_bit_minhash_wikisimple_benchmark.png"

logging.info("> Running performance tests")
card = 200
perf_times = {}
for b in num_bits:
	run_times = [run_perf(card, n, b) for n in num_perms]
	perf_times[b] = run_times


logging.info("> Running accuracy tests")
errors = {}
for b in num_bits:
	errs = [run_acc(n, b) for n in num_perms]
	errors[b] = errs

logging.info("> Plotting result")
fig, axe = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
ax = axe[1]
for i, b in enumerate(num_bits):
	ax.plot(
		num_perms, perf_times[b], marker="+", color=bit_colors[i], label=f"{b} bits"
	)
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Running time (sec)")
ax.set_title("MinHash performance")
ax.grid()
ax.legend()
ax = axe[0]
for i, b in enumerate(num_bits):
	ax.plot(num_perms, errors[b], marker="+", color=bit_colors[i], label=f"{b} bits")
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Absolute error in Jaccard estimation")
ax.set_title("MinHash accuracy")
ax.grid()
ax.legend()

plt.tight_layout()
fig.savefig(output)
logging.info("Plot saved to %s" % output)
