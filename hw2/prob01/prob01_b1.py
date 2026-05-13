import numpy as np
import time

weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
values  = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
W = 165
n = len(weights)

# ── Brute-Force ──────────────────────────────────────────────────────────────
def sum_by_bits(bitstring, list):
    return sum(list[i] for i, c in enumerate(bitstring) if c == '1')

t0 = time.perf_counter()
best_value = 309
best_lambd, best_weight, best_bits, total_valiue = 0xFFFF, 0xFFFF, 0xFFFF, ''
for case in range(2**n):
    bits = np.binary_repr(case, n)
    tw = sum_by_bits(bits, weights)
    tv = sum_by_bits(bits, values)
    if tw > W and tv > best_value:
        tl = (tv - best_value)/((tw - W)**2) 
        if tw < best_weight:
            best_lambd, best_weight, total_valiue, best_bits = tl, tw, tv, bits
t1 = time.perf_counter() - t0

selected_items = [i+1 for i, c in enumerate(best_bits) if c == '1']
print(f"Optimal bitstring : {best_bits}")
print(f"Selected items    : {selected_items}")
print(f"Total weight      : {best_weight}")
print(f"Total value       : {total_valiue}")
print(f"Total lambda      : {best_lambd}")
print(f"Time              : {t1*1000:.2f} ms")