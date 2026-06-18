"""Analyze best basis to understand linf=18 barrier and find cancellation opportunities."""
import numpy as np
import ast, os

BASE = os.path.dirname(os.path.abspath(__file__))
P3_DIR = os.path.join(BASE, "crypto1_repo", "第三题")

with open(os.path.join(P3_DIR, "problem3.txt")) as f:
    A_mat = np.array(ast.literal_eval(f.read().strip().split('\n')[0].split('=',1)[1].strip()), dtype=np.int64)
M, N = A_mat.shape; DIM = M + N; Q = int(np.max(A_mat)) + 1

bp = os.path.join(P3_DIR, "results_p3", "g6k_push_b95_p3.npy")
B_np = np.load(bp).astype(np.int64)

def centered_u(u_raw):
    u = u_raw % Q
    return np.where(u >= Q//2, u - Q, u).astype(np.int64)

def true_linf(vec):
    v = vec[:M]
    u_raw = vec[M:]
    u = centered_u(u_raw)
    lv = int(np.max(np.abs(v)))
    lu = int(np.max(np.abs(u)))
    return max(lv, lu), lv, lu

# Find best basis row
best_row = -1
best_linf = 999
best_vec = None
for i in range(DIM):
    vec = B_np[i, :]
    if np.all(vec[:M] == 0):
        continue
    linf, lv, lu = true_linf(vec)
    if linf < best_linf:
        best_linf = linf
        best_row = i
        best_vec = vec.copy()

print("=" * 60)
print("Best basis row: %d" % best_row)
linf, lv, lu = true_linf(best_vec)
print("linf=%d lv=%d lu=%d" % (linf, lv, lu))

v_best = best_vec[:M]
u_best_raw = best_vec[M:]
u_best = centered_u(u_best_raw)

# Find coordinates with large absolute values
print("\n--- Large coordinates in best row ---")
print("v part (first 120):")
large_v = []
for j in range(M):
    if abs(v_best[j]) >= 14:
        large_v.append((j, v_best[j]))
        print("  v[%3d] = %d" % (j, v_best[j]))

print("\nu part (last 120, centered):")
large_u = []
for j in range(N):
    if abs(u_best[j]) >= 14:
        large_u.append((j, u_best[j]))
        print("  u[%3d] = %d" % (j, u_best[j]))

# Check: are the large coordinates clustered?
print("\n--- Statistics ---")
print("|v| distribution: min=%d p50=%d p90=%d p95=%d p99=%d max=%d" % tuple(
    int(np.percentile(np.abs(v_best), p)) for p in [0, 50, 90, 95, 99, 100]))
print("|u| distribution: min=%d p50=%d p90=%d p95=%d p99=%d max=%d" % tuple(
    int(np.percentile(np.abs(u_best), p)) for p in [0, 50, 90, 95, 99, 100]))

# Count how many coords are above threshold
for t in [12, 14, 16, 18]:
    nv = np.sum(np.abs(v_best) >= t)
    nu = np.sum(np.abs(u_best) >= t)
    print("|coord| >= %d: v=%d u=%d total=%d" % (t, nv, nu, nv+nu))

# Find rows that have opposite-sign values at the large-coordinate positions
print("\n--- Cancellation candidates ---")
other_rows = []
for i in range(DIM):
    if i == best_row:
        continue
    vec = B_np[i, :]
    if np.all(vec[:M] == 0):
        continue
    v = vec[:M]
    u_raw = vec[M:]
    u = centered_u(u_raw)

    # Score: how many large coordinates have opposite sign
    cancel_score = 0
    for j, val in large_v:
        if val * v[j] < 0:  # Opposite signs
            cancel_score += 1
    for j, val in large_u:
        if val * u[j] < 0:
            cancel_score += 1

    linf_i, lv_i, lu_i = true_linf(vec)
    if cancel_score >= 1:
        other_rows.append((cancel_score, linf_i, lv_i, lu_i, i, v.copy(), u.copy()))

other_rows.sort(key=lambda x: -x[0])

print("Rows with cancel_score >= 1 (best first):")
for score, linf_i, lv_i, lu_i, i, v, u in other_rows[:15]:
    print("  Row %3d: cancel_score=%d linf=%d lv=%d lu=%d" % (i, score, linf_i, lv_i, lu_i))

# Try specific combinations
print("\n--- Try 1*best_row + k*other_row ---")
for score, linf_i, lv_i, lu_i, i, v_i, u_i in other_rows[:5]:
    for k in [-2, -1, 1, 2]:
        combined = best_vec + k * B_np[i, :]
        linf_c, lv_c, lu_c = true_linf(combined)
        marker = " ***" if linf_c < best_linf else ""
        print("  best(row %d) + %d*row%d: linf=%d lv=%d lu=%d%s" % (best_row, k, i, linf_c, lv_c, lu_c, marker))

# Try 2*k combinations of top 3 cancellation rows
print("\n--- Try linear combos of top cancellation rows ---")
top_rows = [idx for _, _, _, _, idx, _, _ in other_rows[:3]]
for a in range(-2, 3):
    for b in range(-2, 3):
        for c in range(-2, 3):
            if a == 0 and b == 0 and c == 0:
                continue
            combined = (1 * best_vec +
                       a * B_np[top_rows[0], :] +
                       b * B_np[top_rows[1], :] +
                       c * B_np[top_rows[2], :])
            linf_c, lv_c, lu_c = true_linf(combined)
            if linf_c < best_linf:
                print("  best + %d*row%d + %d*row%d + %d*row%d: linf=%d lv=%d lu=%d" %
                      (a, top_rows[0], b, top_rows[1], c, top_rows[2], linf_c, lv_c, lu_c))

print("\nDone.")
