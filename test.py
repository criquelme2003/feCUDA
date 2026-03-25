import forgethreads as ft
import numpy as np

rng = np.random.default_rng()
ft.set_verbose(False)


def build_matrix(n, m, d, seed=None):
    rng = np.random.default_rng(seed)
    N = n + m
    M = np.zeros((N, N))

    A = rng.uniform(0, 0.5, size=(n, n))
    np.fill_diagonal(A, 1.0)
    M[:n, :n] = A

    D = rng.uniform(0, 0.5, size=(m, m))
    np.fill_diagonal(D, 1.0)
    M[n:, n:] = D

    M[:n, n:] = rng.uniform(0, 0.5, size=(n, m))

    # Máximo de aristas posibles por nodo (nodo en bloque n tiene más conexiones)
    max_degree   = (n - 1) + m
    total_possible = (n**2 - n) + (n * m) + (m**2 - m)

    # Objetivo: d * n aristas en total
    target_edges = int(round(d * n))
    target_edges = min(target_edges, total_possible)  # no puede superar el máximo

    # Celdas modificables con valor < 0.5
    mod = np.zeros((N, N), dtype=bool)
    mod[:n, :n] = True
    mod[:n, n:] = True
    mod[n:, n:] = True
    np.fill_diagonal(mod, False)

    current_edges = int(np.sum(M[mod] > 0.5))
    k_extra = max(0, target_edges - current_edges)

    candidates = np.argwhere(mod & (M < 0.5))
    k_extra = min(k_extra, len(candidates))

    if k_extra > 0:
        chosen = candidates[rng.choice(len(candidates), size=k_extra, replace=False)]
        M[chosen[:, 0], chosen[:, 1]] = rng.uniform(0.5, 1.0, size=k_extra)

    final = int(np.sum(M[mod] > 0.5))
    # print(f"  max_degree={max_degree}, total_posible={total_possible}")
    # print(f"  target={target_edges}, añadidas={k_extra}, final={final}, "
    #       f"densidad_real={final/n:.3f}")
    return M.reshape(1, N, N).astype(np.float16)
  
  

def run_experimental(n, avg_degree,i):
    m1 = build_matrix(n,n,avg_degree,1111)
    m2 = m1.copy()
    paths, values, max_order = ft.maxmin_reduced(m1, m2, 0.5, 6,False,avg_degree)
    if i ==0:
      print(paths.to_numpy())
    return max_order
  
  
  
ns = [10000]
iterations = 1
avg_degrees = []
proms_by_n = []
for n in ns:
    m = n
    max_d = float(n**2 - n + n*m + m**2 - m) / (n + m)

    # avg_degrees = np.array([0.1*max_d, 0.3*max_d, 0.5*max_d, 0.75*max_d], dtype=np.float16)
    
    avg_degrees = np.arange(int(np.log(n)),int(np.log(n)) + 1,step=0.2,dtype=np.float16) 
    print(avg_degrees)
    proms = []
    for deg in avg_degrees:
        acum = 0
        for i in range(iterations):
        
            acum += run_experimental(n, float(deg),i)
        proms.append(acum / iterations)
        print(f"deg={deg} : n={n} ", end="\t")
    print(f"\ncalculated averages for {n} nodes")
    proms_by_n.append(proms)

# p300 = np.load("proms_300n.npy")
# p400 = np.load("proms_400n.npy")
# p500 = np.load("proms_500n.npy")

# proms_by_n.append(p300[0]) E(eta_0) < b log(N), 
# proms_by_n.append(p400[0])
# proms_by_n.append(p500[0])

print(proms_by_n)