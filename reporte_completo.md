# Reporte Completo del Proyecto feCUDA

Generado el: Mon Apr 20 13:21:32 UTC 2026

## Sanitización - Memcheck
```
========= COMPUTE-SANITIZER
GPU: sm_75 (GTX 1650)
Validando MaxMin: comparando GPU vs CPU reference, 3 runs por config


=== B=1  M=4  thr=0.30 ===
  CPU reference: 0 paths
Executing maxmin reduced
[MAXMIN_REDUCED] avg_n=0.1 order=1 N=4
[MAXMIN_REDUCED] Convergencia en step 1 (sin nuevos caminos)
[MAXMIN_REDUCED] Caminos reconstruidos (effective_order=0): 0
  Run 0: GPU=0  OK
Executing maxmin reduced
[MAXMIN_REDUCED] avg_n=0.1 order=1 N=4
[MAXMIN_REDUCED] Convergencia en step 1 (sin nuevos caminos)
[MAXMIN_REDUCED] Caminos reconstruidos (effective_order=0): 0
  Run 1: GPU=0  OK
Executing maxmin reduced
[MAXMIN_REDUCED] avg_n=0.1 order=1 N=4
[MAXMIN_REDUCED] Convergencia en step 1 (sin nuevos caminos)
```

