# Validation Workflow (Sprint 1)

1. Construye los binarios mínima mente necesarios:
   ```bash
   cmake -S . -B build
   cmake --build build
   ```

2. Genera los artefactos destinados a `validation/results`:
   ```bash
   ./build/fecuda_validator
   ```
   Este programa GPU consume `datasets_txt/CC.txt` y `datasets_txt/EE.txt` y emite los `paths_values_<dataset>_<thr>.json` utilizados por las pruebas.

3. Ejecuta el comparador de referencias:
   ```bash
   python validation/validation.py
   ```
   El script detecta rutas relativas automáticamente, por lo que puede ejecutarse desde la raíz o desde `validation/`.

4. Revisa el resumen de la terminal. Los archivos validados se listan como `[OK]` y cualquier diferencia muestra un diff resumido para depurar rápidamente.
