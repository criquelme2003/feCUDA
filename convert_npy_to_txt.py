#!/usr/bin/env python3
"""
Script para convertir archivos .npy a formato de texto plano
que puede ser leído por la función leer_matriz_3d_desde_archivo()
"""

import numpy as np
import os
import sys


def convertir_npy_a_txt(archivo_npy, archivo_txt):
    """
    Convierte un archivo .npy a formato de texto plano

    Args:
        archivo_npy (str): Ruta al archivo .npy de entrada
        archivo_txt (str): Ruta al archivo .txt de salida
    """
    try:
        # Cargar el archivo .npy
        datos = np.load(archivo_npy)

        print(f"Cargando {archivo_npy}...")
        print(f"Forma del array: {datos.shape}")
        print(f"Tipo de datos: {datos.dtype}")
        print(f"Rango de valores: [{datos.min():.6f}, {datos.max():.6f}]")

        # Verificar si es 3D o si necesita ser expandido
        if len(datos.shape) == 2:
            # Si es 2D, expandir a 3D agregando una dimensión K=1
            datos = np.expand_dims(datos, axis=-1)
            print(f"Array expandido a 3D: {datos.shape}")
        elif len(datos.shape) == 1:
            # Si es 1D, necesitamos saber cómo interpretarlo
            print("Advertencia: Array 1D detectado. Se interpretará como (1, 1, len)")
            datos = datos.reshape(1, 1, -1)
            print(f"Array reestructurado: {datos.shape}")
        elif len(datos.shape) > 3:
            print(
                f"Error: Array con {len(datos.shape)} dimensiones no soportado")
            return False

        batch, M, N = datos.shape

        # Escribir al archivo de texto
        with open(archivo_txt, 'w') as f:
            # Escribir encabezado con dimensiones
            f.write(f"# Convertido desde: {os.path.basename(archivo_npy)}\n")
            f.write(f"# Dimensiones: batch={batch}, M={M}, N={N}\n")
            f.write(f"# Rango: [{datos.min():.6f}, {datos.max():.6f}]\n")

            # Escribir datos
            for b in range(batch):
                f.write(f"# Batch {b}\n")
                for i in range(M):
                    for j in range(N):
                        f.write(f"{datos[b, i, j]:.8f}")
                        if j < N - 1:
                            f.write(" ")
                    f.write("\n")
                if b < batch - 1:
                    f.write("\n")  # Separación entre batches

        print(f"Archivo convertido exitosamente: {archivo_txt}")
        return True

    except Exception as e:
        print(f"Error al convertir {archivo_npy}: {e}")
        return False


def main():
    # Directorio con archivos .npy
    directorio_npy = "/home/carlos/feCUDA/forgeffects/forgeffects/dataset"

    # Directorio de salida
    directorio_salida = "/home/carlos/feCUDA/datasets_txt"

    # Crear directorio de salida si no existe
    os.makedirs(directorio_salida, exist_ok=True)

    # Lista de archivos .npy a convertir
    archivos_npy = ["CC.npy", "CE.npy", "EE.npy","reflexive.npy"]

    print("=== Convertidor de archivos .npy a .txt ===\n")

    convertidos = 0
    for archivo in archivos_npy:
        ruta_npy = os.path.join(directorio_npy, archivo)
        ruta_txt = os.path.join(
            directorio_salida, archivo.replace('.npy', '.txt'))

        if os.path.exists(ruta_npy):
            if convertir_npy_a_txt(ruta_npy, ruta_txt):
                convertidos += 1
            print()  # Línea en blanco entre archivos
        else:
            print(f"Archivo no encontrado: {ruta_npy}")

    print(f"=== Conversión completada ===")
    print(f"Archivos convertidos: {convertidos}/{len(archivos_npy)}")
    print(f"Archivos guardados en: {directorio_salida}")


if __name__ == "__main__":
    main()
