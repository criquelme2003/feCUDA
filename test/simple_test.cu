#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>

// Función simple para ejecutar max_min con archivos que TÚ especifiques
void ejecutar_max_min_con_archivos(
    const char *archivo_A, const char *archivo_B,
    const char *salida_min, const char *salida_max,
    int batch, int M, int K, int N)
{
    printf("Ejecutando max_min con:\n");
    printf("  A: %s\n", archivo_A);
    printf("  B: %s\n", archivo_B);
    printf("  Dimensiones: batch=%d, M=%d, K=%d, N=%d\n", batch, M, K, N);

    // TODO: Aquí cargas tus datos desde archivo_A y archivo_B
    // TODO: Ejecutas el kernel
    // TODO: Guardas en salida_min y salida_max

    printf("  Salida min: %s\n", salida_min);
    printf("  Salida max: %s\n", salida_max);
    printf("¡Listo!\n\n");
}

int main()
{

    // Ejemplo 1: CC vs CE
    ejecutar_max_min_con_archivos(
        "datasets_txt/CC.txt",
        "datasets_txt/CE.txt",
        "results/CC_CE_min.txt",
        "results/CC_CE_max.txt",
        10, 16, 16, 16);

    // Ejemplo 2: EE vs reflexive
    ejecutar_max_min_con_archivos(
        "datasets_txt/EE.txt",
        "datasets_txt/reflexive.txt",
        "results/EE_reflexive_min.txt",
        "results/EE_reflexive_max.txt",
        10, 16, 16, 16);

    // Agrega más combinaciones aquí...

    return 0;
}
