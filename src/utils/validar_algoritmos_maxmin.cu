#include "../../include/utils.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Función para validar algoritmos maxmin contra archivos de referencia
void validar_algoritmos_maxmin(FuncionMaxMin funcion_maxmin, const char *nombre_algoritmo)
{

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║          VALIDACIÓN DE ALGORITMO: %-25s ║\n", nombre_algoritmo);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // Estructura con los casos de prueba
    struct CasoPrueba
    {
        const char *nombre;
        const char *archivo_tensor;
        const char *archivo_referencia;
        int batch, M, N, K;
    };

    CasoPrueba casos[] = {
        {"Reflexive vs Reflexive", "../datasets_txt/reflexive.txt", "../results/reflexive_min.txt", 1, 6, 6, 1},
        {"CC vs CC", "../datasets_txt/CC.txt", "../results/CC_min.txt", 10, 16, 16,1},
        {"CC vs CC", "../datasets_txt/EE.txt", "../results/EE_min.txt", 10, 16, 16,1}        

    };

    int num_casos = sizeof(casos) / sizeof(casos[0]);
    int casos_exitosos = 0;

    for (int i = 0; i < num_casos; i++)
    {
        printf("───────────────────────────────────────────────────────────────\n");
        printf("CASO %d/3: %s\n", i + 1, casos[i].nombre);
        printf("───────────────────────────────────────────────────────────────\n");

        // Cargar tensor de entrada
        TensorResult tensor_entrada;
        bool carga_ok = leer_matriz_3d_desde_archivo(casos[i].archivo_tensor, tensor_entrada,
                                                     casos[i].batch, casos[i].M, casos[i].N, casos[i].K);

        if (!carga_ok)
        {
            printf("❌ ERROR: No se pudo cargar %s\n", casos[i].archivo_tensor);
            printf("   Verifica que el archivo existe y tiene el formato correcto.\n\n");
            continue;
        }

        printf("✅ Tensor de entrada cargado: %s\n", casos[i].archivo_tensor);
        printf("   Dimensiones: batch=%d, M=%d, N=%d, K=%d\n",
               tensor_entrada.batch, tensor_entrada.M, tensor_entrada.N, tensor_entrada.K);

        // Ejecutar algoritmo maxmin
        printf("🚀 Ejecutando algoritmo %s...\n", nombre_algoritmo);
        auto inicio = std::chrono::high_resolution_clock::now();

        TensorResult resultado = funcion_maxmin(tensor_entrada, tensor_entrada);

        auto fin = std::chrono::high_resolution_clock::now();
        double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();

        if (resultado.data == nullptr)
        {
            printf("❌ ERROR: El algoritmo devolvió un tensor nulo\n\n");
            safe_tensor_cleanup(tensor_entrada);
            continue;
        }

        printf("✅ Algoritmo ejecutado en %.3f ms\n", tiempo_ms);
        printf("   Resultado: batch=%d, M=%d, N=%d, K=%d\n",
               resultado.batch, resultado.M, resultado.N, resultado.K);

        // Cargar tensor de referencia
        TensorResult tensor_referencia;
        bool ref_ok = leer_matriz_3d_desde_archivo(casos[i].archivo_referencia, tensor_referencia,
                                                   resultado.batch, resultado.M, resultado.N, resultado.K);

        if (!ref_ok)
        {
            printf("⚠️  ADVERTENCIA: No se pudo cargar archivo de referencia %s\n", casos[i].archivo_referencia);
            printf("   El algoritmo se ejecutó, pero no se puede validar el resultado.\n");
            printf("   Considera generar primero los archivos de referencia.\n\n");

            safe_tensor_cleanup(tensor_entrada);
            safe_tensor_cleanup(resultado);
            continue;
        }

        printf("✅ Archivo de referencia cargado: %s\n", casos[i].archivo_referencia);

        // Comparar resultados
        printf("🔍 Comparando resultado con referencia...\n");
        bool son_iguales = comparar_tensores(resultado, tensor_referencia, 1e-5f, false);

        if (son_iguales)
        {
            printf("🎉 ¡ÉXITO! El resultado coincide con la referencia\n");
            casos_exitosos++;
        }
        else
        {
            printf("❌ FALLO: El resultado difiere de la referencia\n");
            printf("   Ejecutando comparación detallada...\n\n");
            comparar_tensores(resultado, tensor_referencia, 1e-5f, true);
        }

        // Limpiar memoria
        safe_tensor_cleanup(tensor_entrada);
        safe_tensor_cleanup(resultado);
        safe_tensor_cleanup(tensor_referencia);

        printf("\n");
    }

    // Resumen final
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                      RESUMEN FINAL                          ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Algoritmo: %-49s ║\n", nombre_algoritmo);
    printf("║ Casos probados: %d/3                                        ║\n", num_casos);
    printf("║ Casos exitosos: %d/3                                        ║\n", casos_exitosos);

    if (casos_exitosos == num_casos)
    {
        printf("║ Estado: ✅ TODOS LOS CASOS PASARON                         ║\n");
    }
    else
    {
        printf("║ Estado: ❌ ALGUNOS CASOS FALLARON                          ║\n");
    }

    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}
