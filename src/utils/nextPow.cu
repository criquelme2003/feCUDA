
#include <iostream>
#include <cstdint>
using namespace std;

// Verifica si un número es potencia de 2
bool isPow2(int n)
{
    if (n <= 0)
        return false;
    // Un número es potencia de 2 si solo tiene un bit encendido
    // n & (n-1) elimina el bit menos significativo
    return (n & (n - 1)) == 0;
}

// Encuentra la siguiente potencia de 2 (redondeando hacia arriba)
__device__ __host__ int nextPow2(int n) {
    if (n <= 1) return 1;
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    
    return n;
}
