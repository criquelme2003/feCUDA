import numpy as np
import time
import matplotlib.pyplot as plt

def mutual_outlinks_A(links):
    nr, nc = links.shape
    tot = 0
    for i in range(nr - 1):
        for j in range(i + 1, nr):
            for k in range(nc):
                tot += links[i, k] * links[j, k]
    return tot / (nr * (nr - 1) / 2)

def mutual_outlinks_B(links):
    nr, nc = links.shape
    T = 0.0
    for i in range(nr - 1):
        tmp = np.dot(links[i+1:, :], links[i, :].T)
        T += np.sum(tmp)
    p = (nr * (nr - 1)) / 2
    return T / p

def simulate_and_compare():
    sizes = [(5,5), (50,50), (500,500), (1000,1000)]
    iterations = 10
    times_A = []
    times_B = []
    
    for nr, nc in sizes:
        total_time_A = 0.0
        total_time_B = 0.0
        
        for _ in range(iterations):    
            links = np.random.choice([0, 1], size=(nr*nc), replace=True).reshape(nr, nc)
            
            # Medir solo el tiempo de A
            start_time_A = time.time()
            result1 = mutual_outlinks_A(links)
            total_time_A += time.time() - start_time_A
            
            # Medir solo el tiempo de B
            start_time_B = time.time()
            result2 = mutual_outlinks_B(links)
            total_time_B += time.time() - start_time_B
            
        times_A.append(total_time_A / iterations)
        times_B.append(total_time_B / iterations)
        
        print(f"Matriz {nr}x{nc} procesada.")
        
    print("\nTiempos promedio para mutual_outlinks_A:", times_A)
    print("Tiempos promedio para mutual_outlinks_B:", times_B)
    
    etiquetas = ['5x5', '50x50', '500x500', '1000x1000']

    plt.figure(figsize=(10, 6))
    plt.plot(etiquetas, times_A, marker='o', color='red', label='Código A (Ciclos For)')
    plt.plot(etiquetas, times_B, marker='s', color='blue', label='Código B (Numpy Vectorizado)')

    plt.title('Comparación de Tiempos: Código A vs Código B')
    plt.xlabel('Dimensión de la Matriz')
    plt.ylabel('Tiempo Promedio (segundos)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Usar escala logarítmica es clave aquí para que la línea del Código B no se vea plana
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('plott.png')

simulate_and_compare()
