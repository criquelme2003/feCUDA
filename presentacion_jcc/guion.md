# Guion: JCC 2026 (≈ 7:35, tope 8:00)

Reparto aproximado: ingeniería ≈ 3:20 (diapositivas 2–6), ciencia ≈ 3:50 (diapositivas 7–12).

---

### 1. Portada · 0:10
- Coautores y tema: profundidad de propagación en efectos olvidados y la herramienta GPU que permite estudiarla.

### 2. El problema · 0:35
- **Idea central:** un efecto olvidado es una influencia indirecta que no está en la matriz y aparece a través de un camino (ejemplo i → k → j).
- La búsqueda avanza orden a orden y cuenta cuántos pares cambian de forma significativa (q_r).
- Se detiene cuando un orden ya no aporta nada. El último orden útil es η₀.
- **Cierre:** las dos preguntas que ordenan la charla, cómo escalar el cálculo y de qué depende η₀.

### 3. La composición max–min · 0:35
- Lectura de la fórmula: mínimo = arista más débil del camino; máximo = mejor camino intermedio.
- Recorrer el ejemplo de la celda: se eligen los mínimos y luego el máximo (0,7).
- La diagonal en 1 hace que la composición sea acumulativa.
- **Puente a lo que sigue:** es un producto de matrices con otras operaciones, por eso valen las técnicas de GEMM. Costo O(N³) por orden.

### 4. Arquitectura · 0:35
- Tres componentes: Python, bindings y motor CUDA; todo vive en la memoria de la GPU.
- DLPack y zero-copy: los tensores se comparten por puntero, sin copias host–GPU.
- El usuario hace una sola llamada y recibe los efectos, q_r y η₀.
- **Énfasis:** la memoria de trabajo es cuadrática.

### 5. Kernel · 0:45
- Punto de partida: el patrón de un GEMM optimizado (tiles en memoria compartida, varias celdas por hilo en registros).
- Usar la tabla para mostrar qué cambia: operación (mínimo), acumulador (máximo) y epílogo.
- **Aporte propio:** el epílogo detecta y cuenta los descubrimientos en la misma pasada, con un contador atómico. q_r sale sin otro kernel.

### 6. Rendimiento y límites · 0:50
- forgeffects como implementación útil en TensorFlow, limitada por su memoria cúbica.
- Evolución del kernel:
  - v1 (bloque por celda) es competitiva en tamaños chicos, pero sin reúso de datos pierde frente a TensorFlow en N = 1200.
  - El tiling (v2, v4) es lo que da la ventaja; v4 rinde mejor en tamaños grandes.
- Cifras a mencionar: 12,8–44× (v2), 14–25× (v4).
- **Límite del dispositivo:** forgeffects se queda sin memoria desde 1300 nodos; forgethreads llega a 32 000.
- **Transición:** esta capacidad es la que habilita el estudio experimental.

### 7. Profundidad de parada: un recurso finito · 0:30
- Recordar las definiciones de q_r y η₀.
- Con ε = 1/2, cada par se descubre una sola vez, porque dos saltos superarían 1.
- Leer el esquema: los descubrimientos se reparten entre órdenes y η₀ es donde aparece el último.

### 8. Diseño experimental · 0:45
- Dos bloques, N y M, con influencia solo de N hacia M; matrices binarias.
- **Punto clave:** c es el grado por fila, pero se reparte distinto. Las filas de M ponen todo en M; las de N lo dividen en mitades.
- Por eso cada bloque llega a grado 1 en un c distinto: M en c = 1, N en c ≈ 2. El bloque NM es un puente sin ciclos y no aporta un cruce propio.
- Dimensión del barrido: 2640 grafos.

### 9. Resultado: perfil de dos ventanas · 0:35
- **Mensaje:** la profundidad no crece con la conectividad.
- Recorrer un panel grande: bajo, pico, valle, pico, caída.
- Notar que los picos caen junto a las líneas c = 1 y c = 2.
- Las barras son intervalos de confianza al 95 %.
- **Pregunta abierta para la siguiente:** ¿por qué ese perfil?

### 10. Cinco regímenes · 0:45
- Interpretar el perfil con los cruces de cada bloque.
- Recorrer los regímenes de izquierda a derecha con su profundidad media:
  - baja conectividad: cadenas cortas;
  - ventana M: M en su umbral;
  - valle: M ya tiene atajos y N aún no llega;
  - ventana N: N en su umbral;
  - atajos: todo se descubre temprano.
- **Idea a dejar:** lo que importa es el grado de cada bloque, no el promedio global.

### 11. Efecto del tamaño · 0:50
- **Izquierda:** la diferencia ventana − valle es negativa o nula en grafos chicos y positiva desde n = 2500, cada vez mayor (≈ +24 órdenes en n = 10 000).
- **Derecha:** cuánto crece η₀ con el tamaño. Es casi 3 veces más en las ventanas que en el valle, y casi nada con muchos atajos.
- **Matiz:** ninguna de las dos ventanas domina a la otra.

### 12. Conclusiones · 0:35
- Repasar los cuatro mensajes numerados: perfil no monótono, picos por bloque, efecto del tamaño y forgethreads como habilitador.
- Trabajo futuro en una frase:
  - medir η₀ por bloque, para atribuir cada ventana;
  - pasar a pesos difusos y otras composiciones;
  - formalizar los regímenes;
  - kernels dispersos.
- Aclarar el alcance: son resultados descriptivos de este diseño.

### 13. Gracias · 0:05
- Abrir a preguntas.
