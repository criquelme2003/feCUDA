#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


// Tu tipo de dato "valor"
struct Record {
    int id;       // este campo será la clave
    float payload;
};

void try_cuco() {
    using Key   = int;
    using Value = Record*;   // <-- almacenamos un puntero como valor

    // Sentinels: valores reservados que NUNCA deben aparecer como clave/valor real
    constexpr Key   empty_key_sentinel   = -1;
    constexpr Value empty_value_sentinel = nullptr;

    // 1. Datos "reales" en device
    thrust::device_vector<Record> records = { {10, 1.1f}, {20, 2.2f}, {30, 3.3f} };
    Record* d_records = thrust::raw_pointer_cast(records.data());

    // 2. Construimos los pares {clave, puntero} a partir de cada valor
    thrust::device_vector<cuco::pair<Key, Value>> pairs(records.size());

    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(records.size()),
        pairs.begin(),
        [d_records] __device__ (int i) {
            Record* ptr = d_records + i;      // puntero al elemento
            Key k = ptr->id;                  // clave derivada del valor (campo id)
            return cuco::pair<Key, Value>{k, ptr};
        });

    // 3. Creamos el static_map con capacidad suficiente
    auto map = cuco::static_map{
        cuco::extent<std::size_t>{records.size() * 2},  // capacidad (con margen, load factor < 0.5-0.7 recomendado)
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel}
    };

    // 4. Insertamos en bulk
    map.insert(pairs.begin(), pairs.end());

    // 5. Buscar (find) por clave, obtener el puntero
    thrust::device_vector<Key> queries = {20, 99};   // 99 no existe
    thrust::device_vector<Value> results(queries.size());

    map.find(queries.begin(), queries.end(), results.begin());

    // results[0] -> puntero al Record con id=20
    // results[1] -> empty_value_sentinel (nullptr), porque no se encontró

}