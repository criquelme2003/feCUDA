[Comparativa de benchmarks](https://bench-web-sk1k.vercel.app/)


## Problema: doble free de managed

Estado actual: Python elimina el tensor, pero luego la clase Tensor result no tiene como saber que el dlpack fue eliminado o no por python, ya que el deleter no enciende ninguna flag. 

TODO: ver como modificar el deleter que genera tensoflow sobre el dlpack


SOLUTION: El tensor solo se encarga de eliminar sus datos, es solo una view del dlpack. la liberacion del dlpack es responsabilidad del owner del python.

