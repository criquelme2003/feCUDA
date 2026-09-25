import forgethreads as ft 

result = ft.path_dag('csr_bin_ties', 10, 9)
print(type(result), len(result))
for k in result.keys():
  print(k, result[k])
