import subprocess as sp

def test_fe(dim) -> bool:  
  res = sp.run(
    ["python","memory_fe.py",str(dim)],
  )
  
  match res.returncode:
    case 0:
      return True
    case 1:
      return False
    case _:
      raise ValueError("External error")
    
def test_ft(dim) -> bool:  
  res = sp.run(
    ["python","memory_ft.py",str(dim)],
  )
  
  match res.returncode:
    case 0:
      return True
    case 1:
      return False
    case _:
      return False


def to_pair(n:int) -> int:
  f = n//2
  return 2*f

def find_sizebounds(init_dim=100,exp_factor = 2,tol=10,abs_limit=100000,test_dim=test_fe) -> tuple[object,object]:
  dim_ok = -1
  dim_fail = -1
  dim = init_dim
  #---- exponential seach -------
  while(dim_fail == - 1):
    success = test_dim(dim)
    if(success):
      dim_ok =dim
      dim = dim * exp_factor
      dim = to_pair(dim)
    else:
      dim_fail = dim

    if dim > abs_limit:
      return dim_ok,-1

  if(dim_ok == -1):
    print("init dim not accepted")

  #----- bisección ------
  while(dim_fail - dim_ok > tol):
    mid = (dim_ok + dim_fail) // 2
    mid = to_pair(mid)

    if mid == dim_ok or mid == dim_fail:
      break

    success = test_dim(mid)

    if(success):
      dim_ok = mid
    else:
      dim_fail = mid

  return dim_ok,dim_fail


# max_dim_fe,_  = find_sizebounds(init_dim=100,test_dim=test_fe)
max_dim_ft,_  = find_sizebounds(init_dim=25600,test_dim=test_ft)

print(f"max dim finded for forgethreads: {max_dim_ft}")