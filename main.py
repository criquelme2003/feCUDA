import numpy as np
import forgethreads

t = forgethreads.getTensor()

arr = np.from_dlpack(t)

print(arr)
