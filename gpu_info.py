import pycuda.driver as cuda
cuda.init()
dev = cuda.Device(0)

print("Max threads per block:", dev.max_threads_per_block)
print("Max block dim:", dev.max_block_dim_x, dev.max_block_dim_y, dev.max_block_dim_z)
print("Max grid dim:", dev.max_grid_dim_x, dev.max_grid_dim_y, dev.max_grid_dim_z)