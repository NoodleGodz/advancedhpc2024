import numba.cuda

numba.cuda.detect()
 
device = numba.cuda.get_current_device()


device_name = device.name
free_mem, total_mem = numba.cuda.current_context().get_memory_info()


total_mem_MB = total_mem / (1024 ** 2)
multiprocessor_count = device.MULTIPROCESSOR_COUNT

print(f"Device Id: {device.id}")
print(f"Device Name: {device_name}")
print(f"Multiprocessor Count: {multiprocessor_count}")
print(f"Total cores Count: {multiprocessor_count*128}")
print(f"Total memory: {total_mem_MB:.2f} MB")


from numba.cuda.cudadrv import enums
attribs= [name.replace("CU_DEVICE_ATTRIBUTE_", "") for name in dir(enums) if name.startswith("CU_DEVICE_ATTRIBUTE_")]
for attr in attribs:
    print(attr, '=', getattr(device, attr))


