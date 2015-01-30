CLC_FLAGS := --platform_filter "AMD Accelerated Parallel Processing" --device_type GPU --add_headers
#CLC_FLAGS := -p=0 -d=0
KERNEL_FLAGS := 
CL_FLAGS := "-I ../ -cl-std=CL1.2 -x clc $(KERNEL_FLAGS)"
#CL_FLAGS := "-I ../ -cl-std=CL2.0 $(KERNEL_FLAGS)"
all:
	clcc $(CLC_FLAGS) --cloptions=$(CL_FLAGS) kernels.cl -o kernels.out

profile:
	CodeXLAnalyzer -c Tahiti -k kernel -s CL -b kernels.out
