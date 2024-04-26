NVCC = NVCC

CFLAGS = -I/usr/lib/x86_64-linux-gnu/openmpi/include \
         -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi \
         -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
         -lmpi \
         -lcufft \
         -std=c++17

TARGET = MPC

SRC = MPC.c

all: $(TARGET)

$(TARGET): $(SRC) 
    $(NVCC) $(SRC) $(CFLAGS) - $(TARGET)

clean:
    rm -f $(TARGET)

.PHONY: all clean
