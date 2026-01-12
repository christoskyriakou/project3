CC = gcc
CFLAGS = -Wall -O3 -march=native -std=c99
LDFLAGS = -lm

TARGET = search
SOURCES = search.c dataload.c kmeans.c ivfpq.c
HEADERS = dataload.h kmeans.h ivfpq.h

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o output.txt

run_mnist: $(TARGET)
	./$(TARGET) -d input.dat -q query.dat -type mnist -kclusters 50 \
		-nprobe 5 -N 10 -R 2000 -seed 1 -o output.txt -ivfflat -range true

run_sift: $(TARGET)
	./$(TARGET) -d input.dat -q query.dat -type sift -kclusters 100 \
		-nprobe 10 -N 10 -R 2.0 -seed 1 -o output.txt -ivfflat -range true

test: $(TARGET)
	./$(TARGET) -d input.dat -q query.dat -type mnist -kclusters 20 \
		-nprobe 3 -N 5 -R 2000 -seed 1 -o output.txt -ivfflat

.PHONY: all clean run_mnist run_sift test

# ./search -d res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte -type mnist -kclusters 50 -nprobe 5 -N 10 -R 2000 -o output.txt -ivfflat
# ./search -d res/t10k-images.idx3-ubyte -q res/train-images.idx3-ubyte -type mnist -kclusters 50 -nprobe 5 -N 10 -R 2000 -o output.txt -ivfflat

#./search -d res/sift_base.fvecs -q res/sift_query.fvecs -kclusters 100 -nprobe 5 -o results.txt -N 10 -R 2 -type sift -range true -seed 1 -ivfflat
#./search -d res/sift_base.fvecs -q res/sift_query.fvecs -kclusters 100 -nprobe 5 -o results.txt -N 10 -R 2 -type sift -seed 1 -ivfflat

# valgrind ./search -d res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte -o results.txt -type mnist -kclusters 50 -nprobe 5 -M 16 -nbits 8 -N 10 -R 2000 -ivfpq
# valgrind ./search -d res/sift_base.fvecs -q res/sift_query.fvecs -o results.txt -type sift -kclusters 50 -nprobe 10 -M 8 -nbits 8 -N 10 -R 2.0 -ivfpq 
# ./search -d res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte -o results.txt -type mnist -kclusters 50 -nprobe 10 -M 32 -nbits 8 -N 5 -R 2000 -ivfpq