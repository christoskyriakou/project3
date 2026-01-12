CC = gcc
CFLAGS = -Wall -O3 -march=native -std=c99
LDFLAGS = -lm

TARGET = search
SOURCES = search.c kmeans.c ivfflat.c dataload.c
HEADERS = keamns.h ivfflat.h dataload.h

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o output.txt

run_mnist: $(TARGET)
	./$(TARGET) -d res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte -type mnist -kclusters 50 \
		-nprobe 5 -N 10 -R 2000 -seed 1 -o output.txt -ivfflat -range true

run_sift: $(TARGET)
	./$(TARGET) -d res/sift_base.fvecs -q res/sift_query.fvecs -type sift -kclusters 100 \
		-nprobe 10 -N 10 -R 2.0 -seed 1 -o output.txt -ivfflat -range true

test: $(TARGET)
	./$(TARGET) -d input.dat -q query.dat -type mnist -kclusters 20 \
		-nprobe 3 -N 5 -R 2000 -seed 1 -o output.txt -ivfflat

.PHONY: all clean run_mnist run_sift test

# valgrind ./search -d res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte -type mnist -kclusters 50 -nprobe 5 -N 10 -R 2000 -o output.txt -ivfflat
# ./search -d res/t10k-images.idx3-ubyte -q res/train-images.idx3-ubyte -type mnist -kclusters 50 -nprobe 5 -N 10 -R 2000 -o output.txt -ivfflat

# valgrind ./search -d res/sift_base.fvecs -q res/sift_query.fvecs -kclusters 50 -nprobe 5 -o results.txt -N 10 -R 2 -type sift -range true -seed 1 -ivfflat
#./search -d res/sift_base.fvecs -q res/sift_query.fvecs -kclusters 50 -nprobe 5 -o results.txt -N 10 -R 2 -type sift -seed 1 -ivfflat
