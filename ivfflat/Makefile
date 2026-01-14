CC = gcc
CFLAGS = -O3 -march=native -Wall

OBJS = kmeans.o ivfflat.o dataload.o ivfflat_knn.o

all: ivfflat_knn

ivfflat_knn: $(OBJS)
	$(CC) $(CFLAGS) -o ivfflat_knn $(OBJS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o ivfflat_knn