CC = gcc
CFLAGS = -O2 -Wall -lm
OBJS = main.o hc.o

all: hypercube

hypercube: $(OBJS)
	$(CC) $(OBJS) -lm -o hypercube

main.o: main.c hc.h
	$(CC) -c main.c

hc.o: hc.c hc.h
	$(CC) -c hc.c

clean:
	rm -f *.o hypercube
