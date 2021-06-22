COMPILER = mpicxx
FLAGS= -O3

all: Parallel_Tempering.x

Parallel_Tempering.x: Parallel_Tempering.o
	$(COMPILER) Parallel_Tempering.o $(FLAGS) -o Parallel_Tempering.x

Parallel_Tempering.o: Parallel_Tempering.cpp Parallel_Tempering.h
	$(COMPILER) -c Parallel_Tempering.cpp  $(FLAGS) -o Parallel_Tempering.o

clear:
	rm -f *.o *.x

clean:
	./clean.sh
