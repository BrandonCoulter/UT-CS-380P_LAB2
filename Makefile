CC = g++ 
SRCS = ./src/*.cpp
INC = ./src/
NEW_OPTS = -std=c++17 -Wall -Werror -O3
OPTS = -std=c++17 -Wall -Werror -lpthread -O3
DOPTS = -std=c++17 -Wall -Werror -lpthread -O0 -g -fsanitize=address
PRINT = -D__PRINT__
VERBOSE = -D__VERBOSE__

EXEC = bin/kmeans

all: clean compile
print: clean pcompile
debug: clean dcompile

compile:
	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

pcompile:
	$(CC) $(SRCS) $(OPTS) ${PRINT} -I$(INC) -o $(EXEC)

dcompile:
	$(CC) $(SRCS) $(DOPTS) ${PRINT} ${VERBOSE} -I$(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)