CC = gcc
CFLAGS = -Wall -O3 -std=c99
LDFLAGS = -lm

TARGET = kmeans_serial
SRC = kmeans_serial.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	del /Q $(TARGET).exe 2>nul || rm -f $(TARGET)

# Test run with sample data
test: $(TARGET)
	@echo Running serial K-means test...
	./$(TARGET) ../datasets/data_10k.txt 5 100

.PHONY: all clean test
