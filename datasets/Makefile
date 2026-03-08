CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

TARGET = data_generator
SRC = data_generator.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	del /Q $(TARGET).exe 2>nul || rm -f $(TARGET)

# Generate sample datasets
generate_samples: $(TARGET)
	@echo Generating sample datasets...
	./$(TARGET) 10000 5 2 data_10k.txt
	./$(TARGET) 100000 8 2 data_100k.txt
	./$(TARGET) 500000 10 2 data_500k.txt
	@echo Done!

.PHONY: all clean generate_samples
