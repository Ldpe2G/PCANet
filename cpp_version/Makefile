ROOTDIR = $(CURDIR)
CC=g++
CFLAGS += -O0 -g3 -Wall -c -fmessage-length=0 -fopenmp
LDFLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann

# If you do not know where your opencv files are, open the Terminal and type:
#   pkg-config --cflags opencv
OPENCV_PATH = /usr/include/opencv
OPENCV_LIB = /usr/lib

all: utils.o main.o PCANet


utils.o: utils.cpp
	$(CC) -I$(OPENCV_PATH) $(CFLAGS) -o utils.o utils.cpp
main.o: main.cpp
	$(CC) -I$(OPENCV_PATH) $(CFLAGS) -o main.o main.cpp
PCANet: main.o utils.o
	$(CC) -L$(OPENCV_LIB) -fopenmp -o "PCANet"  main.o utils.o  $(LDFLAGS)
clean:
	rm *.o PCANet


