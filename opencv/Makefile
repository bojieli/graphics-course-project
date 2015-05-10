OPENCV_DIR=/cygdrive/d/Application/opencv
CFLAGS=-Wall -g -O3 -I$(OPENCV_DIR)/include
LDFLAGS=-L$(OPENCV_DIR)/build/x86/mingw/bin -llibopencv_features2d244  -llibopencv_imgproc244.dll -llibopencv_objdetect244 -llibopencv_video244
#libopencv_calib3d244.dll.a  libopencv_flann244.dll.a       libopencv_legacy244.dll.a   libopencv_photo244.dll.a      libopencv_videostab245.dll.a
#libopencv_contrib244.dll.a  libopencv_gpu244.dll.a         libopencv_ml244.dll.a       libopencv_stitching244.dll.a
#libopencv_core244.dll.a     libopencv_highgui244.dll.a     libopencv_nonfree244.dll.a  libopencv_ts244.a


tennis: tennis.cpp
	gcc $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f tennis
