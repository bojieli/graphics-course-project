#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define uchar unsigned char

const char *images[] = {
	"images/1-10.jpg",
	"images/1-12.jpg",
	"images/1-13.jpg",
	"images/1-17.jpg",
	"images/1-18.jpg",
	"images/1-19.jpg",
	NULL
};

static void print_time(const char *label)
{
	static unsigned int frame_count=0;
	static unsigned int begin_sec=0, begin_msec=0;
	static unsigned int prev_sec=0, prev_msec=0;

    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon  = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min  = wtm.wMinute;
    tm.tm_sec  = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);

	double diff = (unsigned)clock - prev_sec + (int)((unsigned)wtm.wMilliseconds - prev_msec)/1000.0;

    prev_sec = (unsigned)clock;
    prev_msec = (unsigned)wtm.wMilliseconds;
	if (begin_sec == 0) {
		begin_sec = prev_sec;
		begin_msec = prev_msec;
	}
	
	double total_time = prev_sec - begin_sec + (int)(prev_msec - begin_msec) / 1000.0;
	fprintf(stderr, "[%d] start = %.3lf (+%.3lf), fps = %lf ",
		frame_count, total_time, diff,
		1.0 / diff);
	++frame_count;
}

char files[][50] = {
	"images/lena.bmp",
	"images/Rect1.bmp",
	"images/Rect2.bmp",
	"images/map.bmp",
	"images/Couple.bmp",
	"images/Girl.bmp",
	"images/blood1.bmp",
	"images/cameraman.bmp",
	"images/pout.bmp",
};

int linear(int argc, char **argv, IplImage *src)
{
	if (argc < 2) {
		printf("Please provide parameters <fa> <fb>\n");
		printf("    D[b] = f(D[a]) = fa * D[a] + fb\n");
		return 1;
	}
	double fa = atof(argv[1]);
	double fb = atof(argv[2]);
	IplImage *dst = cvCreateImage(cvGetSize(src), 8, 1);

	int i, j;
	for (i=0; i<src->width; i++)
		for (j=0; j<src->height; j++) {
			CvScalar p = cvGet2D(src, j, i);
			p.val[0] = (int)(p.val[0] * fa + fb);
			if (p.val[0] < 0)
				p.val[0] = 0;
			else if (p.val[0] > 255)
				p.val[0] = 255;
			cvSet2D(dst, j, i, p);
		}

	cvShowImage("linear", dst);
	cvReleaseImage(&dst);
	return 0;
}

int extension(int argc, char** argv, IplImage *src)
{
	if (argc < 4) {
		printf("Usage: <x1> <y1> <x2> <y2>\n");
		return 1;
	}
	int x1 = atoi(argv[1]);
	int y1 = atoi(argv[2]);
	int x2 = atoi(argv[3]);
	int y2 = atoi(argv[4]);
	if (x2 == x1) {
		printf("Error");
		return 1;
	}
	IplImage *dst = cvCreateImage(cvGetSize(src), 8, 1);

	int i, j;
	for (i=0; i<src->width; i++)
		for (j=0; j<src->height; j++) {
			CvScalar p = cvGet2D(src, j, i);
			int x = (int)p.val[0];
			if (x < x1)
				x = y1 * x / x1;
			else if (x <= x2)
				x = (x-x1) * (y2-y1) / (x2-x1) + y1;
			else
				x = (255-y2) * (x-x2) / (255-x2) + y2;
			p.val[0] = x;
			cvSet2D(dst, j, i, p);
		}

	cvShowImage("extension", dst);
	cvReleaseImage(&dst);
	return 0;
}

int histogram(int argc, char** argv, IplImage *src)
{
	int min = 0, max = 255;
	if (argc == 3) {
		min = atoi(argv[1]);
		max = atoi(argv[2]);
	}
	int total_height = 600;
	int bar_width = 5;
	CvSize size;
	size.width = bar_width*(max-min+1);
	size.height = total_height;
	IplImage *dst = cvCreateImage(size, 8, 1);

	int counter[256] = {0};
	int i, j, k;
	for (i=0; i<src->width; i++)
		for (j=0; j<src->height; j++) {
			CvScalar p = cvGet2D(src, j, i);
			++counter[(int)p.val[0]];
		}
	int maxcount = 0;
	for (i=min; i<=max; i++)
		if (counter[i] > maxcount)
			maxcount = counter[i];

	CvScalar outp;
	for (i=min; i<=max; i++) {
		int height = total_height * counter[i] / maxcount;
		outp.val[0] = i;
		for (j=total_height-height; j<total_height; j++)
			for (k=0; k<bar_width; k++)
				cvSet2D(dst, j, (i-min)*bar_width+k, outp);
	}

	cvShowImage("histogram", dst);
	cvReleaseImage(&dst);
	return 0;
}

int histogram_balance(int argc, char** argv, IplImage *src)
{
	IplImage *dst = cvCreateImage(cvGetSize(src), 8, 1);
	cvEqualizeHist(src, dst);

	cvNamedWindow("source", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("source", 200, 200);
	cvNamedWindow("dest", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("dest", 600, 200);

	cvShowImage("source", src);
	cvShowImage("dest", dst);
	histogram(argc, argv, dst);
	cvReleaseImage(&dst);
	return 0;
}

#define fix(i,max) ((i)==(max) ? (max)-1 : ((i)==-1 ? 0 : (i)))
#define get(i,j) (cvGet2D(src, fix(i,src->height), fix(j,src->width)).val[0])

#define abs(n) ((n) > 0 ? (n) : -(n))
#define max(a,b) ((a) > (b) ? (a) : (b))

#define ref(i,j) (delta[(i) * src->width + (j)])

#define GEN_EDGE_DETECTOR(func)                                      \
void func(IplImage* src, IplImage* dst, int blackwhite)              \
{                                                                    \
    int i, j;														 \
	int maxpixel = 0;												 \
    int *delta = (int*)malloc(src->height * src->width * sizeof(int)); \
    for (i=0; i<src->height; i++)                                    \
        for (j=0; j<src->width; j++) {                               \
            ref(i,j) = (int)(CALC_DELTA(i,j));						 \
            if (ref(i,j) > maxpixel)                                 \
                maxpixel = ref(i,j);                                 \
        }                                                            \
    if (maxpixel == 0)                                               \
        maxpixel = 1;                                                \
    for (i=0; i<src->height; i++)                                    \
        for (j=0; j<src->width; j++) {                               \
            CvScalar outp;                                           \
			if (blackwhite == 0)												\
				outp.val[0] = delta[i*src->width + j] > 100 ? 255 : 0;			\
			else if (blackwhite == 1)											\
				outp.val[0] = delta[i*src->width + j] > 255 ? 255 : delta[i*src->width+j];	\
			else														\
				outp.val[0] = delta[i*src->width + j] * 255 / maxpixel; \
            cvSet2D(dst, i, j, outp);                                \
        }                                                            \
	free(delta);													 \
}

#define CALC_DELTA(i,j)            \
      abs(get(i,j) - get(i+1,j+1)) \
    + abs(get(i+1,j) - get(i,j+1))
GEN_EDGE_DETECTOR(Roberts)
#undef CALC_DELTA

#define CALC_DELTA(i,j)                                             \
    max(															\
        abs((-1)*get(i-1,j-1) + (-2)*get(i-1,j) + (-1)*get(i-1,j+1) \
            + 1*get(i+1,j-1) + 2*get(i+1,j) + 1*get(i+1,j+1)),      \
        abs((-1)*get(i-1,j-1) + 1*get(i-1,j+1)                      \
            + (-2)*get(i,j-1) + 2*get(i,j+1)                        \
            + (-1)*get(i+1,j-1) + 1*get(i+1,j+1))                   \
       )
GEN_EDGE_DETECTOR(Sobel)
#undef CALC_DELTA

#define CALC_DELTA(i,j)                                             \
    max(                                                            \
        abs((-1)*get(i-1,j-1) + (-1)*get(i-1,j) + (-1)*get(i-1,j+1) \
            + 1*get(i+1,j-1) + 1*get(i+1,j) + 1*get(i+1,j+1)),      \
        abs(1*get(i-1,j-1) + (-1)*get(i-1,j+1)                      \
            + 1*get(i,j-1) + (-1)*get(i,j+1)                        \
            + 1*get(i+1,j-1) + (-1)*get(i+1,j+1))                   \
       )
GEN_EDGE_DETECTOR(Prewitt)
#undef CALC_DELTA

#define CALC_DELTA(i,j)                             \
    abs(+ get(i-1,j)                                  \
        + get(i,j-1) + (-4)*get(i,j) + get(i,j+1)   \
        + get(i+1,j))
GEN_EDGE_DETECTOR(Laplace1)
#undef CALC_DELTA

#define CALC_DELTA(i,j)                             \
    abs(- get(i-1,j-1) - get(i-1,j) - get(i-1,j+1)  \
        - get(i,j-1)   + 8*get(i,j) - get(i,j+1)    \
        - get(i+1,j-1) - get(i+1,j) - get(i+1,j+1))
GEN_EDGE_DETECTOR(Laplace2)
#undef CALC_DELTA

int edge_detect(int argc, char** argv, IplImage *src, int blackwhite)
{
	cvNamedWindow("source", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("source", 0, 0);
	cvShowImage("source", src);

    IplImage *dst;
#define PROCESS(func, left, top)				\
    dst = cvCreateImage(cvGetSize(src), 8, 1);	\
    func(src, dst, blackwhite);					\
    cvNamedWindow(#func, CV_WINDOW_AUTOSIZE);	\
	cvMoveWindow(#func, (top), (left));			\
	cvShowImage(#func, dst);					\
    cvReleaseImage(&dst);

    PROCESS(Roberts, 0, 300)
    PROCESS(Sobel, 300, 0)
    PROCESS(Prewitt, 300, 300)
    PROCESS(Laplace1, 0, 600)
    PROCESS(Laplace2, 300, 600)
    return 0;
}

// dir = 1: forward transform
// dir = -1: reverse transform
static void DFT(int dir, int m, double *x1, double *y1)
{
   long i,k;
   double arg;
   double cosarg,sinarg;
   double *x2 = (double*)malloc(m*sizeof(double));
   double *y2 = (double*)malloc(m*sizeof(double));

   for (i=0;i<m;i++) {
      x2[i] = 0;
      y2[i] = 0;
      arg = - dir * 2.0 * 3.141592654 * (double)i / (double)m;
      for (k=0;k<m;k++) {
         cosarg = cos(k * arg);
         sinarg = sin(k * arg);
         x2[i] += (x1[k] * cosarg - y1[k] * sinarg);
         y2[i] += (x1[k] * sinarg + y1[k] * cosarg);
      }
   }

   /* Copy the data back */
   if (dir == 1) {
      for (i=0;i<m;i++) {
         x1[i] = x2[i] / (double)m;
         y1[i] = y2[i] / (double)m;
      }
   } else {
      for (i=0;i<m;i++) {
         x1[i] = x2[i];
         y1[i] = y2[i];
      }
   }

   free(x2);
   free(y2);
}

#define SWAP(x,y) { double t=x; x=y; y=t; }
static void swap_quad(int height, int width, double* mat)
{
    int i, j;
	for(i = 0; i < height/2; i++)
		for(j = 0; j < width/2; j++) {
			SWAP(mat[i*width + j], mat[(i+height/2)*width + j+width/2])
			SWAP(mat[i*width + j+width/2], mat[(i+height/2)*width + j])
		}
}

static void log_image(int height, int width, double* mat)
{
	int i;
    for (i=0; i<height*width; i++)
        mat[i] = log(mat[i] + 1);
}

static void normalize(int height, int width, double* mat)
{
    int i;
    double max = 0;
	for (i=0; i<height*width; i++)
		if (mat[i] > max)
			max = mat[i];
    for (i=0; i<height*width; i++)
        mat[i] = mat[i] * 255.0 / max;
}

static void image_ift(int height, int width, double *in_mag, double *in_phase, double *mag, double *phase)
{
	int i, j;
	// 存储实部虚部的临时空间
	double *xt = (double*)malloc(height*width*sizeof(double));
    double *yt = (double*)malloc(height*width*sizeof(double));
	double *x1 = (double*)malloc(height * sizeof(double));
    double *y1 = (double*)malloc(height * sizeof(double));
	for (j=0; j<width; j++) {
		// 先把输入的 msg 和 phase 变成实部虚部
		for (i=0; i<height; i++) {
			x1[i] = in_mag[i*width + j] * cos(in_phase[i*width + j]);
			y1[i] = in_mag[i*width + j] * sin(in_phase[i*width + j]);
		}
		// 原地对这一列进行 Fourier 反变换
		DFT(-1, height, x1, y1);
		for (i=0; i<height; i++) {
			int index = i*width + j;
			xt[index] = x1[i];
			yt[index] = y1[i];
		}
	}
	for (i=0; i<height; i++) {
		// 原地对这一行进行 Fourier 反变换
		DFT(-1, width, xt + i*width, yt + i*width);
		// 写入幅度图，相位图
		for (j=0; j<width; j++) {
			int index = i*width+j;
			mag[index] = sqrt(xt[index]*xt[index] + yt[index]*yt[index]);
			phase[index] = xt[index] >= 0 ? atan(yt[index] / xt[index]) : 3.1415926 + atan(yt[index] / xt[index]);
		}
	}
	free(xt);
	free(yt);
	free(x1);
	free(y1);
}

static void image_dft(int height, int width, double *in_mag, double *in_phase, double *mag, double *phase)
{
	int i, j;
	// 存储实部虚部的临时空间
	double *xt = (double*)malloc(height*width*sizeof(double));
    double *yt = (double*)malloc(height*width*sizeof(double));
	for (i=0; i<height; i++) {
		// 先把输入的 msg 和 phase 变成实部虚部
		for (j=0; j<width; j++) {
			xt[i*width + j] = in_mag[i*width + j] * cos(in_phase[i*width + j]);
			yt[i*width + j] = in_mag[i*width + j] * sin(in_phase[i*width + j]);
		}
		// 原地对这一行进行 Fourier 变换
		DFT(1, width, xt + i*width, yt + i*width);
	}
    double *x1 = (double*)malloc(height * sizeof(double));
    double *y1 = (double*)malloc(height * sizeof(double));
	for (j=0; j<width; j++) {
		// 取出中间结果的一列
		for (i=0; i<height; i++) {
		    x1[i] = xt[i*width + j];
			y1[i] = yt[i*width + j];
		}
		// 原地对这一列进行 Fourier 变换
		DFT(1, height, x1, y1);
		// 写入幅度图，相位图
		for (i=0; i<height; i++) {
			mag[i*width + j] = sqrt(x1[i]*x1[i] + y1[i]*y1[i]);
			phase[i*width + j] = x1[i] >= 0 ? atan(y1[i] / x1[i]) : 3.1415926 + atan(y1[i] / x1[i]);
		}
	}
	free(xt);
	free(yt);
	free(x1);
	free(y1);
}

#define IMAGE_SIZE (src->width * src->height * sizeof(double))

static void output_img(const char* window, IplImage* src, double* value, int left, int top, bool swap, bool log)
{
	int i, j;
    // do not overwrite original values
    double *out = (double*)malloc(IMAGE_SIZE);
    memcpy(out, value, IMAGE_SIZE);
	if (swap)
		swap_quad(src->height, src->width, out);
	if (log)
		log_image(src->height, src->width, out);
	normalize(src->height, src->width, out);

	IplImage *dst = cvCreateImage(cvGetSize(src), 8, 1);
	for (i=0; i<src->height; i++)
		for (j=0; j<src->width; j++) {
			cvSet2D(dst, i, j, cvScalar((int)out[i*src->width + j]));
		}
    free(out);

	cvNamedWindow(window, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(window, left, top);
	cvShowImage(window, dst);
    cvReleaseImage(&dst);
}

int fourier_transform(int argc, char** argv, IplImage *src)
{
	int i, j;

    cvNamedWindow("source", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("source", 0, 0);
    cvShowImage("source", src);

	// 从源图像读取实部，虚部为0
	double *s1 = (double*)malloc(IMAGE_SIZE);
	double *s2 = (double*)malloc(IMAGE_SIZE);
    for (i=0; i<src->height; i++) {
		for (j=0; j<src->width; j++) {
			s1[i*src->width + j] = cvGet2D(src, i, j).val[0];
			s2[i*src->width + j] = 0;
		}
    }

	double *mag = (double*)malloc(IMAGE_SIZE);
	double *phase = (double*)malloc(IMAGE_SIZE);
	image_dft(src->height, src->width, s1, s2, mag, phase);
    free(s1);
    free(s2);

    output_img("magnitude transformed", src, mag, 0, 300, true, true);
    output_img("phase transformed", src, phase, 300, 0, true, false);

	double *empty = (double*)malloc(IMAGE_SIZE);
    for (i=0; i<src->height * src->width; i++)
        empty[i] = 0;

	double *newmag = (double*)malloc(IMAGE_SIZE);
	double *newphase = (double*)malloc(IMAGE_SIZE);
	image_ift(src->height, src->width, mag, empty, newmag, newphase);
	output_img("magnitude reverse transform", src, newmag, 300, 300, false, true);

	image_ift(src->height, src->width, mag, phase, newmag, newphase);
	output_img("full reverse transform", src, newmag, 600, 0, false, false);

	free(mag);
	free(phase);
	free(newmag);
	free(newphase);
    free(empty);

    return 0;
}

int image_smooth(int argc, char** argv, IplImage* src)
{
    int i, j, k, l;

    cvNamedWindow("source", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("source", 0, 0);
    cvShowImage("source", src);

    // add 3% salt-pepper noise
    srand((int)time(NULL));
    for (i=0; i<src->height; i++)
        for (j=0; j<src->width; j++) {
            if (rand() % 10000 < 300) { // 3% probability
                CvScalar p;
                p.val[0] = (rand() & 1) ? 255 : 0;
                cvSet2D(src, i, j, p);
            }
        }
    cvNamedWindow("salt-pepper noise", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("salt-pepper noise", 300, 0);
    cvShowImage("salt-pepper noise", src);

    // average smoothing
    IplImage *dst = cvCreateImage(cvGetSize(src), 8, 1);
    for (i=0; i<src->height; i++)
        for (j=0; j<src->width; j++) {
            double total = 0;
            for (k=-1; k<=1; k++)
                for (l=-1; l<=1; l++)
                    total += get(i+k, j+l);
            CvScalar p;
            p.val[0] = total / 9;
            cvSet2D(dst, i, j, p);
        }
    cvNamedWindow("average smoothing", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("average smoothing", 0, 300);
    cvShowImage("average smoothing", dst);

    // median smoothing
    for (i=0; i<src->height; i++)
        for (j=0; j<src->width; j++) {
            double ordered[9] = {0};
            int pos = 0;
            for (k=-1; k<=1; k++)
                for (l=-1; l<=1; l++) {
                    // insertion sort
                    int m, t;
                    double cur = get(i+k, j+l);
                    for (m=0; m<pos; m++)
                        if (cur <= ordered[m])
                            break;
                    for (t=pos; t>m; t--)
                        ordered[t] = ordered[t-1];
                    ordered[m] = cur;
                    pos++;
                }
            CvScalar p;
            p.val[0] = ordered[4]; // median
            cvSet2D(dst, i, j, p);
        }
    cvNamedWindow("median smoothing", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("median smoothing", 300, 300);
    cvShowImage("median smoothing", dst);

    cvReleaseImage(&dst);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("Please specify transformation:\n");
		printf("    1 for linear transformation\n");
		printf("    2 for extension\n");
		printf("    3 for histogram\n");
		printf("    4 for histogram balancing\n");
		printf("    5 for edge detection (black and white)\n");
		printf("    6 for edge detection (>255 normalized to 255)\n");
		printf("    7 for edge detection (normalize to 0..255)\n");
		printf("    8 for Fourier transform\n");
        printf("    9 for image smoothing\n");
		return 1;
	}
	printf("Please press any key to switch images, 'q' to exit\n");
	fflush(stdout);

	int thisfile = 0;
	while (1) {
		IplImage *src = cvLoadImage(files[thisfile], 0);
		if (src == NULL) {
			printf("Cannot open file %s\n", files[thisfile]);
			return 1;
		}
		int action = atoi(argv[1]);
		switch (action) {
		case 1:
			if (linear(argc-1, argv+1, src)) return 1;
			break;
		case 2:
			if (extension(argc-1, argv+1, src)) return 1;
			break;
		case 3:
			if (histogram(argc-1, argv+1, src)) return 1;
			break;
		case 4:
			if (histogram_balance(argc-1, argv+1, src)) return 1;
			break;
        case 5:
            if (edge_detect(argc-1, argv+1, src, 0)) return 1;
            break;
		case 6:
			if (edge_detect(argc-1, argv+1, src, 1)) return 1;
			break;
		case 7:
			if (edge_detect(argc-1, argv+1, src, 2)) return 1;
			break;
        case 8:
            if (fourier_transform(argc-1, argv+1, src)) return 1;
            break;
        case 9:
            if (image_smooth(argc-1, argv+1, src)) return 1;
            break;
		default:
			printf("Invalid action\n");
			return 1;
		}
		cvReleaseImage(&src);
		if (cvWaitKey() == 'q')
			return 0;

		++thisfile;
		if (thisfile == sizeof(files)/sizeof(files[0]))
			thisfile = 0;
	}
}
