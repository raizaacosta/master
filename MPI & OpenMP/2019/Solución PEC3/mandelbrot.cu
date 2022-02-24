#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
const int blocksize = 800;	  /* x resolution */
const int Y_RESN = 800;       /* y resolution */
const int X_RESN = 800;       /* x resolution */

typedef struct complextype
{
	float real, imag;
} Compl;

__global__
void mandelbrot(int *buff_device)
{
	/* Mandlebrot variables */
	int j, k;
	Compl	z, c;
	float	lengthsq, temp;
		
	for(j=0; j < Y_RESN; j++) {

        z.real = z.imag = 0.0;
        c.real = ((float) j - 400.0)/200.0;               /* scale factors for 800 x 800 window */
	    c.imag = ((float) threadIdx.x - 400.0)/200.0;
        k = 0;

        do  {                                             /* iterate for pixel color */

            temp = z.real*z.real - z.imag*z.imag + c.real;
            z.imag = 2.0*z.real*z.imag + c.imag;
            z.real = temp;
            lengthsq = z.real*z.real+z.imag*z.imag;
            k++;

        } while (lengthsq < 4.0 && k < 100);

       if (k == 100){ 
			/* draw point return 1 */
			buff_device[(threadIdx.x*Y_RESN)+j] = 1;
	   } else {
			/* draw point return 0 */
			buff_device[(threadIdx.x*Y_RESN)+j] = 0;
	   }

     }
}

void paintMandelbrot(int *buff)
{

	int i, j;
	Window		win;                            /* initialization for a window */
	unsigned
	int             width, height,                  /* window size */
                        x, y,                           /* window position */
                        border_width,                   /* border width in pixels */
                        screen;                         /* which screen */

	char            *window_name = "Mandelbrot Set", *display_name = NULL;
	GC              gc;
	unsigned
	long		valuemask = 0;
	XGCValues	values;
	Display		*display;
	XSizeHints	size_hints;

	XSetWindowAttributes attr[1];

	/* connect to Xserver */

	if (  (display = XOpenDisplay (display_name)) == NULL ) {
	   fprintf (stderr, "drawon: cannot connect to X server %s\n",
				XDisplayName (display_name) );
	exit (-1);
	}
	
	/* get screen size */

	screen = DefaultScreen (display);
	
	/* set window size */

	width = X_RESN;
	height = Y_RESN;

	/* set window position */

	x = 0;
	y = 0;

        /* create opaque window */

	border_width = 4;
	win = XCreateSimpleWindow (display, RootWindow (display, screen),
				x, y, width, height, border_width, 
				BlackPixel (display, screen), WhitePixel (display, screen));

	size_hints.flags = USPosition|USSize;
	size_hints.x = x;
	size_hints.y = y;
	size_hints.width = width;
	size_hints.height = height;
	size_hints.min_width = 300;
	size_hints.min_height = 300;
	
	XSetNormalHints (display, win, &size_hints);
	XStoreName(display, win, window_name);

        /* create graphics context */

	gc = XCreateGC (display, win, valuemask, &values);

	XSetBackground (display, gc, WhitePixel (display, screen));
	XSetForeground (display, gc, BlackPixel (display, screen));
	XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);

	attr[0].backing_store = Always;
	attr[0].backing_planes = 1;
	attr[0].backing_pixel = BlackPixel(display, screen);

	XChangeWindowAttributes(display, win, CWBackingStore | CWBackingPlanes | CWBackingPixel, attr);

	XMapWindow (display, win);
	XSync(display, 0);

	// Draw the points
	for(i=0; i < X_RESN; i++){ 
		for(j=0; j < Y_RESN; j++){
			if(buff[(i*Y_RESN)+j] == 1){
				XDrawPoint (display, win, gc, j, i);
			}
		}
	}
	
	XFlush (display);
	sleep (30);

}

int main()
{
	
	// Master buffer
	int buff[X_RESN*Y_RESN];
	// Pointer to shared buffer memory
	int *buff_device;
	// Size of the buffer
	const int buff_size = (X_RESN*Y_RESN)*sizeof(int);
	
	// Reserve resources
	cudaMalloc( (void**)&buff_device, buff_size );
	// Copy memory to kernels
	cudaMemcpy( buff_device, buff, buff_size, cudaMemcpyHostToDevice );
	
	// Define size block
	dim3 dimBlock( blocksize, 1 );
	
	// Define size grid
	dim3 dimGrid( 1, 1 );
	
	// Call parallel
	mandelbrot<<<dimGrid, dimBlock>>>(buff_device);
	
	// Receive results
	cudaMemcpy( buff, buff_device, buff_size, cudaMemcpyDeviceToHost );
	
	// Free resources
	cudaFree( buff_device );
	
	// Paint the image
	paintMandelbrot(buff);
		
	return EXIT_SUCCESS;
	
}