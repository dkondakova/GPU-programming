# NSU CUDA course

### Задачи

1. Allocate GPU array arr of 10^8 float elements and initialize it with the kernel as follows: arr[i] = sin((i% 360) * Pi/180). Copy array in CPU memory and count error as err = sum_i(abs (sin((i% 360) * Pi/180) - arr [i]))/10^8. Investigate the dependence of the use of functions: sin, sinf, __ sinf. Explain the result. Check the result for array of double data type.
1. Implement a program for applying filters to your images. Possible filters: blur, edge detection, denoising. Implement three versions of the program, namely, using global, shared memory and texture. Compare the time.
To work with image files, it is recommended to use libpng (man libpng). Examples are in /usr/share/doc/libpng12-dev/examples/.
1. Modify the previous program so as to use all GPUs available for the program. The program should determine the amount of available GPU and distribute the work on them.
