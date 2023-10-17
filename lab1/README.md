### Задание

Allocate GPU array arr of 10^8 float elements and initialize it with the kernel as follows: arr[i] = sin((i% 360) * Pi/180). Copy array in CPU memory and count error as err = sum_i(abs (sin((i% 360) * Pi/180) - arr [i]))/10^8. Investigate the dependence of the use of functions: sin, sinf, __ sinf. Explain the result. Check the result for array of double data type.

### Результаты

|                  | Error         | Time, ms |
|------------------|---------------|----------:|
| (float, sin)     | 1.14873e-08   | 44088  |
| (float, sinf)    | 4.61235e-08   | 2304   |
| (float, __sinf)  | 1.27782e-07   | 2024   |
| (double, sin)    | 8.89527e-18   | 42556  |
| (double, sinf)   | 4.61235e-08   | 2460   |
| (double, __sinf) | 1.27782e-07   | 2681   |

### Выводы

* `sin` обрабатывает аргументы типа `double` и возвращает результат типа `double`
* `sinf` обрабатывает аргументы типа `float` и возвращает результат типа `float`
* `__sinf` предназначена для аппаратного ускорения на GPU, обрабатывает аргументы типа `float` и возвращает результат типа `float`

* Использование функции `sin` с `double` дает наибольшую точность, но наиболее затратно по времени.
* Использование `__sinf` может обеспечить более высокую производительность, но с некоторой потерей точности.

