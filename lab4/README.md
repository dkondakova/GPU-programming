### Задание

Use the method of least squares to find a circle in the image. For each random sample points organize their processing on the GPU. Random sampling arrange with library `CURAND`.

**Input**: an image size of $640\times480$ (for example, a speed limit sign), the number of samples $N$, the number of elements in each sample $K$.  
**Output**: The image with a circle painted on it.

**Recommendation**: before processing apply Sobel filter for edge detection and consider the point at which the `normalized color value>0.5`.

### Результаты

#### Входное изображение:

<img src="https://github.com/dkondakova/GPU-programming/assets/44597105/b7305b28-0f90-4cad-96e0-fdd27c92c9dd" width="300"> 


#### Обработанное фильтром Собеля (с выделенными пикселями, у `которых normalized color value>0.9`):

<img src="https://github.com/dkondakova/GPU-programming/assets/44597105/6de530ec-fddc-4659-b22e-248130259d27" width="300"> 


#### Итоговое изображение:

Может быть такое, что ни в одном из $N$ наборов из $K$ случайнных точек не будет всех $K$, лежащих на одной реальной окружности. Тогда будет выбираться набор точек, наиболее подходящих для окружности.

Поэтому могут получаться вот такие результаты:

<table>
  <tr>
    <td><img src="https://github.com/dkondakova/GPU-programming/assets/44597105/be5f6163-e87c-4445-af78-ede142c887f6" alt="Image 1" width="300"/></td>
    <td><img src="https://github.com/dkondakova/GPU-programming/assets/44597105/58dab3fd-122c-4de3-84b7-d745a9be75cc" alt="Image 2" width="300"/></td>
    <td><img src="https://github.com/dkondakova/GPU-programming/assets/44597105/5d6e46ba-7a90-4b1e-9ffc-e542ac41f06c" alt="Image 3" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/dkondakova/GPU-programming/assets/44597105/b09284ac-e519-42e7-8f2e-2f01211158ad" alt="Image 4" width="300"/></td>
    <td><img src="https://github.com/dkondakova/GPU-programming/assets/44597105/1ec7aea0-bb36-442e-a129-75f333f76a24" alt="Image 5" width="300"/></td>
    <td><img src="https://github.com/dkondakova/GPU-programming/assets/44597105/1976ebf8-796c-4299-97fc-11fe1bcb27ca" alt="Image 6" width="300"/></td>
  </tr>
</table>

На каждом изображении показаны: найденная окужность и набор из $K$ точек, покоторым была найдена эта окружность.
