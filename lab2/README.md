### Задание

Implement a program for applying filters to your images. Possible filters: **blur**, edge detection, denoising. Implement three versions of the program, namely, using `global`, `shared` memory and `texture`. Compare the time.
To work with image files, it is recommended to use libpng (man libpng).

### Результаты

Time (global): 3355&mu;s.  
Time (shared): 3392&mu;s.  
Time (texture): 6374&mu;s.  

##### Возможое объяснение результатов:
+ **Почему нет разнинцы между `global` и `shared`?**  
  Мультипроцессоры (SM) могут выполнять различное количество блоков одновременно в зависимости от ресурсов SM и характеристик блоков. Размер сетки (количество блоков) на одном SM может быть ограничен различными факторами, включая доступ к разделяемой памяти, регистрам и ядрам потоков.  

  Рассмотрим параметры GPU:
```c++
int main() {
  int deviceID = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceID);
  
  int numSM = deviceProp.multiProcessorCount;
  int maxBlocksPerSM = deviceProp.maxBlocksPerMultiProcessor;
  int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
  int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
  size_t sharedMemPerMultiprocessor = deviceProp.sharedMemPerMultiprocessor;
  size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;
  
  std::cout << "Количество мультипроцессоров: " << numSM << std::endl;
  std::cout << "Максимальное количество блоков на одном SM: " << maxBlocksPerSM <<std::endl;
  std::cout << "Максимальное количество ядер потоков на одном SM: " << maxThreadsPerSM <<std::endl;
  std::cout << "Максимальное количество потоков на блок: " << maxThreadsPerBlock << std::endl;
  std::cout << "Максимальный размер разделяемой памяти на SM: " << sharedMemPerMultiprocessor / 1024 << "Кбайт" << std::endl;
  std::cout << "Максимальный размер разделяемой памяти на блок: " << sharedMemPerBlock / 1024 << "Кбайт" << std::endl;
}
```

```
Количество мультипроцессоров: 36
Максимальное количество блоков на одном SM: 16
Максимальное количество ядер потоков на одном SM: 1024
Максимальное количество потоков на блок: 1024

Максимальный размер разделяемой памяти на SM: 64Кбайт
Максимальный размер разделяемой памяти на блок: 48Кбайт
```

  Нас больше всего интересует `Максимальное количество ядер потоков на одном SM: 1024`.  
  Размер сетки для `global` и `shared` – 64х64.  
  Размер блока для `global` – 16х16, для `shared` – 32х32.  
  То есть для `global` в одном блоке выполняется 256 потоков, а для `shared` – 1024 – максимально возможное число.  

  Тогда для `global` получаем, что одновременно может выполняться 144 блока потоков:
```
Количество SM на устройстве: 36
Количество блоков, которые можно запустить на SM: 4
Максимальное количество блоков, которое можно запустить: 144
```

  А для `shared` в 4 раза меньше – 36:
```
Количество SM на устройстве: 36
Количество блоков, которые можно запустить на SM: 1
Максимальное количество блоков, которое можно запустить: 36
```

  Что касается используемой разделяемой памяти:
```
__shared__ uchar3 sharedData[BLOCK_SIZE + 2 * BLUR_RADIUS][BLOCK_SIZE + 2 * BLUR_RADIUS];
Разделяемая память, требуемая блоком: 3072 байт (3 Кбайт)

__shared__ float3 sharedData[BLOCK_SIZE + 2 * BLUR_RADIUS][BLOCK_SIZE + 2 * BLUR_RADIUS];
Разделяемая память, требуемая блоком: 12288 байт (12 Кбайт)
```
  + **Почему `texture` в два раза дольше`global`?**
      + Возможно это из-за того, что используется линейная текстура, а не `cudaArray`? И мб используется неправильно?
      + Может быть размера доступной линейной текстуры не хватает для всего изображения? Но [тут](https://en.wikipedia.org/wiki/CUDA#:~:text=Maximum%20width%20for%201D%20texture%20reference%20bound%20to%20linear%0Amemory) пишут, что размер линейной текстуры – $2^{27}$(байт?) = 128Мбайт, а размер массива, в котором хранится изображение – 3Мбайта.
