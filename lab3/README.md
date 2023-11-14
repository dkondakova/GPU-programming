### Задание

Modify the [previous program](https://github.com/dkondakova/GPU-programming/tree/main/lab2) so as to use all GPUs available for the program. The program should determine the amount of available GPU and distribute the work on them.

### Результаты

<img src="https://github.com/dkondakova/GPU-programming/assets/44597105/1c9ff763-e8d2-485c-9076-ab1ae19fdfa5" width="300"> 
<img src="https://github.com/dkondakova/GPU-programming/assets/44597105/75a35051-3edd-4faa-9950-aa7f2bd66722" width="300">

<table>
    <thead>
        <tr>
            <th rowspan=2>Mem type</th>
            <th colspan=2>One GPU</th>
            <th colspan=2>Two GPUs</th>
        </tr>
        <tr>
            <th>Time, μs</th>
            <th>Time with data copying, μs</th>
            <th>Time per thread, μs</th>
            <th>Time with data copying, μs</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td>global</td>
            <td>1 216</td>
            <td>75 224</td>
            <td>290</td>
            <td>2 991</td>
        </tr>
        <tr>
            <td>shared</td>
            <td>1 021</td>
            <td>74 890</td>
            <td>200</td>
            <td>2 632</td>
        </tr>
        <tr>
            <td>texture</td>
            <td>3 802</td>
            <td>78 505</td>
            <td>811</td>
            <td>3 244</td>
        </tr>
    </tbody>
</table>

---

Чтобы узнать количество доступных GPU на узле, необходимо выполнить следующий код:
```c++
int nGPUs;
cudaGetDeviceCount(&nGPUs);
```

Программа запускалась на сервере с двумя видеокартами.
