# Steps to debug the C++ file

File stored in  the Kria KV260 at path: `~/vai_runtime/`

First, compile the `resnet50_pt.cpp ` file using the configuration contained in `/.vscode/tasks.json`, option Terminal menu, "Ejecutar tarea de compilación" or in the terminal using "bash -x build_plankton.sh". Note that it is important to have the following arguments in tasks.json:
1. Include the "-g" option; this is a compiler flag used with gcc or g++ to include debugging information in the compiled executable. 
2. Modify the compiler optimization to "-O0". Thus, the compiler does not apply any optimization to the code, making it easier to debug. For production, use "-O2".

```json
{
    "tasks": [
        {
            "type": "shell",
            "label": "C/C++: g++ compilar archivo activo",
            "command": "/usr/bin/g++",
            "args": [
                "-g",
                "-O0",
```

Second, set the debugger path in launch.json

That's all! You're ready now to debug in VSCode connected to the Kria connected via SSH =)

Instructions for a normal use:
sudo ./resnet50_pt resnet50_DYB.xmodel ../DYB_val_full [3000] [random]

P.S.: in this folder, the file main.cc is not used, instead, resnet50_pt.cpp has been copied from `/Vitis-AI/examples/vai_runtime/resnet50_pt` folder of the Vitis-AI 3.0 docker image.

Source code forked from [Vitis-AI repo](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_runtime/resnet50_pt).

# Steps to run the C++ file

`cd ~/vai_runtime`

`./resnet50_pt resnet50_DYB.xmodel ../test01`

# Steps to run the Python file

`cd ~/vai_runtime/python`

`sudo python resnet50.py ../resnet50_DYB.xmodel [number_of_images_to_process]`

`sudo python resnet50_pytorch.py ../resnet50_DYB.xmodel [number_of_images_to_process]`

# Steps to change the permissions of the DPU device
When running any of the above Python or C++ programs, these need to be executed as root by using the sudo command. This is because the dpu device needs root permissions, as shown below:

`ls -l /dev/dpu`

`crw------- 1 root root 10, 124 Jun  1 01:56 /dev/dpu`

To prevent using sudo, especially when debugging the C++ program, we can make the /dev/dpu device available to any user by the following command (you need to execute it in every restart of the Petalinux system):

`sudo chmod a+rw /dev/dpu`

Now, you can see the changes by:

`ls -l /dev/dpu`

`crw-rw-rw- 1 root root 10, 124 Jun  1 01:56 /dev/dpu `

If you wish to restore the initial permissions of the /dev/dpu device, then:

`sudo chmod u=rw,g=,o= /dev/dpu`

# RESULTS


## C++

Para 1000 y 2000 imágenes respectivamente.
```json
xilinx-kv260-starterkit-20232:~/vai_runtime$ ./resnet50_pt resnet50_DYB.xmodel ../targetKria 1000
Average FPS: 66.5283
Accuracy: 92.9%, Precision: 99.8925%, Recall: 92.993%, F1 Score: 96.3193%
xilinx-kv260-starterkit-20232:~/vai_runtime$ ./resnet50_pt resnet50_DYB.xmodel ../targetKria 2000
Average FPS: 66.578
Accuracy: 93.5%, Precision: 100%, Recall: 93.5%, F1 Score: 96.6408%
```

Tiempo total desde que lanzo el script, carga las imágenes, y las procesa:
```json
xilinx-kv260-starterkit-20232:~/vai_runtime$ ./run_resnet50_cpp.sh 
Average FPS: 38.7914
Accuracy: 92.9%, Precision: 99.8925%, Recall: 92.993%, F1 Score: 96.3193%
El comando se ejecutó en 26 segundos.
FPS: 38.46
```

## Python

Para 1000 imágenes, con precisión decimal casi idéntica a C++.
```json
xilinx-kv260-starterkit-20232:~/vai_runtime/python$ python resnet50.py 1 ../resnet50_DYB.xmodel
FPS=81.72, total frames = 1000.00 , time=12.237606 seconds
Accuracy: 73.90%, Precision: 99.60%, Recall: 74.12%, F1 Score: 84.99%
```
Para 2000 imágenes, con precisión decimal casi idéntica a C++.
```json
xilinx-kv260-starterkit-20232:~/vai_runtime/python$ python resnet50.py 1 ../resnet50_DYB.xmodel
FPS=81.30, total frames = 2000.00 , time=24.598747 seconds
Accuracy: 74.00%, Precision: 100.00%, Recall: 74.00%, F1 Score: 85.06%
```

Para 1000 imágenes, con una normalización de los canales según script original de Vitis-AI.
```json
xilinx-kv260-starterkit-20232:~/vai_runtime/python$ python resnet50.py 1 ../resnet50_DYB.xmodel
FPS=80.66, total frames = 1000.00 , time=12.398101 seconds
Accuracy: 83.20%, Precision: 99.88%, Recall: 83.28%, F1 Score: 90.83%
```
Para 2000 imágenes, con una normalización de los canales según script original de Vitis-AI.
```json
xilinx-kv260-starterkit-20232:~/vai_runtime/python$ python resnet50.py 1 ../resnet50_DYB.xmodel
FPS=80.36, total frames = 2000.00 , time=24.886486 seconds
Accuracy: 83.55%, Precision: 100.00%, Recall: 83.55%, F1 Score: 91.04%
```

Tiempo total desde que lanzo el script, carga las imágenes, y las procesa:
```json
xilinx-kv260-starterkit-20232:~/vai_runtime/python$ ./run_resnet50_python.sh 
FPS=32.19, total frames = 1000.00 , time=31.062088 seconds
Accuracy: 83.20%, Precision: 99.88%, Recall: 83.28%, F1 Score: 90.83%
El comando se ejecutó en 33 segundos.
FPS: 30.30
```