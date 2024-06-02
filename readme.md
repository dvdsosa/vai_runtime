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

`sudo python resnet50.py 1 ../resnet50_DYB.xmodel`

# Steps to change the permissions of  the DPU device
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
