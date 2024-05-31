# Steps to debug the C++ file

File stored in  the Kria KV260 at path: `~/vai_runtime/`

First, compile the `resnet50_pt.cpp ` file using the configuration contained in `/.vscode/tasks.json`, option Terminal menu, "Ejecutar tarea de compilación" or in the terminal using "bash -x build_plankton.sh". Note that it is important to include in the arguments of tasks.json the -g option; this is a compiler flag used with gcc or g++ to include debugging information in the compiled executable.

Second, set the debugger path in launch.json

That's all! ready to debug in VSCode connected to the Kria connected via SSH =)

Instructions for a normal use:
sudo ./resnet50_pt resnet50_DYB.xmodel ../DYB_val_full [3000] [random]

P.S.: in this folder, the file main.cc is not used, instead, resnet50_pt.cpp has been copied from `/Vitis-AI/examples/vai_runtime/resnet50_pt` folder of the Vitis-AI 3.0 docker image.

Source code forked from [Vitis-AI repo](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_runtime/resnet50_pt).
