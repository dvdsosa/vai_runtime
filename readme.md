# Steps to debug the C++ file

File stored in  the Kria KV260 at path: `/home/root/Vitis-AI/examples/vai_runtime/resnet50`

First, compile the `resnet50_pt.cpp ` file using the configuration contained in `/.vscode/tasks.json`, option Terminal menu, "Ejecutar tarea de compilación" or in the terminal using "bash -x build_plankton.sh". Note that it is important to include in the arguments of tasks.json the -g option; this is a compiler flag used with gcc or g++ to include debugging information in the compiled executable.

Second, set the debugger path in launch.json

That's all! ready to debug in VSCode connected to the Kria connected via SSH =)

Instructions for a normal use:
./resnet50_pt resnet50_DYB.xmodel image.list.txt

P.S.: in this folder, the file main.cc is not used, instead, resnet50_pt.cpp has been copied from `/workspace/examples/vai_runtime/resnet50` folder of the Vitis-AI 3.0 docker image.