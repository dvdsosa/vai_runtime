# Run a custom-trained CNN model into the AMD Kria KV26

After compiling a FP32 model to INT8, we present here the steps for compiling the C++ code for performing the inference on a test set extracted from DYB-PlanktonNet dataset.

## Copy the dataset to the Kria

When logged in the host computer, in the terminal, copy the image test set by using the command below:

```bash
cd ~/tesis/DYB-linearHead
scp -r test petalinux@192.168.1.142:~/DYB-linearHead/test
```

## Compile and run the C++ file

In terms of FPS, we achieved the optimal performance when compiling and running the C++ source code compared with the Python script.

Compile the `resnet50_pt.cpp` file removing the "-g" argument and changing the "-O0" by "-Ofast" in `task.json` configuration file. Then, you are ready to go using the below syntax:

```bash
cd ~/vai_runtime
./resnet50_pt resnet50_int8.xmodel ../DYB-linearHead/test [1000]
```

Where the number 1000 between brackets is an optional value to control the total quantity of files intended for inference. Use this option without the brackets.

If an error arises due to *cannot open /dev/dpu*, refer to [this](#change-dpu-device-permissions) section.

## Debug the C++ file

File stored in  the Kria KV260 at path: `~/vai_runtime/`

First, compile the `resnet50_pt.cpp ` file using the configuration contained in `/.vscode/tasks.json`, option Terminal menu, "Ejecutar tarea de compilación" or in the terminal using "bash -x build_plankton.sh". Note that it is important to have the following arguments in tasks.json:
1. Include the "-g" option; this is a compiler flag used with gcc or g++ to include debugging information in the compiled executable. 
2. Modify the compiler optimization to "-O0". Thus, the compiler does not apply any optimization to the code, making it suitable to debug. For production, use "-O2".

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

Second, set the input args and the debugger path in launch.json:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/resnet50_pt",
            "args": ["resnet50_int8.xmodel", "../DYB-original/test", "5"],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "targetArchitecture": "arm"
        }
    ]
}
```

That's all! You're ready now to debug in VSCode connected to the Kria via SSH =)

Instructions for a normal use:

```bash
sudo ./resnet50_pt resnet50_int8.xmodel ../DYB-linearHead/test
```

P.S.: in this folder, the file main.cc is not used, instead, our main processing file resnet50_pt.cpp has been copied from `/Vitis-AI/examples/vai_runtime/resnet50_pt` folder of the [Vitis-AI 3.0 GitHub repo](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_runtime/resnet50_pt).

## Run the Python file

The development of the Python code was abandoned due to low FPS performance, so we centered our efforts in the C++ code, go to [that section](#compile-and-run-the-c-file). Below is what documented till the last modification of the resnet50.py code.

```bash
$ cd ~/vai_runtime/python
$ sudo python resnet50.py ../resnet50_int8.xmodel [number_of_images_to_process]
$ sudo python resnet50_pytorch.py ../resnet50_int8.xmodel [number_of_images_to_process]
```

When the optional argument [number_of_images_to_process] is not introduced, the script processes all the images of the provided path.

# Misc

## Load a diferent DPU configuration

Once the Petalinux image is built, including the custom PL (.XSA file), and burnt into a microSD, boot it, log in, 
and copy the folder that contains the app files (myApp.bit.bin, myApp.dtbo, shell.json) to `/lib/firmware/xilinx/` folder.
Finally, load the DPU app with:
```bash
sudo xmutil unloadapp
sudo xmutil loadapp myApp
```
Check which DPU configuration is running with the command:
```bash
sudo xdputil query
```

Check the available apps for the DPU stored at:
```bash
ls /lib/firmware/xilinx
```

For loading a specific app by default at startup, change dir to:
```bash
cd /etc/dfx-mgrd
```
...duplicate the default file 'k26-starter-kits' but with the name 'dpu325', and edit it with the text 'dpu325'. 
Finally, modify the content of default_firmware file with the text 'dpu325', and that's the new app that will load by default.

Source of information: [AMD Support Forums - Automatic loading of an accelerated application](https://adaptivesupport.amd.com/s/question/0D54U00008FFBLUSA5/automatic-loading-of-an-accelerated-application).

## Change DPU device permissions

**TL;DR**: if having the following error when executing the C++ program:

```bash
F20241016 13:10:47.245815  3602 dpu_controller_dnndk.cpp:70] Check failed: fd >= 0 (-1 vs. 0) cannot open /dev/dpu
Aborted
```

then, execute the following command to grant permission to the DPU device:

```bash
sudo chmod a+rw /dev/dpu
```

---

When running any of the above Python or C++ programs, these need to be executed as root by using the sudo command, else, it will return an error.

```bash
F20241016 13:10:47.245815  3602 dpu_controller_dnndk.cpp:70] Check failed: fd >= 0 (-1 vs. 0) cannot open /dev/dpu
Aborted
```

If we check the /dev/dpu permissions, it requires root access:

```bash
$ ls -l /dev/dpu
crw------- 1 root root 10, 124 Jun  1 01:56 /dev/dpu
```

To prevent using sudo, specially when debugging the C++ program, we can make the /dev/dpu device available to any user using the command below (you need to execute it in every restart of the Petalinux system):

```bash
sudo chmod a+rw /dev/dpu
```

Now, we can check how the dpu permissions were changed:

```bash
$ ls -l /dev/dpu
crw-rw-rw- 1 root root 10, 124 Jun  1 01:56 /dev/dpu 
```

If you wish to restore the initial permissions of the /dev/dpu device, then:

```bash
sudo chmod u=rw,g=,o= /dev/dpu
```

## Set the correct fingerprint

When running the compiled model on the Kria using our C++ compiled runtime, the following error may occur:

```bash
W20241016 14:14:35.654453  8442 dpu_runner_base_imp.cpp:733] CHECK fingerprint fail! model_fingerprint 0x101000056010407 is un-matched with actual dpu_fingerprint 0x101000047010407. Please re-compile xmodel with dpu_fingerprint 0x101000047010407 and try again.
F20241016 14:14:35.655059  8442 dpu_runner_base_imp.cpp:695] fingerprint check failure.
Aborted
```

If this is the case, when running the docker Vitis-AI image, configure the DPU fingerprint correctly using the dpu_fingerprint hexadecimal number provided in the above message. Steps below:

```bash
$ sudo vi /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
i.e.:
{
    "target":"0x101000047010407"
}
# Once the fingerprint has been changed, quit and save in vi with the following:
# 1. Exit insert mode if you are in it by pressing `Esc`.
# 2. Type `:wq` for save and quit. If error, type `:w !sudo tee %` and press `Enter` to save with root privileges.
```

For saving the new fingerprint and performing this change only once on your Vitis-AI docker image, commit your docker running instance by the command `docker commit <container_id_or_name> <new_image_name>`.

More info at [AMD Knowledge Base](https://support.xilinx.com/s/article/DPU-fingerprint-ERROR?language=en_US).

## ModuleNotFoundError: No module named 'torch'

Use sudo -E to preserve the environment variables:
```bash
sudo -E ./resnet50_pt resnet50_int8.xmodel ../DYB-linearHead/test
```

# Model visualization

[Link to several websites](https://www.kaggle.com/discussions/getting-started/253300) for constructing a visual representation of our model.

# Kria™ SOM: Hardware Platform Statistics

It is possible to monitor the power consumption, temperature, free memory of the SOM, etc. via the dashboard served by the Petalinux internal webserver at the address: 
[http://192.168.1.142:5006/kria-dashboard](http://192.168.1.142:5006/kria-dashboard). Change the IP according to your Kria IP address.

It is also possible to extract the SOM parameters via the following command:
```bash
xmutil xlnx_platformstats
```

Or to access directly the power consumption via the following command:
```bash
cat /sys/class/hwmon/hwmon0/power1_input | awk '{print $1/1000000 " W"}'
```

Another [way](https://adaptivesupport.amd.com/s/question/0D52E00007G0okuSAB/power-estimates-on-kria-k26-som-wp529?language=en_US) (look third comment of the link) to acces the SOM power consumption:
```bash
cat /sys/bus/i2c/drivers/ina260-adc/1-0040/iio\:device1/in_power2_raw | awk '{print $1/100 " W"}'
```

# Switch on/off the PL

Using this command, but you need to include `/sys/kernel/debug/zynqmp-firmware/pm` when compiling petalinux ([source](https://adaptivesupport.amd.com/s/article/68514?language=en_US)):
```bash
sudo xmutil pwrctl --status
```

```bash
sudo xmutil pwrctl --off
```

Also, [download](https://github.com/Xilinx/xmutil/blob/master/xmutil) the xmutil code in python, edit the code and add at the end of the product_name function, the line:
```python
    prod = 'kv260'
```
... just before the `return prod`

# Git repository
Add local repo to Bitbucket and Github accounts:

	git remote add origin git@bitbucket.org:davidsosatrejo/vai_runtime.git
	git remote set-url --add origin git@github.com:dvdsosa/vai_runtime.git
	git remote -v
	git push origin master

# RESULTS

## C++

Processing a DYB-PlanktonNet test set comprised of 7147 images.

```bash
xilinx-kv260-starterkit-20232:~/vai_runtime$ ./resnet50_pt resnet50_int8.xmodel ../DYB-linearHead/test
[======================================================================] 100 %
Total number of processed images: 7147

Average FPS: 73.0284
Accuracy: 93.33%
Precision: 90.37%
Recall: 86.64%
F1 Score: 87.46%
```