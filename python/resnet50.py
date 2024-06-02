"""
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys
import re

# Declarar las variables globales
inferred_classes = []
ground_truth_classes = []

"""
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
"""
def CPUCalcSoftmax(data, size, scale):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i] * scale)
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    return result

def get_script_directory():
    path = os.getcwd()
    return path

"""
Get topk results according to its probability
datain: data result of softmax
filePath: filePath in witch that records the infotmation of kinds
"""
def TopK(datain, size, ground_truth_filename, data1):

    global inferred_classes
    global ground_truth_classes

    cnt = [i for i in range(size)]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)

    inferred_class_id = int(data1[cnt_new[0]].strip("\n").split('_')[0])
    ground_truth_class_id = int(re.match(r'(\d+)', ground_truth_filename).group(1))
    # index -->  cnt_new[0]+1
    #print("Top[%d] Inferenced --> %02d, %02d <-- Ground Truth. Queried filename %s" % (0, inferred_class_id, ground_truth_class_id, ground_truth_filename))

    inferred_classes.append(inferred_class_id)
    ground_truth_classes.append(ground_truth_class_id)

"""
pre-process for resnet50 (caffe)
"""
_B_MEAN = 10.455
_G_MEAN = 9.0525
_R_MEAN = 10.6845
MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
SCALES = [0.05086340632, 0.0429525589, 0.04089226932]

def preprocess_one_image_fn(image_path, fix_scale, width=224, height=224):
    means = MEANS
    scales = SCALES
    image = cv2.imread(image_path)
    # Resize the image to 256 on the smallest side
    smallest_side = 256.0
    scale = smallest_side / min(image.shape[0], image.shape[1])
    new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    resized_image = cv2.resize(image, new_size)
    
    # Center crop the image
    y, x, _ = resized_image.shape
    startx = x//2 - (width//2)
    starty = y//2 - (height//2)
    crop_img = resized_image[starty:starty+height, startx:startx+width]

    B, G, R = cv2.split(crop_img) 
    B = (B - means[0]) * scales[0] * fix_scale
    G = (G - means[1]) * scales[1] * fix_scale
    R = (R - means[2]) * scales[2] * fix_scale
    image = cv2.merge([B, G, R])
    image = image.astype(np.int8)
    return image


SCRIPT_DIR = get_script_directory()
calib_image_dir = SCRIPT_DIR + "/../../test01/"
global threadnum
threadnum = 0
"""
run resnt50 with batch
runner: dpu runner
img: imagelist to be run
cnt: threadnum
"""
def runResnet50(runner: "Runner", img, ground_truth_filename, cnt):
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])

    output_ndim = tuple(outputTensors[0].dims)
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    n_of_images = len(img)

    # Read the file once outside the function
    with open("./plankton_list.txt", "r") as fp:
        data1 = fp.readlines()

    count = 0
    while count < cnt:
        runSize = input_ndim[0]
        """prepare batch input/output """
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]
        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        """run with batch """
        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)
        """softmax&TopK calculate with batch """
        """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
        """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
        for j in range(runSize):
            softmax = CPUCalcSoftmax(outputData[0][j], pre_output_size, output_scale)
            TopK(softmax, pre_output_size, ground_truth_filename[count], data1)

        count = count + runSize

"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def main(argv):
    global threadnum

    listimage = os.listdir(calib_image_dir)
    threadAll = []
    threadnum = int(argv[1])
    i = 0
    runTotal = len(listimage)
    g = xir.Graph.deserialize(argv[2])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    time_start = time.time()
    """image list to be run """
    img = []
    # List of integers
    ground_truth_filename = []
    for i in range(runTotal):
        path = os.path.join(calib_image_dir, listimage[i])
        img.append(preprocess_one_image_fn(path, input_scale))
        ground_truth_filename.append(listimage[i])
    """
      The cnt variable is used to control the number of times a single-thread DPU runs.
      Users can modify the value according to actual needs. It is not recommended to use
      too small number when there are few input images, for example:
      1. If users can only provide very few images, e.g. only 1 image, they should set
         a relatively large number such as 360 to measure the average performance;
      2. If users provide a huge dataset, e.g. 50000 images in the directory, they can
         use the variable to control the test time, and no need to run the whole dataset.
    """
    cnt = 1
    """run with batch """
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runResnet50, args=(all_dpu_runners[i], img, ground_truth_filename, cnt))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners

    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = cnt * int(threadnum)
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )
    # Calcular la precisión
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(ground_truth_classes)):
        if ground_truth_classes[i] == inferred_classes[i]:  # True Positive
            true_positives += 1
        elif inferred_classes[i] not in ground_truth_classes:  # False Positive
            false_positives += 1
        else:  # False Negative
            false_negatives += 1

    total = true_positives + false_positives + false_negatives
    accuracy = true_positives / total
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Imprimir los resultados
    print("Accuracy: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1 Score: %.2f%%" % (accuracy*100, precision*100, recall*100, f1_score*100))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage : python3 resnet50.py <thread_number> <resnet50_xmodel_file>")
    else:
        main(sys.argv)
