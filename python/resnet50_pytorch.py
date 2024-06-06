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
from tqdm import tqdm
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
import random
import torch
from torchvision import transforms, datasets

# Declarar las variables globales
inferred_classes = []
ground_truth_classes = []
start_time=[]
start_time_2=[]

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
def TopK(softmax, read_image_filename, target_classes):

    global inferred_classes
    global ground_truth_classes

    max_value=max(softmax)
    max_index=softmax.index(max_value)

    inferred_class_id = int(target_classes[max_index].strip("\n").split('_')[0])
    ground_truth_class_id = int(re.search(r'/(\d+)_', read_image_filename[0]).group(1))

    # index -->  cnt_new[0]+1
    #print("Top[%d] Inferenced --> %02d, %02d <-- Ground Truth. Queried filename %s" % (0, inferred_class_id, ground_truth_class_id, ground_truth_filename))

    inferred_classes.append(inferred_class_id)
    ground_truth_classes.append(ground_truth_class_id)

global threadnum
threadnum = 0
"""
run resnt50 with batch
runner: dpu runner
val_loader: dataset to be run
"""
def runResnet50(runner: "Runner", val_loader):
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])

    output_ndim = tuple(outputTensors[0].dims)
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)

    # Read the file once outside the function
    with open("./plankton_list.txt", "r") as fp:
        target_classes = fp.readlines()

    try: 
        for image, _, read_image_filename in val_loader:
            runSize = input_ndim[0]
            """prepare batch input/output """
            inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
            outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]
            """init input image to input buffer """
            for j in range(val_loader.batch_size):
                imageRun = inputData[0]
                imageRun[j, ...] = image[j].permute(1, 2, 0)
            """run with batch """
            job_id = runner.execute_async(inputData, outputData)
            runner.wait(job_id)
            """softmax&TopK calculate with batch """
            """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
            """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
            for j in range(runSize):
                softmax = CPUCalcSoftmax(outputData[0][j], pre_output_size, output_scale)
                TopK(softmax, read_image_filename, target_classes)
    except KeyboardInterrupt:
        print("\nInterruption detected, stopping the loop...")

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

class CustomImageFolder(datasets.ImageFolder):
    # This method is called when an item from the dataset is accessed
    def __getitem__(self, index):
        # Get the image and label as you normally would
        image, label = super().__getitem__(index)
        # Get the image file name
        image_file = self.samples[index][0]
        # Return the image, label, and file name
        return image, label, image_file

def load_data(data_dir='dataset/imagenet',
              batch_size=1,
              subset_len=None,
              sample_method='random',
              fix_scale=8,
              **kwargs):

    means = [10.6845, 9.0525, 10.455]
    scales = [0.04089226932, 0.0429525589, 0.05086340632]
    #normalize = transforms.Normalize(mean=means, std=scales)
    val_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the smallest side to 256
        transforms.CenterCrop(224),  # Crop the center of the image
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Lambda(lambda x: x[[2, 1, 0], :, :]),  # Change the channel order from RGB to BGR
        transforms.Lambda(lambda x: x * 255),
        transforms.Lambda(lambda x: (x - torch.tensor(means).view(3, 1, 1)) * torch.tensor(scales).view(3, 1, 1) * fix_scale),  # Apply the normalization
        transforms.Lambda(lambda x: x.to(torch.int8)),  # Convert the tensor to integers
    ])
    dataset = CustomImageFolder(root=data_dir, transform=val_transform)

    if subset_len:
        assert subset_len <= len(dataset)
        if sample_method == 'random':
            dataset = torch.utils.data.Subset(dataset, random.sample(range(0, len(dataset)), subset_len))
        else:
            dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

def main(argv):
    global threadnum
    global start_time

    threadAll = []
     # Fixed to 1 due the maximum DPU cores that can be implemented in the Kria KV260 is one
    threadnum = 1

    # Check if the user has introduced the quantity of images to process
    if len(argv) > 2:
        image_count = int(argv[2])
        if image_count > 0:
            subset_len = image_count
        else:
            subset_len = None
    else:
        subset_len = None

    # Load the .xmodel to perform the inference
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    # Set the start time
    start_time = time.time()

#    data_dir= '../../DYB-original/test',
#    data_dir= '../../test02/',
    val_loader = load_data(
        data_dir= '../../DYB-original/test',
        batch_size=1,
        subset_len=subset_len,
        sample_method='random',
        fix_scale=input_scale
        )

    # When the user has not introduced the quantity of images to process set it to the max value
    if 'image_count' not in locals():
        image_count = len(val_loader)
    
    end_time_1 = time.time()
    diff_1 = (end_time_1 - start_time) * 1000

    # for iteration, (images, labels) in enumerate(val_loader):
    #     print(labels)

    start_time_2 = time.time()
    """run with batch """
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runResnet50, args=(all_dpu_runners[i], val_loader))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners

    end_time_2 = time.time()
    timetotal = (end_time_2 - start_time)
    print(f"Elapsed time for dataset image loading: {diff_1} ms")
    print(f"Elapse time since for inference: {timetotal} ms")

    total_frames = image_count * int(threadnum)
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )

    # Calculate accuracies
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

    # Print results
    print("Accuracy: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1 Score: %.2f%%" % (accuracy*100, precision*100, recall*100, f1_score*100))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage : python3 resnet50_pytorch.py <resnet50_xmodel_file> [thread_number]")
    else:
        main(sys.argv)
