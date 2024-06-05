/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <xir/graph/graph.hpp>
#include <typeinfo>

#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vitis/ai/collection_helper.hpp"

#include <vector>
#include <string>
#include <random>
#include <chrono>

std::vector<int> inferred_classes;
std::vector<int> ground_truth_classes;
namespace fs = std::filesystem;

static std::vector<std::pair<int, float>> post_process(
    vart::TensorBuffer* tensor_buffer, float scale, int batch_idx);
static std::vector<float> convert_fixpoint_to_float(vart::TensorBuffer* tensor,
                                                    float scale, int batch_idx);

static std::vector<float> softmax(const std::vector<float>& input);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);
static void print_topk(const std::vector<std::pair<int, float>>& topk, int ground_truth_class);
static const char* lookup(int index);

static cv::Mat croppedImage(const cv::Mat& image, int height, int width) {
  cv::Mat cropped_img;
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
  return cropped_img;
}

static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size) {
  float smallest_side = 256;
  float scale =
      smallest_side / ((image.rows > image.cols) ? image.cols : image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size(image.cols * scale, image.rows * scale));
  return croppedImage(resized_image, size.height, size.width);
}

// preprocessing for resnet50
static void setImageRGB(const cv::Mat& image, void* data, float fix_scale) {
/*   mean value and scale are dataset specific, we need to calculate them before running the model.
  float mean[3] = {B, G, R} values of ImageNet dataset multiplied by 255 (original values are mean=[0.485, 0.456, 0.406] in RGB order).
  float mean[3] = {103.53f, 116.28f, 123.675f};
  float scales[3] = {stdB, stdG, stdR} values of ImageNet dataset calculated as stdB = 1/255/originalstdChannel (original values are std=[0.229, 0.224, 0.225] in RGB order).
  float scales[3] = {0.017429f, 0.017507f, 0.01712475f}; */
  
  // # Mean: tensor([0.0419, 0.0355, 0.0410]), Std: tensor([0.0959, 0.0913, 0.0771]), Total pixels: 7588451567  --> ESTE ES EL UTILIZADO en ORDEN RGB
  float mean[3] = {10.455f, 9.0525f, 10.6845f};
  float scales[3] = {0.05086340632f, 0.0429525589f, 0.04089226932f};

  signed char* data1 = (signed char*)data;
  int c = 0;
  for (auto row = 0; row < image.rows; row++) {
    for (auto col = 0; col < image.cols; col++) {
      auto v = image.at<cv::Vec3b>(row, col);
      // substract mean value and times scale;
      auto B = (float)v[0];
      auto G = (float)v[1];
      auto R = (float)v[2];
      auto nB = (B - mean[0]) * scales[0] * fix_scale;
      auto nG = (G - mean[1]) * scales[1] * fix_scale;
      auto nR = (R - mean[2]) * scales[2] * fix_scale;
      nB = std::max(std::min(nB, 127.0f), -128.0f);
      nG = std::max(std::min(nG, 127.0f), -128.0f);
      nR = std::max(std::min(nR, 127.0f), -128.0f);
      data1[c++] = (int)(nR);
      data1[c++] = (int)(nG);
      data1[c++] = (int)(nB);
    }
  }
}
// fix_point to scale for input tensor
static float get_input_scale(const xir::Tensor* tensor) {
  int fixpos = tensor->template get_attr<int>("fix_point");
  return std::exp2f(1.0f * (float)fixpos);
}
// fix_point to scale for output tensor
static float get_output_scale(const xir::Tensor* tensor) {
  int fixpos = tensor->template get_attr<int>("fix_point");
  return std::exp2f(-1.0f * (float)fixpos);
}

class ImageLoader {
public:
    ImageLoader(const std::vector<std::string>& imagePaths, const std::vector<int>& labels, int batchSize, cv::Size imgSize)
        : imagePaths(imagePaths), labels(labels), batchSize(batchSize), imgSize(imgSize), index(0) {}

    bool nextBatch(std::vector<cv::Mat>& batchImages, std::vector<int>& batchLabels) {
        if (index >= imagePaths.size()) {
            return false;
        }

        for (int i = 0; i < batchSize; ++i) {
            if (index >= imagePaths.size()) {
                break;
            }

            cv::Mat img = cv::imread(imagePaths[index]);
            cv::resize(img, img, imgSize);
            batchImages.push_back(img);
            batchLabels.push_back(labels[index]);
            ++index;
        }

        return true;
    }

private:
    std::vector<std::string> imagePaths;
    std::vector<int> labels;
    int batchSize;
    cv::Size imgSize;
    size_t index;
};

std::vector<std::string> getAllImagePaths(const std::string& rootFolder) {
    std::vector<std::string> imagePaths;
    std::filesystem::path rootPath(rootFolder);

    if (!std::filesystem::exists(rootPath) || !std::filesystem::is_directory(rootPath)) {
        std::cout << "Invalid root folder" << std::endl;
        return imagePaths;
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(rootPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            imagePaths.push_back(entry.path().string());
        }
    }

    return imagePaths;
}

std::vector<int> extractNumbersFromPaths(const std::vector<std::string>& imagePaths) {
    std::vector<int> numbers;
    for (const auto& path : imagePaths) {
        std::string parentDirName = std::filesystem::path(path).parent_path().filename().string();
        std::string numberStr = parentDirName.substr(0, parentDirName.find('_'));
        int number = std::stoi(numberStr);
        numbers.push_back(number);
    }
    return numbers;
}

void printProgress(double percentage) {
    const int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * percentage;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(percentage * 100.0) << " %\r";
    std::cout.flush();
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "usage: " << argv[0]
         << " <resnet50.xmodel> <folder_path> [limit] [random]\n";
    return 0;
  }
  auto xmodel_file = std::string(argv[1]);
  std::string rootFolder = (argc > 2) ? std::string(argv[2]) : "../DYB-original/test";
  int limit = (argc > 3 && std::isdigit(argv[3][0])) ? std::stoi(argv[3]) : 0;
  bool random = (argc > 3 && std::string(argv[argc-1]) == "random");
  int batch_size = 1;

  // Get all the image paths and labels  
  std::vector<std::string> imagePaths = getAllImagePaths(rootFolder);
  std::vector<int> labels = extractNumbersFromPaths(imagePaths);

  // Initialize total time and image count
  auto start = std::chrono::high_resolution_clock::now();

  // Create my custom ImageLoader mimicking the one from PyTorch
  ImageLoader loader(imagePaths, labels, batch_size, cv::Size(224, 224));
  // Update the value of limit if it is 0
  if (limit == 0) {
    limit = labels.size();
  }

  //  create dpu runner
  auto graph = xir::Graph::deserialize(xmodel_file);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* subgraph = nullptr;
  for (auto c : root->children_topological_sort()) {
    CHECK(c->has_attr("device"));
    if (c->get_attr<std::string>("device") == "DPU") {
      subgraph = c;
      break;
    }
  }
  auto attrs = xir::Attrs::create();
  std::unique_ptr<vart::RunnerExt> runner =
      vart::RunnerExt::create_runner(subgraph, attrs.get());

  // get input & output tensor buffers
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();
  CHECK_EQ(input_tensor_buffers.size(), 1u) << "only support resnet50 model";
  CHECK_EQ(output_tensor_buffers.size(), 1u) << "only support resnet50 model";

  // get input_scale & output_scale
  auto input_tensor = input_tensor_buffers[0]->get_tensor();
  auto input_scale = get_input_scale(input_tensor);

  auto output_tensor = output_tensor_buffers[0]->get_tensor();
  auto output_scale = get_output_scale(output_tensor);

  auto dpu_batch = input_tensor->get_shape().at(0);
  auto height = input_tensor->get_shape().at(1); // 224 for resnet50
  auto width = input_tensor->get_shape().at(2); // by 224 for resnet50

  // Iterar sobre los lotes de imágenes
  std::vector<cv::Mat> batchImages;
  std::vector<int> batchLabels;
  int iteration = 0;
  // loop for running input images
  while (loader.nextBatch(batchImages, batchLabels) && (iteration < limit)){
      // Aquí puedes procesar el lote de imágenes
      for (size_t i = (batchImages.size() - batch_size); i < batchImages.size(); ++i) {

          // Batch Size
          auto run_batch = dpu_batch;
          auto images = std::vector<cv::Mat>(run_batch);

          // preprocessing, resize the input image to a size of 224 x 224 (the model's input size)
          uint64_t data_in = 0u;
          size_t size_in = 0u;
          for (auto batch_idx = 0; batch_idx < run_batch; ++batch_idx) {
            images[batch_idx] = preprocess_image(batchImages[i + batch_idx],
                                                cv::Size(width, height));
            // set the input image and preprocessing
            std::tie(data_in, size_in) =
                input_tensor_buffers[0]->data(std::vector<int>{batch_idx, 0, 0, 0});
            CHECK_NE(size_in, 0u);
            setImageRGB(images[batch_idx], (void*)data_in, input_scale);
          }

          // sync data for input
          for (auto& input : input_tensor_buffers) {
            input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                        input->get_tensor()->get_shape()[0]);
          }
          // start the dpu
          auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
          auto status = runner->wait((int)v.first, -1);
          CHECK_EQ(status, 0) << "failed to run dpu";
          // sync data for output
          for (auto& output : output_tensor_buffers) {
            output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                        output->get_tensor()->get_shape()[0]);
          }

          // postprocessing
          for (auto batch_idx = 0; batch_idx < run_batch; ++batch_idx) {
            auto topk = post_process(output_tensor_buffers[0], output_scale, batch_idx);
            // print the result
            print_topk(topk, batchLabels[i]);
          }

          printProgress(double(iteration) / limit);
          iteration += run_batch;
      }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;

  // Calculate average FPS
  std::cout << std::endl << std::endl << "Total number of images: " << iteration << std::endl;
  std::cout << "Time elapsed: " << diff.count() << " seconds" << std::endl;

  double avg_fps = iteration / diff.count();
  std::cout << std::endl << "Average FPS: " << avg_fps << std::endl;

  // Calculate precision
  int true_positives = 0;
  int false_positives = 0;
  int false_negatives = 0;

  for (int i = 0; i < ground_truth_classes.size(); i++) {
      if (ground_truth_classes[i] == inferred_classes[i]) {  // True Positive
          true_positives += 1;
      } else if (std::find(ground_truth_classes.begin(), ground_truth_classes.end(), inferred_classes[i]) == ground_truth_classes.end()) {  // False Positive
          false_positives += 1;
      } else {  // False Negative
          false_negatives += 1;
      }
  }

  int total = true_positives + false_positives + false_negatives;
  double accuracy = static_cast<double>(true_positives) / total;
  double precision = static_cast<double>(true_positives) / (true_positives + false_positives);
  double recall = static_cast<double>(true_positives) / (true_positives + false_negatives);
  double f1_score = 2 * (precision * recall) / (precision + recall);

  // Print the results
  std::cout << "Accuracy: " << accuracy*100 << "%, Precision: " << precision*100 << "%, Recall: " << recall*100 << "%, F1 Score: " << f1_score*100 << "%" << std::endl;
 
  return 0;
}

static std::vector<std::pair<int, float>> post_process(
    vart::TensorBuffer* tensor_buffer, float scale, int batch_idx) {
  // int to float & run softmax
  auto softmax_input =
      convert_fixpoint_to_float(tensor_buffer, scale, batch_idx);
  auto softmax_output = softmax(softmax_input);
  // print top5
  constexpr int TOPK = 1;
  return topk(softmax_output, TOPK);
}

static std::vector<float> convert_fixpoint_to_float(
    vart::TensorBuffer* tensor_buffer, float scale, int batch_idx) {
  uint64_t data = 0u;
  size_t size = 0u;
  std::tie(data, size) = tensor_buffer->data(std::vector<int>{batch_idx, 0});
  signed char* data_c = (signed char*)data;
  auto ret = std::vector<float>(size);
  transform(data_c, data_c + size, ret.begin(),
            [scale](signed char v) { return ((float)v) * scale; });
  return ret;
}

static std::vector<float> softmax(const std::vector<float>& input) {
  auto output = std::vector<float>(input.size());
  std::transform(input.begin(), input.end(), output.begin(), expf);
  auto sum = accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
                 [sum](float v) { return v / sum; });
  return output;
}

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score, int K) {
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [&score](int a, int b) { return score[a] > score[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  std::transform(
      indices.begin(), indices.begin() + K, ret.begin(),
      [&score](int index) { return std::make_pair(index, score[index]); });
  return ret;
}

static void print_topk(const std::vector<std::pair<int, float>>& topk, int ground_truth_class) {

  std::string lookupResult = lookup(topk[0].first);
  std::string classStr = lookupResult.substr(0, lookupResult.find("_"));
  int inferred_class_id = std::stoi(classStr);

  // Append the inferred and real class to each vector
  inferred_classes.push_back(inferred_class_id);
  ground_truth_classes.push_back(ground_truth_class);

/*   std::cout << "Top[" << 0 << "] Inferenced --> " 
  << std::setw(2) << std::setfill('0') << inferred_class_id << ", " 
  << std::setw(2) << std::setfill('0') << ground_truth_class
  << " <-- Ground Truth." << std::endl; */
}

static const char* lookup(int index) {
  static const char* table[] = {
#include "plankton_list.inc"
  };

  if (index < 0) {
    return "";
  } else {
    return table[index];
  }
};