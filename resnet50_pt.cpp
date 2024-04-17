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
#include <memory>
#include <numeric>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xir/graph/graph.hpp>

#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vitis/ai/collection_helper.hpp"

#include <random>
#include <filesystem>
namespace fs = std::filesystem;

static std::vector<std::pair<int, float>> post_process(
    vart::TensorBuffer* tensor_buffer, float scale, int batch_idx);
static std::vector<float> convert_fixpoint_to_float(vart::TensorBuffer* tensor,
                                                    float scale, int batch_idx);

static std::vector<float> softmax(const std::vector<float>& input);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);
static void print_topk(const std::vector<std::pair<int, float>>& topk);
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
  // mean value and scale are dataset specific, we need to calculate them before running the model.
  //float mean[3] = {103.53f, 116.28f, 123.675f}; // [B, G, R] mean values of ImageNet dataset multiplied by 255 (original values mean=[0.485, 0.456, 0.406] in RGB order).
  //float scales[3] = {0.017429f, 0.017507f, 0.01712475f}; // [stdB, stdG, stdR] std values of ImageNet dataset calculated as stdB = 1/255/originalstdChannel (original values std=[0.229, 0.224, 0.225] in RGB order).
  // These are the original values for DYB-PlanktonNet dataset --mean "0.0361, 0.0326, 0.0357" --std "0.0997, 0.0968, 0.0858".
  // Below converted as stated in the first lines.
  float mean[3] = {9.10f, 8.31f, 9.20f};
  float scales[3] = {0.045706f, 0.040512f, 0.039334f};

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

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "usage: " << argv[0]
         << " <resnet50.xmodel> <folder_path> [limit] [random]\n";
    return 0;
  }
  auto xmodel_file = std::string(argv[1]);
  std::string folder_path = std::string(argv[2]);
  int limit = (argc > 3 && std::isdigit(argv[3][0])) ? std::stoi(argv[3]) : 100;
  bool random = (argc > 3 && std::string(argv[argc-1]) == "random");

  // read input images
  std::vector<cv::Mat> input_images;
  std::vector<std::string> lines;
  std::vector<std::string> filenames;

  for (const auto & entry : fs::directory_iterator(folder_path)) {
    if (entry.path().extension() == ".jpg") {
      filenames.push_back(entry.path().string());
    }
  }

  if (random) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(filenames.begin(), filenames.end(), g);
  }

  for (int i = 0; i < std::min(limit, (int)filenames.size()); i++) {
    cv::Mat img = cv::imread(filenames[i]);
    if (img.empty()) {
      std::cout << "Cannot load image : " << filenames[i] << std::endl;
      continue;
    }
    input_images.push_back(img);
    lines.push_back(filenames[i]);  // Store the filename
  }

  if (input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
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

  auto batch = input_tensor->get_shape().at(0);
  auto height = input_tensor->get_shape().at(1); // 224 for resnet50
  auto width = input_tensor->get_shape().at(2); // by 224 for resnet50

  // Initialize total time and image count
  auto total_time = std::chrono::duration<double>(0);
  int total_images = 0;

  // loop for running input images
  for (auto i = 0; i < input_images.size(); i += batch) {
    auto start = std::chrono::high_resolution_clock::now();

    auto run_batch = std::min(((int)input_images.size() - i), batch);
    auto images = std::vector<cv::Mat>(run_batch);

    // preprocessing, resize the input image to a size of 224 x 224 (the model's input size)
    uint64_t data_in = 0u;
    size_t size_in = 0u;
    for (auto batch_idx = 0; batch_idx < run_batch; ++batch_idx) {
      images[batch_idx] = preprocess_image(input_images[i + batch_idx],
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

    // std::cout << "Batch size: " << run_batch << "\n";

    // postprocessing
    for (auto batch_idx = 0; batch_idx < run_batch; ++batch_idx) {
      std::cout << "Image name: " << lines[i + batch_idx] << "\n";
      auto topk = post_process(output_tensor_buffers[0], output_scale, batch_idx);
      // print the result
      //print_topk(topk);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;

    // Calculate last image FPS
    //double fps = 1.0 / diff.count();
    //std::cout << "FPS: " << fps << std::endl << std::endl;

    total_time += diff;
    total_images += run_batch;
  }

  // Calculate average FPS
  double avg_fps = total_images / total_time.count();
  std::cout << "Average FPS: " << avg_fps << std::endl;

  return 0;
}

static std::vector<std::pair<int, float>> post_process(
    vart::TensorBuffer* tensor_buffer, float scale, int batch_idx) {
  // int to float & run softmax
  auto softmax_input =
      convert_fixpoint_to_float(tensor_buffer, scale, batch_idx);
  auto softmax_output = softmax(softmax_input);
  // print top5
  constexpr int TOPK = 5;
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

static void print_topk(const std::vector<std::pair<int, float>>& topk) {
  for (const auto& v : topk) {
    std::cout << std::setiosflags(std::ios::left) << std::setw(11)
              << "score[" + std::to_string(v.first) + "]"
              << " =  " << std::setw(12) << v.second
              << " text: " << lookup(v.first) << std::resetiosflags(std::ios::left)
              << std::endl;
  }
  //commented so FPS: xx.xxxx data appears just below last line of accuracies.
  //std::cout << std::endl;
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