#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include <random>

class ImageLoader {
public:
    ImageLoader(const std::vector<std::string>& imagePaths, const std::vector<int>& labels, int batchSize, cv::Size imgSize)
        : imagePaths(imagePaths), labels(labels), batchSize(batchSize), imgSize(imgSize), index(0) {}

    bool nextBatch(std::vector<cv::Mat>& batchImages, std::vector<int>& batchLabels, std::vector<std::string>& batchImagePaths) {
        if (index >= imagePaths.size()) {
            return false;
        }

        //std::cout << "Current index class ImageLoader: " << index << std::endl;
        for (int i = 0; i < batchSize; ++i) {
            if (index >= imagePaths.size()) {
                break;
            }

            cv::Mat img = cv::imread(imagePaths[index]);
            cv::resize(img, img, imgSize);
            batchImages.push_back(img);
            batchLabels.push_back(labels[index]);
            batchImagePaths.push_back(imagePaths[index]);
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

std::vector<std::string> getAllImagePaths(const std::string& rootFolder, const std::string& order = "None") {
    std::vector<std::string> imagePaths;
    std::filesystem::path rootPath(rootFolder);

    if (!std::filesystem::exists(rootPath) || !std::filesystem::is_directory(rootPath)) {
        std::cout << "Invalid root folder" << std::endl;
        return imagePaths;
    }

    if (order == "random") {
        std::vector<std::string> tempPaths;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(rootPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                tempPaths.push_back(entry.path().string());
            }
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(tempPaths.begin(), tempPaths.end(), g);
        imagePaths = tempPaths;
    } else {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(rootPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                imagePaths.push_back(entry.path().string());
            }
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

int main(int argc, char* argv[]) {

    auto start = std::chrono::high_resolution_clock::now();

    std::string rootFolder = "../DYB-original/test";
    std::vector<std::string> imagePaths = getAllImagePaths(rootFolder, "random");
    std::vector<int> labels = extractNumbersFromPaths(imagePaths);

    for (const auto& imagePath : imagePaths) {
        //std::cout << imagePath << std::endl;
    }
    std::cout << "Total size of imagePaths: " << imagePaths.size() << std::endl;

    for (const auto& number : labels) {
        //std::cout << number << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time elapsed: " << duration.count() << " ms" << std::endl;
    
    int batch_size = 4;
    // Create the ImageLoader
    ImageLoader loader(imagePaths, labels, batch_size, cv::Size(224, 224));

    // Iterate over the batches of images
    std::vector<cv::Mat> batchImages;
    std::vector<int> batchLabels;
    std::vector<std::string> batchImagePaths;
    int limit = 16; // Set the limit here
    int iteration = 0;
    bool continueLoading = loader.nextBatch(batchImages, batchLabels, batchImagePaths);
    int nowBatchSize = batchImages.size();

    while ( continueLoading && (iteration < limit)){
        start = std::chrono::high_resolution_clock::now();
        
        // Here you can process the batch of images
        for (size_t i = (nowBatchSize - batch_size); i < nowBatchSize; ++i) {
            std::cout <<  "VALOR DEL INDICE i: " << i << "<-- DEL BUCLE" << std::endl;
            std::cout << batchImagePaths[i] << std::endl;
            //std::cout << "Batch size: " << batchImages.size() << std::endl;

            if (i == (nowBatchSize - 1)) {
                continueLoading = loader.nextBatch(batchImages, batchLabels, batchImagePaths);
            }
            ++iteration;
        }
       
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        nowBatchSize = batchImages.size();
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time elapsed: " << duration.count() << " ms" << std::endl;
}