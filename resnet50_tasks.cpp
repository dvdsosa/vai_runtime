#include <thread>
#include <future>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void tarea1(std::promise<void>& prom) {
    // Código de la tarea 1
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    std::cout << "Tarea 1 completada.\n";
    prom.set_value();
}

void tarea2(std::promise<bool>& prom, std::vector<cv::Mat>& batchImages, std::vector<int>& batchLabels, std::vector<std::string>& batchImagePaths) {
    // Código de la tarea 2
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    std::cout << "Tarea 2 completada.\n";

    bool result = true;
    prom.set_value(result);
    
    // Realizar operaciones con las variables batchImages, batchLabels y batchImagePaths    
}

void tarea3() {
    // Código de la tarea 3
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "Tarea 3 completada.\n";
}

int main(int argc, char* argv[]) {
    std::vector<cv::Mat> batchImages;
    std::vector<int> batchLabels;
    std::vector<std::string> batchImagePaths;

    std::thread t2;
    for (int i = 0; i < 5; ++i) {
        std::promise<void> prom1;
        std::promise<bool> prom2;
        std::future<void> fut1 = prom1.get_future();
        std::future<bool> fut2 = prom2.get_future();

        std::thread t1(tarea1, std::ref(prom1));
        std::thread t3(tarea3);

        if (true) {
            t2 = std::thread(tarea2, std::ref(prom2), std::ref(batchImages), std::ref(batchLabels), std::ref(batchImagePaths));
        }

        fut1.wait();
        if (t2.joinable()) {
            fut2.wait();
        }

        std::cout << "Las tareas 1 y 2 han terminado.\n";

        t1.join();
        if (t2.joinable()) {
            t2.join();
        }
        t3.join();

        if (t2.joinable()) {
            bool result = fut2.get();
            std::cout << "EL RETORNO ES: " << result << std::endl;
        }
    }

    return 0;
}
