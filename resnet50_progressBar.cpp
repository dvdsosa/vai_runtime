#include <iostream>
#include <chrono>
#include <thread>

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

int main() {
    const int numIterations = 100;
    for (int i = 0; i <= numIterations; ++i) {
        printProgress(double(i) / numIterations);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Simulate work
    }
    std::cout << std::endl;
    return 0;
}