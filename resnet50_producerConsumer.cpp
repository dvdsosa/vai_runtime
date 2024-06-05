#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

// Safe Queue for threads
template <typename T>
class SafeQueue {
private:
    std::queue<Image> queue;
    std::mutex mutex;
    std::condition_variable cond;
    bool isProducerFinished = false;
public:
    void push(Image value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(value);
        cond.notify_one();
    }

    Image pop() {
        std::unique_lock<std::mutex> lock(mutex);
        while(queue.empty() && !isProducerFinished) {
            cond.wait(lock);
        }
        if(queue.empty() && isProducerFinished) {
            return Image();  // Devuelve un Image vacío si la cola está vacía y el productor ha terminado
        }
        Image value = queue.front();
        queue.pop();
        return value;
    }

    void producerFinished() {
        std::lock_guard<std::mutex> lock(mutex);
        isProducerFinished = true;
        cond.notify_all();
    }
};

class Image {
public:
    std::vector<cv::Mat> batchImages;
    std::vector<int> batchLabels;
    std::vector<std::string> batchImagePaths;
};

// Task B (producer)
void tareaB(SafeQueue<Image>& queue, int limite) {
    int count = 0;
    while(count < limite) {
        std::vector<cv::Mat> batchImages;
        std::vector<int> batchLabels;
        std::vector<std::string> batchImagePaths;

        cargarImagen(batchImages, batchLabels, batchImagePaths);

        Image image;
        image.batchImages = batchImages;
        image.batchLabels = batchLabels;
        image.batchImagePaths = batchImagePaths;

        queue.push(image);
        count++;
    }
    queue.producerFinished();  // Avisa que tareaB ha terminado
}

// Tasks A y C (consumers)
void tareaAC(SafeQueue<Image>& queue) {
    while(true) {
        Image image = queue.pop();

        if(image.batchImages.empty() && image.batchLabels.empty() && image.batchImagePaths.empty()) {
            break;  // Termina el bucle si recibe un Image vacío
        }

        // Tarea A
        normalizarImagen(image);
        // Tarea C
        procesarImagen(image);
    }
}

int main() {
    SafeQueue<Image> queue;
    int limite = 5;

    std::thread t1(tareaB, std::ref(queue), limite);
    std::thread t2(tareaAC, std::ref(queue));

    t1.join();
    t2.join();

    return 0;
}
