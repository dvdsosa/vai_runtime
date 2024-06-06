#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

class Image {
public:
    std::vector<cv::Mat> batchImages;
    std::vector<int> batchLabels;
    std::vector<std::string> batchImagePaths;
};

class Semaphore {
private:
    std::mutex mutex;
    std::condition_variable cv;
    int count;

public:
    Semaphore(int count_ = 0) : count(count_) {}

    void notify() {
        std::unique_lock<std::mutex> lock(mutex);
        count++;
        cv.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        while(count == 0) {
            cv.wait(lock);
        }
        count--;
    }

    bool try_acquire() {
        std::unique_lock<std::mutex> lock(mutex);
        if(count > 0) {
            count--;
            return true;
        } else {
            return false;
        }
    }
};


// Cola segura para hilos
template <typename T>
class SafeQueue {
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(value);
        cond.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        while(queue.empty()) {
            cond.wait(lock);
        }
        T value = queue.front();
        queue.pop();
        return value;
    }
};

// Tarea B (productor)
void tareaB(SafeQueue<Image>& queue, Semaphore& sem) {
    while(true) {
        //Image image = cargarImagen();
        //queue.push(image);
        sem.notify(); // Notificar la finalización de la tarea B
    }
}

// Tarea AC (consumidor y productor)
void tareaAC(SafeQueue<Image>& inputQueue, SafeQueue<Image>& outputQueue, Semaphore& semB, Semaphore& semAC) {
    while(true) {
        if(semB.try_acquire()) { // Comprobar si la tarea B ha terminado
            break; // Si la tarea B ha terminado, terminar la tarea AC
        }
        //Image image = inputQueue.pop();
        // Tarea A
        //normalizarImagen(image);
        // Tarea C
        //Processed processed = procesarImagen(image);
        //outputQueue.push(processed);
        semAC.notify(); // Notificar la finalización de la tarea AC
    }
}

// Tarea X (consumidor)
void tareaX(SafeQueue<Image>& queue, Semaphore& semAC) {
    while(true) {
        if(semAC.try_acquire()) { // Comprobar si la tarea AC ha terminado
            break; // Si la tarea AC ha terminado, terminar la tarea X
        }
        //Processed processed = queue.pop();
        // Código de la tarea X
    }
}

int main() {
    SafeQueue<Image> queue1;
    SafeQueue<Image> queue2;
    Semaphore semB(0), semAC(0);

    std::thread t1(tareaB, std::ref(queue1), std::ref(semB));
    std::thread t2(tareaAC, std::ref(queue1), std::ref(queue2), std::ref(semB), std::ref(semAC));
    std::thread t3(tareaX, std::ref(queue2), std::ref(semAC));

    t1.join();
    t2.join();
    t3.join();

    return 0;
}
