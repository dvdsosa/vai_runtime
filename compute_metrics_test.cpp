#include <Python.h>
#include <iostream>
#include <vector>

void compute_metrics(const std::vector<int>& ground_truth_classes, const std::vector<int>& inferred_classes, int num_classes) {
    Py_Initialize();

    // Add the current directory to the Python path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");

    PyObject* pName = PyUnicode_DecodeFSDefault("compute_metrics");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "compute_metrics");

        if (PyCallable_Check(pFunc)) {
            PyObject* pGroundTruth = PyList_New(ground_truth_classes.size());
            PyObject* pInferred = PyList_New(inferred_classes.size());

            for (size_t i = 0; i < ground_truth_classes.size(); ++i) {
                PyList_SetItem(pGroundTruth, i, PyLong_FromLong(ground_truth_classes[i]));
                PyList_SetItem(pInferred, i, PyLong_FromLong(inferred_classes[i]));
            }

            PyObject* pNumClasses = PyLong_FromLong(num_classes);
            PyObject* pArgs = PyTuple_Pack(3, pGroundTruth, pInferred, pNumClasses);
            PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != nullptr) {
                double acc = PyFloat_AsDouble(PyTuple_GetItem(pValue, 0));
                double prec = PyFloat_AsDouble(PyTuple_GetItem(pValue, 1));
                double rec = PyFloat_AsDouble(PyTuple_GetItem(pValue, 2));
                double f1 = PyFloat_AsDouble(PyTuple_GetItem(pValue, 3));

                std::cout << "Accuracy: " << acc * 100 << "%" << std::endl;
                std::cout << "Precision: " << prec * 100 << "%" << std::endl;
                std::cout << "Recall: " << rec * 100 << "%" << std::endl;
                std::cout << "F1 Score: " << f1 * 100 << "%" << std::endl;

                Py_DECREF(pValue);
            } else {
                PyErr_Print();
            }

            Py_DECREF(pGroundTruth);
            Py_DECREF(pInferred);
            Py_DECREF(pNumClasses);
        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
    }

    Py_Finalize();
}

int main() {
    std::vector<int> inferred_classes = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0};
    std::vector<int> ground_truth_classes = {1, 0, 1, 1, 0, 1, 0, 0, 1, 0};
    int num_classes = 2;

    compute_metrics(ground_truth_classes, inferred_classes, num_classes);

    return 0;
}
