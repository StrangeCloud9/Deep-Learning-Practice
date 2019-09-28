import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python test_mnist.py <predict_file>")
        exit()

    pred_filename = sys.argv[1]
    pred_labels = np.loadtxt(pred_filename, delimiter=",")
    gt_labels = np.load("test_labels.npy")

    assert pred_labels.shape == gt_labels.shape, "shape does not match!"
    accuracy = np.mean(pred_labels == gt_labels)
    print("test accuracy: {}".format(accuracy))

