import os
import sys

python_cmd = "python3"
testcase_dir = "testcase"
tests = [
    ["adder",       "1_adder.py"],
    ["initializer", "2_init.py"],
    ["assign",      "3_assign.py"],
    ["context",     "4_context.py"],
    ["autodiff",    "5_mnist_grad.py"],
    ["GD optimizer",   "6_gd_optimizer.py"],
    ["multilayer perceptron", "7_ml_perceptron.py"],
    ["Adam optimizer",        "8_adam_optimizer.py"],
    ["CNN Check",       "9_cnn_check.py"],
    ["CNN GD",       "10_cnn_gd.py"],
    ["CNN Adam with Dropout",       "11_cnn_adam.py"]
]

def main(model_name):
    for i, item in enumerate(tests):
        test_name = item[0]
        file_name = item[1]

        print("========== test %d %s ==========" % (i+1, test_name))

        os.system("cp %s %s" % (os.path.join(testcase_dir, file_name),
                                                    file_name))
        os.system("sed -i 's/your_model/%s/' %s" % (model_name, file_name))
        ret = os.system("%s %s" % (python_cmd, file_name))
        if ret != 0:
            exit(0)
        os.system("rm %s" % (file_name))
        print("")
    print("Pass all!")

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: python run_test.py name_of_your_model")
        exit(0)
    main(sys.argv[1])
