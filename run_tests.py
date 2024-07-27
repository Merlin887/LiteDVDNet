import torch
from infrastructure.test_runner import TestRunner

if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True
    test_suite_path = './test_options/7_1_cached_inference.yaml'
    test_runner = TestRunner(test_suite_path)
    test_runner.run()





