import sys
sys.path.insert(0, r'A:\AI\InfernoTactics\infernotactics\src')
from train.train_relative import get_device
print('Device (auto):', get_device())
print('Device (force_cpu):', get_device(force_cpu=True))