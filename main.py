import torch
import matplotlib

if __name__ == '__main__':
    print('Setup complete. Using torch %s %s' % (
        torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
