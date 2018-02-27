class Config(object):
    """Holds model hyperparams and data information. Copied from 224n"""
    
    batch_size = 64
    n_epochs = 5
    lr = 0.02
    n_test_samples = 5
    results_dir='./experiments/deconvolution_model/'
    