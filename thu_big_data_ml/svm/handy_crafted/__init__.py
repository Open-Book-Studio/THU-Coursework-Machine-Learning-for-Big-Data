import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'kernel',
        'linear',
    },
    submod_attrs={
        'kernel': [
            'KernelSupportVectorClassifier',
        ],
        'linear': [
            'BinaryHingeLoss',
            'HingeSupportVectorClassifier',
            'MultiClassHingeLoss',
            'Strategy',
            'get_max_values_without_true',
            'scheduler_lmd_leon_bottou_asgd',
            'scheduler_lmd_leon_bottou_sgd',
            'separate_weight_decay',
        ],
    },
)

__all__ = ['BinaryHingeLoss', 'HingeSupportVectorClassifier',
           'KernelSupportVectorClassifier', 'MultiClassHingeLoss', 'Strategy',
           'get_max_values_without_true', 'kernel', 'linear',
           'scheduler_lmd_leon_bottou_asgd', 'scheduler_lmd_leon_bottou_sgd',
           'separate_weight_decay']
