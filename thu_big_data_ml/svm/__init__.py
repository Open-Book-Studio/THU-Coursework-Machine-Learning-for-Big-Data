import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'handy_crafted',
        'infra',
        'kernel_hpo',
        'use_lib',
        'vis',
    },
    submod_attrs={
        'handy_crafted': [
            'BinaryHingeLoss',
            'HingeSupportVectorClassifier',
            'KernelSupportVectorClassifier',
            'MultiClassHingeLoss',
            'Strategy',
            'get_max_values_without_true',
            'kernel',
            'linear',
            'scheduler_lmd_leon_bottou_asgd',
            'scheduler_lmd_leon_bottou_sgd',
            'separate_weight_decay',
        ],
        'infra': [
            'ReturnType',
            'get_torch_dataset',
            'make_train_val_test',
            'process_sklearn_dataset_dict',
            'sklearn_to_X_y_categories',
        ],
        'kernel_hpo': [
            'SupportVectorClassifierConfig',
            'dict_to_dataclass',
            'draw_probs',
            'evaluate_svm',
            'fixed_meta_params',
            'frozen_rvs',
            'objective_svm',
            'sqlite_url',
            'study',
            'study_path',
        ],
        'vis': [
            'make_meshgrid',
            'plot_binary_classification_2d',
            'plot_contours',
            'try_svm_and_plot_for_binary_2d',
        ],
    },
)

__all__ = ['BinaryHingeLoss', 'HingeSupportVectorClassifier',
           'KernelSupportVectorClassifier', 'MultiClassHingeLoss',
           'ReturnType', 'Strategy', 'SupportVectorClassifierConfig',
           'dict_to_dataclass', 'draw_probs', 'evaluate_svm',
           'fixed_meta_params', 'frozen_rvs', 'get_max_values_without_true',
           'get_torch_dataset', 'handy_crafted', 'infra', 'kernel',
           'kernel_hpo', 'linear', 'make_meshgrid', 'make_train_val_test',
           'objective_svm', 'plot_binary_classification_2d', 'plot_contours',
           'process_sklearn_dataset_dict', 'scheduler_lmd_leon_bottou_asgd',
           'scheduler_lmd_leon_bottou_sgd', 'separate_weight_decay',
           'sklearn_to_X_y_categories', 'sqlite_url', 'study', 'study_path',
           'try_svm_and_plot_for_binary_2d', 'use_lib', 'vis']
