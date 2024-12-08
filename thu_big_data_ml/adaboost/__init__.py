import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'vis',
    },
    submod_attrs={
        'vis': [
            'calculate_gini_index',
            'calculate_gini_index_for_subset',
            'plot_binary_classification_3d',
        ],
    },
)

__all__ = ['calculate_gini_index', 'calculate_gini_index_for_subset',
           'plot_binary_classification_3d', 'vis']
