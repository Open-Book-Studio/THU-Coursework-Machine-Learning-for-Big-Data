import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'hidden_markov_model',
    },
    submod_attrs={
        'hidden_markov_model': [
            'HiddenMarkovModel',
        ],
    },
)

__all__ = ['HiddenMarkovModel', 'hidden_markov_model']
