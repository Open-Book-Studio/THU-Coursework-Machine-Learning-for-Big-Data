import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'anova',
        'matrix_decomposition',
    },
    submod_attrs={
        'anova': [
            'anova_oneway',
            'auto_anova_for_df',
            'auto_friedman_for_df',
            'auto_kruskal_for_df',
            'draw_box',
            'draw_hist',
            'homogeneity_of_variance',
            'test_normality_group',
        ],
        'matrix_decomposition': [
            'MatrixFactorization',
            'MatrixFactorizationSetting',
            'compute_weighted_sum_on_matrix',
            'draw_metrics_df',
            'ensure_tensor',
            'fixed_meta_params',
            'frozen_rvs',
            'get_X_train_weighted',
            'get_rating_matrix',
            'get_similarities',
            'jax_masked_mse_loss',
            'masked_mse_loss',
            'masked_rmse_loss',
            'objective',
            'sqlite_url',
            'study',
            'study_path',
            'test_normality_small_sample',
            'train_matrix_factorization',
        ],
    },
)

__all__ = ['MatrixFactorization', 'MatrixFactorizationSetting', 'anova',
           'anova_oneway', 'auto_anova_for_df', 'auto_friedman_for_df',
           'auto_kruskal_for_df', 'compute_weighted_sum_on_matrix', 'draw_box',
           'draw_hist', 'draw_metrics_df', 'ensure_tensor',
           'fixed_meta_params', 'frozen_rvs', 'get_X_train_weighted',
           'get_rating_matrix', 'get_similarities', 'homogeneity_of_variance',
           'jax_masked_mse_loss', 'masked_mse_loss', 'masked_rmse_loss',
           'matrix_decomposition', 'objective', 'sqlite_url', 'study',
           'study_path', 'test_normality_group', 'test_normality_small_sample',
           'train_matrix_factorization']
