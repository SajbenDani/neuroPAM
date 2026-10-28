# DeepComBat: vendored copy and patches

Source: <https://github.com/hufengling/DeepComBat>, commit `bf0eda1c8c6b4690239351065f6ae92c316b4ff4`
(MIT licence, see `LICENSE` and `LICENSE.md`). Method: Hu et al. (2024), Human Brain Mapping 45(11):
e26708.

Only the files needed to install the R package are copied (`DESCRIPTION`, `NAMESPACE`, `R/`, `man/`,
licence files, `README.md`). Four lines of `R/deepcombat_dataloader.R` were changed so that the
package runs with a single covariate and a data-frame covariate table. The algorithm is untouched.

| Line | Original | Patched | Why |
|---|---|---|---|
| 56, 136 | `model.matrix(covariate_formula, ...)[, -1]` | `[, -1, drop = FALSE]` | With one covariate (here the diagnosis) R drops the result to a vector and the next call (`apply(df, 2, ...)`, `ncol(df)`) fails. |
| 42, 69 | `covariate_matrix[batch_formula[[2]]]` | `covariate_matrix[, as.character(batch_formula[[2]])]` | Indexing a data frame with a formula symbol raises "invalid subscript type 'symbol'". The package's own `create_model_matrices_from_train` already uses the patched form (line 122). |

Install into the local R library with

```
R CMD INSTALL third_party/DeepComBat
```

The package also needs `torch` (R; run `torch::install_torch()` once), `coro`, `dplyr` and the R
`neuroCombat` package (`remotes::install_github("Jfortin1/neuroCombat_Rpackage")`).

The Python bridge is `src/vae_harmonization/baselines/deep.py`. It follows the README's recipe for
external harmonization: `deepcombat_setup(..., use_default_optim = FALSE)`, training with
`optim_adamw(lr = 0.01)` and the default schedule (5 pre-training epochs, 30 epochs of cyclic KL
annealing to lambda = 0.1, 5 final epochs), `deepcombat_harmonize` on the training data, then
`deepcombat_setup_from_train` and `deepcombat_harmonize(setup, model, harmonized_train)` for new
subjects. All other settings are the package defaults (batch size 64, three hidden layers, latent size
the power of two nearest a quarter of the number of features).
