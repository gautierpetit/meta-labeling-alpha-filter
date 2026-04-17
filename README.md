# Experimental Branch — Meta-Labeling Alpha Filter

This branch preserves research experiments that were not carried into the final `main` branch implementation.

It is kept for transparency and reference only. It contains exploratory stacker variants, intermediate ideas, and partially retained research artifacts that supported the final project design.

> **Status:** Experimental / research-only. This branch is not the canonical project version and may contain incomplete, superseded, or partially documented components.  
> **Licensing:** The source code in this branch is licensed under the **BSD 3-Clause License**. Project documentation and the case report remain licensed separately under **CC BY-NC-ND 4.0**.

## What differs from `main`

Compared with the final `main` branch, this branch includes experimentation with alternative meta-model designs, including:

- MLP stacker prototype on the full feature set
- MLP stacker prototype on a leaner feature set
- simple blender prototype
- linear regression prototype
- classwise blender development iterations
- meta-features constructed from base LightGBM and MLP predictions

## Documentation

- Full case report (PDF): see the [`main` branch report](https://github.com/gautierpetit/meta-labeling-alpha-filter/blob/main/docs/Meta-Labeling%20Alpha%20Filter%20-%20Case%20Report.pdf)
- Main project overview and canonical implementation: see the [`main` branch README](https://github.com/gautierpetit/meta-labeling-alpha-filter/tree/main)

## Note

Readers looking for the final project should use the `main` branch. This branch is retained only to document the research path that led to the final stacker design.