"""
avocado_ripeness - non-destructive avocado ripeness estimation from photos.

MMA3001 Numerical Methods and Machine Learning, individual project.

Modules
-------
- `avocado_ripeness.features`       : fruit segmentation and HSV colour features
- `avocado_ripeness.baseline`       : single-feature colour-threshold baseline classifier
- `avocado_ripeness.classifier`     : linear regression (final model) and alternative models, kiosk message
- `avocado_ripeness.ripening_model` : logistic ripening curve fitting (scipy curve_fit), Q10

Run the full experiment with ``python scripts/run_pipeline.py --image-dir <folder>``.
"""

__version__ = "1.0.0"
