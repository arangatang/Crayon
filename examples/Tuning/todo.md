## Electricity dataset
### In progress
python3 tune_mqrnn ; python3 tune_deepfactor.py ; python3 tune_deepar.py ; python3 tune_canonicalRNN.py ; python3 tune_gaussian_process.py ; python3 tune_simplefeedforward.py ; python3 tune_transformer.py 
### DONE GPU
* 2.3 [tune_canonicalRNN](/home/leonardo/Documents/crayon/tuning/canonicalRNN/grid_search_summary_March-21-2021--20-01-34.json) 
* 0.97 [tune_simplefeedforward](/home/leonardo/Documents/crayon/tuning/simplefeedforward/grid_search_summary_March-21-2021--05-40-32.json)
* 0.57 [tune_deepar](/home/leonardo/Documents/crayon/tuning/deepar/grid_search_summary_March-21-2021--09-31-53.json)
* 9.84 [tune_deepfactor](/home/leonardo/Documents/crayon/tuning/deepFactorEstimator/grid_search_summary_March-25-2021--18-45-30.json)
* 0.84 [tune_nbeats_ensamble](/home/leonardo/Documents/crayon/tuning/NBEATSEnsembleEstimator/grid_search_summary_March-26-2021--09-23-01.json)
* 2.68 [tune_gaussian_process](/home/leonardo/Documents/crayon/tuning/gaussianProcessEstimator/grid_search_summary_March-29-2021--00-16-57.json)
* 7.16 [MQRNN](/home/leonardo/Documents/crayon/tuning/MQRNNEstimator/grid_search_summary_March-30-2021--10-59-10.json) # loads of NaN values
* 0.73 [tune_transformer](/home/leonardo/Documents/crayon/tuning/TransformerEstimator/grid_search_summary_March-29-2021--23-58-29.json)


### FAILED GPU
* tune_deepstate # Memory leak on gpu and cpu image
* tune_deepvar   # Issues with dimensionality as electricity is univariate
* tune_npts      # the shell is trying to pass validation data to the estimator which raises a TypeError
* tune_mqcnn     # GPU issues: gluonts.core.exception.GluonTSUserError: Got NaN in first epoch. Try reducing initial learning rate.
* tune_wavenet   # Failed due to some fork exception

### TODO GPU
* tune_nbeats   # is it worth it? This is like training part of an algorithm only...
* tune_gpvar    # multivariate
* tune_lstnet   # multivariate 


### DONE CPU
* 1.20 [MQCNN](/home/leonardo/Documents/crayon/tuning/MQCNNEstimator/grid_search_summary_March-30-2021--14-43-02.json)
* 12.02 [deepfactor](/home/leonardo/Documents/crayon/tuning/deepFactorEstimator/grid_search_summary_March-31-2021--01-44-45.json)
* 0.72 [DeepAR](/home/leonardo/Documents/crayon/tuning/deepar/grid_search_summary_March-31-2021--09-30-18.json)
* 2.27 [canonicalRNN](/home/leonardo/Documents/crayon/tuning/canonicalRNN/grid_search_summary_March-31-2021--19-58-56.json)
* 0.92 [simplefeedforward](/home/leonardo/Documents/crayon/tuning/simplefeedforward/grid_search_summary_March-31-2021--22-15-16.json)

### TODO CPU

* tune_wavenet
* tune_mqrnn
* tune_gaussian_process
* tune_nbeats


### TO WRITE!

* Prophet # Does not need tuning as it is a predictor, Also needs a dockerfile with deps installed


## Traffic dataset

### DONE GPU
* [tune_simplefeedforward](/home/leonardo/Documents/crayon/tuning/simplefeedforward/grid_search_summary_March-21-2021--18-10-54.json)
* [tune_deepar](/home/leonardo/Documents/crayon/tuning/deepar/grid_search_summary_March-21-2021--09-31-53.json)
* [NBEATS Ensemble](/home/leonardo/Documents/crayon/tuning/NBEATSEnsembleEstimator/grid_search_summary_March-22-2021--13-20-41.json)