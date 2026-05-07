This directory provides the scripts to run CMTs and benchmark methods on the Swissmetro dataset.

## Dependencies

All scripts are now expected to run under Python 3. Install the shared dependencies from the repository root with:

`python3 -m pip install -r requirements.txt`

The SwissMetro scripts use NumPy, pandas, joblib, scikit-learn, seaborn, and TensorFlow. The tree leaf-model code now uses a TF2-compatible training loop instead of TensorFlow Estimator. On Apple Silicon Macs, `tensorflow-metal` is installed automatically through the platform marker in `requirements.txt`.

For the TensorFlow MNL leaf model, training is controlled by `steps`. The `epochs` keyword remains accepted for compatibility with existing scripts but does not change the optimization count.

## Description of scripts

`cmt.py` and `cmt+.py`: estimation of the CMT on the 10 data splits for a maximum depth of 14.

`mnlkm.py` and `mnlkm+.py`: estimation of the MNLKM benchmark on the 10 data splits with a search over the number of clusters 5, 10, ... 295. 

`mnldt.py` and `mnldt+.py`: estimation of the MNLDT benchmark on the 10 data splits for a maximum depth of 14. 

`mnlicot.py` and `mnlicot+.py`: estimation of the MNLICOT benchmark on the 10 data splits. The ICOT tree is hardcoded. Code that generates the ICOT tree is provided in the folder `src/ICOT`. The same tree is produced across 5 distinct splits of the data.

`tastenet.py`: estimation of the TasteNet benchmark on the 10 data splits. Code is implemented using the tensorflow library. Input data is used in long format. Note: random seed was not saved.

`mnlint.py`: estimation of the MNLINT benchmark on the 10 data splits. Code is implemented using the tensorflow library. Input data is in used in long format. Note: random seed was not saved.

`plots.py`: visualisation of the Pareto curve showing the predictive performance as a function of the number of segments for MNLKM and CMT.

`prepare_data.py`: code to generate the 10 random splits 75%-12.5%-12.5%, both in long/wide formats. Note: random seed was not saved.




