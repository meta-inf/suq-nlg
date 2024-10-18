Code for the paper "On Subjective Uncertainty Quantification and Calibration in Natural Language Generation". 

## Using the Code

### Environment Setup

```sh
conda create -n suq python=3.10 pip
conda activate suq
pip install -r requirements.txt
```

To reproduce any LM generation experiment, you will need to set up API access:

* For the OpenAI models, follow their quickstart guide [here](https://platform.openai.com/docs/quickstart).
* For any other model, first set up Google cloud authentication following [this guide](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal#expandable-1). For the non-Gemini models you also need to set up permissions following the guide in [this section](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-partner-models).

Alternatively, if you only want to reproduce the analysis, you can download the preprocessed
data and model samples [here](https://drive.google.com/drive/folders/1uxLEnj6bup5m0CVN_vx3aswMUlhs4V6B?usp=drive_link), 
and extract them using
```sh
tar xf data.tar.xz && tar xf run.tar.xz
```

### QA Experiments

For preprocessing:
```sh
# export the datasets from Lin et al (2024). Note this command should be executed in a Python
# environment created following the instructions in https://github.com/zlin7/UQ-NLG
python scripts/preprocessing/qa-lin-datasets.py
# export SciQ and TruthfulQA
python scripts/preprocessing/qa-additional-datasets.py
# generate mask for outdated questions in the trivia datasets
for d in nq_open trivia; do
    python scripts/preprocessing/qa-mask-trivia.py -d $d -o ./data/mask-$d.pkl
done
```

To generate responses from any LM, check the instructions in `scripts/qa-sample.sh` and modify 
it accordingly, before running it in the current (base) directory. 
The samples will be saved to a directory with the format `run/qa/<dataset>-*-<model>`.

To evaluate the utility and uncertainty measures, set up OpenAI API access and run
```sh
# submit batch jobs
bash scripts/qa-submit-eval.sh <your_experiment_dirs>
# once the jobs complete:
bash scripts/qa-retrieve-eval.sh <your_experiment_dirs>
```
In our experiments with the Llama-3.1-70B model, generation cost ~28 M tokens and evaluation cost ~25 M tokens.

The following script will reproduce Fig. 3 and Fig. 6-8 in paper, which also include full 
information from Fig. 1-2. Modfiy it if you don't have all experiments completed.
```sh
python qa/plot-all.py
```

### MT Experiments

For preprocessing, set up Google cloud authentication and run 
```sh
mkdir data/flores
cd data/flores
wget https://github.com/openlanguagedata/flores/releases/download/v2.0-rc.3/floresp-v2.0-rc.3.zip
unzip floresp-v2.0-rc.3.zip  # obtain the password from the repository homepage above
# generate embeddings for the many-shot ICL predictor
python scripts/preprocessing/mt-prep-emb.py
# generate completion inputs x_{n+1:N}
PREFIX=./run/mt
python -m mt.main --mode preprocess -gt 128 -comp 16 -N -1 --dest_lang tam_Taml  -dir $PREFIX/preproc-tam-full --retrieval_method mixed-4
python -m mt.main --mode preprocess-replace -gt 128 -comp 16 -N -1  -lp $PREFIX/preproc-tam-full/ --dest_lang kmr_Latn -dir $PREFIX/preproc-kmr-full --retrieval_method mixed-4
```

To draw samples, run
```sh
bash scripts/mt.sh
```
Experiments should cost 15-20M tokens for each (LM, language) pair.

For analysis, run 
```sh
python mt/plot-all.py
```
This reproduces all plots and tables in paper. Modify the code if you don't have all experiments
completed.

## Acknowledgement

This repository contains code adapted from the repository [zlin7/UQ-NLG](https://github.com/zlin7/UQ-NLG).

