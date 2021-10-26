#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.

###
mkdir -p results/Alzheimerheimer
###
python -u BuildKG.py --db tcrd \
	--o results/ProteinDisease_GRAPH.pkl
###
python -u GenStaticFeatures.py --db tcrd \
	--outputdir results
###
python -u PrepTrainingAndTestSets.py --i data/Alzheimers.xlsx \
	--symbol_or_pid "symbol" \
	--use_default_negatives \
	--db tcrd
###
 python -u GenTrainingAndTestFeatures.py \
	--trainingfile data/Alzheimers.pkl \
	--predictfile data/Alzheimers_predict.pkl \
	--outputdir results/Alzheimer \
	--kgfile results/ProteinDisease_GRAPH.pkl \
	--static_data "gtex,lincs,ccle,hpa" \
	--static_dir results \
	--db tcrd
###
python -u TrainModelML.py XGBCrossValPred \
	--trainingfile results/Alzheimer/Alzheimers_TrainingData.pkl \
	--resultdir results/Alzheimer \
	--xgboost_param_file XGBparams.txt \
	--db tcrd \
	--static_data "gtex,lincs,ccle,hpa" \
	--static_dir results
###
python3 -u PredictML.py XGBPredict \
	--predictfile results/Alzheimer/Alzheimers_predict_PredictData.pkl \
	--model results/Alzheimer/XGBCrossValPred.model \
	--resultdir results/Alzheimer \
	--db tcrd \
	--infofile data/plotDT.xlsx
###