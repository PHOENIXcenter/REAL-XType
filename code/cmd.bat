%echo off

echo Benchmarking
for %%i in (LogisticRegression XGBoost RandomForest DNN XType) do (
	for /l %%j in (0, 1, 0) do (
		for /l %%k in (0, 1, 5) do (
			for /l %%l in (0, 1, 4) do (
				echo method=%%i para_id=%%j fold=%%k seed=%%l
				python code\benchmark.py --method %%i --para_id %%j --valid_fold %%k  --seed %%l
			)
		)
	)
)
python code\benchmark.py --evaluate True
python code\plot.py

echo Application
for /l %%i in (0, 1, 0) do (
	for /l %%j in (0, 1, 5) do (
		echo para_id=%%i 
		python code\training.py --para_id %%i --valid_fold %%j
	)
)
python code\evaluation.py


