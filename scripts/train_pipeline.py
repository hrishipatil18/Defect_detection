import subprocess

print("Starting Data Preprocessing...")
subprocess.run(["python", "scripts/preprocess.py"])

print("Training Models...")
subprocess.run(["python", "scripts/train.py"])

print("Evaluating Models...")
subprocess.run(["python", "scripts/evaluate.py"])

print("Logging to MLflow...")
subprocess.run(["python", "scripts/monitor.py"])
