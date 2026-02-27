import os, json

dataset = []
for root, _, files in os.walk("C:/Sushant/testWorkspace/microservices/testProj"):
    for f in files:
        if f.endswith(".java"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as file:
                code = file.read()
                dataset.append({"input": code, "output": "Explain this code"})

with open("C:/Sushant/AI Learning/Datasets/java_dataset.jsonl", "w", encoding="utf-8") as out:
    for entry in dataset:
        out.write(json.dumps(entry) + "\n")
