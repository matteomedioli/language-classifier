
import os
for lr in [5, 10, 50]:
    for batch_size in [32, 64, 128, 256]:
        for input_dim in [4, 8, 16]:
            for embed_dim in [16, 32, 64, 128, 256]:
                request = {
                    "epochs": 15, 
                    "lr": lr, 
                    "step_size": 1.0, 
                    "gamma": 0.1, 
                    "batch_size": batch_size, 
                    "input_dim": input_dim, 
                    "embed_dim": embed_dim, 
                    "num_classes": 17, 
                    "eval_every": 100,
                    "model_path": f"B_{batch_size}_ED_{embed_dim}_IN_{input_dim}_LR_{lr}"
                }
                request = str(request)
                request = request.replace('\'', '\\"')
                cmd = f'curl -X POST http://127.0.0.1:5000/train -H "Content-Type: application/json" -d "{request}"'
                os.system(cmd)
