"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    fc = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")
        if k.startswith("fc"):
            print(k)
            fc[k] = v
        else:
            newmodel[k] = v

    with open(sys.argv[2], "wb") as f:
        torch.save(newmodel, f)

    with open(sys.argv[3], "wb") as f:
        torch.save(fc, f)
