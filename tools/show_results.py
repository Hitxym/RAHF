import argparse
import os
import json

def print_result(result_path, flag=False, filter=""):
    if filter != "":
        json_files = [os.path.join(result_path, i) for i in os.listdir(result_path) if (i.endswith(".json") and filter in i)]
    else:
        json_files = [os.path.join(result_path, i) for i in os.listdir(result_path) if i.endswith(".json")]
    for json_file in json_files:
        with open(json_file) as f:
            if flag: # for other methods
                print(f"{json_file}  ROC_AUC:  {json.load(f)['metrics']['roc_auc']:.4f}")
            else:
                print(f"{json_file}  ROC_AUC:  {json.load(f)['val_ROC_AUC']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./results/main/polish/Qwen2-7B")
    parser.add_argument('--other', action="store_true")
    parser.add_argument('--filter', type=str, default="")
    args = parser.parse_args()
    print_result(args.result_path, args.other, args.filter)