import os
import numpy as np

path= "data"

def load_dataset():
    all_data = {"training": None, "testing": None}
    labels = {"training": [], "testing": []}
    for root,dirs,files in os.walk(path,topdown=False):
        for file in files:
            if ".txt" in file:
                p = f"{root}/{file}"
                num = int(file[1:].replace(".txt",""))
                if file == "e10.txt":

                    arr = make_bag_of_words(p,scramble=True)
                else:
                    arr = make_bag_of_words(p,scramble=False)
                
                if num < 10:
                    training = True
                    if type(all_data["training"]) == type(None):
                        all_data["training"] = arr
                    else:
                        all_data["training"] = np.vstack( (all_data["training"],arr) )
                else:
                    training = False
                    if type(all_data["testing"]) == type(None):
                        all_data["testing"] = arr
                    else:
                        all_data["testing"] = np.vstack( (all_data["testing"],arr) )
                
                if file == "e10.txt":
                    print(f"Bag of words for e10: {arr}")

                if training:
                    if "e" == file[0]:
                        labels["training"].append(0)
                    elif "j" == file[0]:
                        labels["training"].append(1)
                    else:
                        labels["training"].append(2)
                else:
                    if "e" == file[0]:
                        labels["testing"].append(0)
                    elif "j" == file[0]:
                        labels["testing"].append(1)
                    else:
                        labels["testing"].append(2)
    return all_data,labels
                


def make_bag_of_words(path,scramble=False):
    vector = [0 for i in range(27)]
    with open(path, "r+") as f:
        data = f.read()
        if scramble:
            import random
            data = list(data)
            random.shuffle(data)
            data = "".join(data)

        for c in data:
            ind = ord(c) - ord("a")
            if ind < 0:
                ind = -1
            vector[ind]+=1

    return np.array(vector)

if __name__ == "__main__":
    data,labels = load_dataset()
    print(data.shape)