## try to reproduce yake.
import yake
import pandas as pd
import tqdm
import glob
import os
dataset_list = glob.glob("../datasets/*")

os.system("rm -rvf ./results_for_datasets_yake")
os.system("rm -rvf ./eval_results/*")

num_keywords = 20


def read_contents(flx):
    return open(flx).read()


dnames = []
for dataset in dataset_list:
    dname = dataset.split("/")[-1]
    dnames.append(dname)

    directory = "results_for_datasets/{}".format(dname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fx = dataset + "/docsutf8/*"
    for kwf in tqdm.tqdm(glob.glob(fx)):
        text = read_contents(kwf)
        kw_extractor = yake.KeywordExtractor()
        keywords = kw_extractor.extract_keywords(text)[0:num_keywords]
        kwid = kwf.split("/")[-1].replace(".txt", "")
        kwx = directory + "/{}".format(kwid + ".key")
        btx = "\n".join(list(x[1] for x in keywords))
        with open(kwx, "w") as wx:
            wx.write(btx)

for name in dnames:
    cm1 = "python3 keyphrase2trec.py --datasetdir ../datasets/{} --results results_for_datasets/".format(
        name, name)
    os.system(cm1)
    command = "./trec_eval -m set_F -M1000  output/{}.qrel output/{}_results_for_datasets.out > eval_results/extracts_{}.scores".format(
        name, name, name)
    os.system(command)

### some processing of the data.
results = []
for j in glob.glob("eval_results/*"):
    try:
        dfx_rows = []
        with open(j) as fx:
            for line in fx:
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        if parts[2] == ".":
                            continue
                        dfx_rows.append({
                            "score": parts[0],
                            "docID": parts[1],
                            "performance": float(parts[2])
                        })
                    except Exception as es:
                        print(es)

        dfx2 = pd.DataFrame(dfx_rows)
        dfx2['performance'] = dfx2['performance'].astype(float)
        dfx = dfx2.groupby(["score"])['performance'].mean()
        results.append({"dataset": j, "score": dfx['set_F']})

    except Exception as es:
        print(es)
pd.DataFrame(results).to_csv(
    "final_eval/final_{}_F_score.tsv".format(num_keywords), sep="\t")
