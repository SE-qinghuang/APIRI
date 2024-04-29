import csv
import module_1
import module_2
import module_3
import pandas as pd
import os

def write2file(dict,output_path):
    df = pd.DataFrame(dict)
    df.to_csv(output_path, header=False, index=False)


def run_APIRI(data_path,outpath_1,outpath_2,outpath_3):
    outcome_1 = []
    outcome_2 = []
    outcome_3 = []
    with open(data_path, mode='r',encoding='utf-8') as file:
        csvFile = csv.reader(file)
        result = list(csvFile)
        for i in range(len(result)):
            try:
                print(i)
                task_1 = {"API_Text":None,"Non_FQNs":None,"FQNs":None,"API_list":None}
                task_2 = {"API_Text":None,"Pair_knowledge":None}

                non_fqns, fqns, API_pair_list = module_1.execute_module_1(result[i][0])
                task_1["API_Text"] = result[i][0]
                task_1["Non_FQNs"] = non_fqns
                task_1["FQNs"] = fqns
                task_1["API_list"] = API_pair_list
                outcome_1.append(task_1)
                write2file(outcome_1,outpath_1)

                if API_pair_list is not None:
                    pair_knowledge = module_2.execute_module_2(API_pair_list)
                    task_2["API_Text"] = result[i][0]
                    task_2["Pair_knowledge"] = pair_knowledge
                    outcome_2.append(task_2)
                    write2file(outcome_2, outpath_2)

                    m3_res = module_3.execute_module_3(pair_knowledge)
                    outcome_3.extend(m3_res)
                    write2file(outcome_3, outpath_3)
            except Exception as e:
                print(e)
                print(result[i][0])

if __name__ == "__main__":
    # input—data, module-1's output, module-2's output, module-3's output
    run_APIRI("","","","")