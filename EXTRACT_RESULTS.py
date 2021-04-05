import numpy as np

def main():
    models = ["bert", "mnli_base", "mnli_adv"]
    examples = [1,2,4,8,10,100]
    for model in models:
        for seen_examples in examples:
            results = []
            for run in [1,2,3,4,5]:
                with open(f'experiments_v2/0_bert_baseline/yelp/finetuned/{model}-trained_on_{seen_examples}-run_{run}.log', encoding='utf8') as f:
                    for line in f:
                        if line.__contains__("accuracy"):
                            l = line.split()
                            results.append(float(l[1]) * 100)
            results_array = np.array(results)
            print(f"Model: {model} --- Seen Examples: {seen_examples}")
            print(f"Check: {len(results_array)}")
            print(np.average(results_array))
            print(np.std(results_array))

if __name__ == "__main__":
    main()