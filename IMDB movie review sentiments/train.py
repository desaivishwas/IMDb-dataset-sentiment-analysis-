'''
Author: Tanmay Sawaji
This file was used to plot various graphs by taking different values of train-test split and laplace smoothing parameter alpha and comparing them using 
accuracy, recall and precision. 
'''

from NaiveBayesClassifier import NaiveBayes
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

def read_data(split = False, train_size = 25000, test_size = 25000):
    filename = "IMDB Dataset.txt"
    separator = ","
    examples = []
    labels = []
    with open(filename, 'r', encoding='utf8') as f:
        for index,line in enumerate(f):
            if index == 0:
                continue
            parsed = line.strip().rsplit(separator, 1)
            labels.append(parsed[1] if len(parsed) > 0 else "")
            example = parsed[0].replace("<br />", "") if len(parsed) > 1 else ""
            examples.append(example)
    if split:
        test_index = len(examples) - test_size
        train_data = {"examples": examples[:train_size], "labels": labels[:train_size], "classes": list(set(labels))}
        test_data = {"examples": examples[test_index:], "labels": labels[test_index:], "classes": list(set(labels))}
        return train_data, test_data
    else:
        train_data = {"examples": examples, "labels": labels, "classes": list(set(labels))}
        return train_data, 0

def model_evaluation(predictions, actual):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for val1, val2 in zip(predictions, actual):
        if val1 == "positive" and val2 == "positive":
            true_positives += 1
        if val1 == "negative" and val2 == "negative":
            true_negatives += 1
        if val1 == "positive" and val2 == "negative":
            false_positives += 1
        if val1 == "negative" and val2 == "positive":
            false_negatives += 1
    accuracy = (true_positives + true_negatives) / len(actual)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return accuracy, precision, recall

def save_model(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def main_training():
    train_data, test_data = read_data(split=True)
    model = NaiveBayes()
    print("Beginning training...")
    model.classifier(train_data, "examples", "labels", "classes")
    print("Saving the model...")
    # save_model("test_accuracy.pkl", model)
    print("Model saved successfully !")

    print("Testing examples...")
    results = model.fit_naive_bayes(test_data, "examples")
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("-"*50)
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
    acc, precision, recall = model_evaluation(results, test_data["labels"])
    print(f"Accuracy : {acc}")
    print(f"Precision : {precision}\nRecall : {recall}")
    
def plot_alpha_values():
    train_plot = []
    accuracy_plot = []
    evaluation = {
        "accuracy" : [],
        "precision" : [],
        "recall" : []
    }
    for i in range(1,11):
        train_data, test_data = read_data(split=True, train_size=25000, test_size=10000)
        model = NaiveBayes()
        model.classifier(train_data, "examples", "labels", "classes", alpha=i)
        results = model.fit_naive_bayes(test_data, "examples")
        correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
        accuracy = 100.0 * correct_ct / len(test_data["labels"])
        train_plot.append(i)
        accuracy_plot.append(accuracy)
        acc, precision, recall = model_evaluation(results, test_data["labels"])
        evaluation["accuracy"].append(acc)
        evaluation["precision"].append(precision)
        evaluation["recall"].append(recall)
    print(f"Alpha value : {train_plot}\nAccuracy : {accuracy_plot}")
    print("Model evaluation is :\n", evaluation)
    df = pd.DataFrame({
        'Alpha value' : train_plot,
        'Accuracy (in %)' : accuracy_plot
    })
    sns.lineplot(data=df, x="Alpha value", y="Accuracy (in %)")
    plt.show()

def plot_training_size_vs_accuracy():
    train_plot = []
    accuracy_plot = []
    evaluation = {
        "accuracy" : [],
        "precision" : [],
        "recall" : []
    }
    for i in range(1000, 40000, 1000):
        train_data, test_data = read_data(split=True, train_size=i, test_size=10000)
        model = NaiveBayes()
        model.classifier(train_data, "examples", "labels", "classes")
        results = model.fit_naive_bayes(test_data, "examples")
        correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
        accuracy = 100.0 * correct_ct / len(test_data["labels"])
        train_plot.append(i)
        accuracy_plot.append(accuracy)
        acc, precision, recall = model_evaluation(results, test_data["labels"])
        evaluation["accuracy"].append(acc)
        evaluation["precision"].append(precision)
        evaluation["recall"].append(recall)

    # train_plot = list(range(10))
    # accuracy_plot = [35,45,91,66,38,65,90,76,42,67]
    print(f"Training size : {train_plot}\nAccuracy : {accuracy_plot}")
    print("Model evaluation is :\n", evaluation)
    df = pd.DataFrame({
        'Training size' : train_plot,
        'Accuracy (in %)' : accuracy_plot
    })
    sns.lineplot(data=df, x="Training size", y="Accuracy (in %)")
    plt.show()

def plot_precision():
    alpha_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    evaluation = {'accuracy': [0.854, 0.8556, 0.856, 0.8545, 0.8539, 0.8537, 0.8534, 0.8537, 0.8532, 0.8533], 'precision': [0.8717250052399916, 0.8738719832109129, 0.8746061751732829, 0.873109243697479, 0.8718540268456376, 0.8718002517834662, 0.8720319394830847, 0.8727387463188894, 0.8721345951629863, 0.8724747474747475], 'recall': [0.8306371080487318, 0.8316357100059916, 0.8316357100059916, 0.8300379468743758, 0.8302376672658278, 0.8298382264829239, 0.8288396245256641, 0.8286399041342121, 0.8282404633513082, 0.8280407429598562]}
    # evaluation = {'accuracy': [0.7935, 0.8242, 0.8281, 0.8312, 0.8352, 0.8352, 0.8408, 0.8434, 0.8459, 0.8482, 0.8514, 0.8495, 0.8492, 0.8497, 0.8518, 0.8524, 0.8544, 0.8544, 0.8536, 0.8547, 0.8539, 0.8531, 0.8533, 0.8544, 0.854, 0.8542, 0.8555, 0.8557, 0.8567, 0.8568, 0.8584, 0.8582, 0.8572, 0.8563, 0.8565, 0.857, 0.857, 0.8579, 0.8577], 'precision': [0.8258750553832521, 0.8413532254675352, 0.8559982676483326, 0.8648054517476368, 0.8661434488772618, 0.86235167206041, 0.8621420996818664, 0.8628981227589116, 0.865149599662874, 0.8690501375079331, 0.8718057022175291, 0.8706604572396275, 0.8705782673162465, 0.8711864406779661, 0.8734901462174189, 0.8723897911832946, 0.8738681827753211, 0.8738681827753211, 0.8730259001895135, 0.8737904922170804, 0.8718540268456376, 0.8719512195121951, 0.872318047959613, 0.8718324607329843, 0.8717250052399916, 0.8702274149801794, 0.871506049228202, 0.8714047519799917, 0.8724468528553564, 0.8726287262872628, 0.8752351097178683, 0.8745564600292215, 0.8742940807362476, 0.8728070175438597, 0.8734838979506483, 0.8736160434510132, 0.8740849194729137, 0.8754187604690117, 0.8755238893545683], 'recall': [0.7445576193329339, 0.7996804473736768, 0.7894947074096266, 0.7857000199720392, 0.7934891152386658, 0.7982824046335131, 0.8118633912522468, 0.817056121429998, 0.8200519273017776, 0.8204513680846814, 0.8244457759137208, 0.8214499700419413, 0.8208508088675854, 0.8212502496504893, 0.823247453565009, 0.8260435390453366, 0.8288396245256641, 0.8288396245256641, 0.8280407429598562, 0.8296385060914719, 0.8302376672658278, 0.8282404633513082, 0.8282404633513082, 0.8314359896145397, 0.8306371080487318, 0.8330337527461554, 0.8344317954863192, 0.8350309566606751, 0.8360295586179349, 0.8360295586179349, 0.8364289994008388, 0.8368284401837428, 0.8348312362692231, 0.8346315158777711, 0.8342320750948672, 0.8352306770521271, 0.8346315158777711, 0.8350309566606751, 0.8344317954863192]}
    # train_size = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000]
    df = pd.DataFrame({
        'Alpha' : alpha_vals,
        'Precision' : evaluation['precision']
    })
    sns.lineplot(data=df, x="Alpha", y="Precision")
    plt.show()

if __name__ == "__main__":
    plot_precision()

    