# print("hello")

# DATA_FILE = 'data/dialog_acts.dat'

def return_most_freq_label():
    label_list = []

    # read data file
    with open('data/dialog_acts.dat','r') as file:
        # for each line of the data file
        for line in file:
            # get the first word of each line which is the label (dialog act) and append it in the list
            label_list.append(line.split()[0])

    # make a dictionary with unique labels as keys and their frequency in the label list as values
    label_counter = {}
    for label in label_list:
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1

    # get the most frequent label
    most_frequent_label = max(label_counter, key=label_counter.get)

    print("Percentage of most most frequent label:", label_counter[most_frequent_label]/len(label_list))

    # return the most frequent label
    return most_frequent_label

def baseline_model_most_freq_label_predict(utterance):
    y_pred = return_most_freq_label()
    return y_pred

res = baseline_model_most_freq_label_predict("hello")
print(res)