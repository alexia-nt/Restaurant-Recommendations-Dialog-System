# print("hello")

# DATA_FILE = 'data/dialog_acts.dat'

def return_most_freq_label():
    dialog_act_list = []

    # read data file
    with open('data/dialog_acts.dat','r') as f:
        # for each line of the data file
        for line in f:
            # get the first word of each line which is the label (dialog act) and append it in the list
            dialog_act_list.append(line.split(' ')[0])

    # make a dictionary with the label and the number of occurences
    label_counter = {}
    for label in dialog_act_list:
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1

    # sort the dictionary by the number of occurences
    frequent_labels = sorted(label_counter, key = label_counter.get, reverse = True)

    # get the most frequent (majority) label
    top_1 = frequent_labels[:1]

    # return the most frequent (majority) label 
    return(top_1[0])

def baseline_model_most_freq_label_predict(utterance):
    y_pred = return_most_freq_label()
    return y_pred

res = baseline_model_most_freq_label_predict("hello")
print(res)