import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

DATA_FILE = 'data/dialog_acts.dat'

def create_dataframe(data_file):
    """Creates a Pandas DataFrame from the given data file.

    Args:
        data_file: The path of the data file.

    Returns:
        df: A DataFrame with two columns: 'label' and 'utterance'.
    """

    labels = []
    utterances = []

    with open(data_file, 'r') as file:
        for line in file:
            label, utterance = line.split(maxsplit=1)
            labels.append(label)
            utterances.append(utterance)

    data_dict = {'label': labels, 'utterance': utterances}
    df = pd.DataFrame(data_dict)

    return df

def return_majority_label(data_file):
    """Returns the most frequent label in the data file.

    Args:
        data_file: The path of the data file.

    Returns:
        majority_label: The most frequent label in the data file.
    """

    label_list = []
    with open(data_file, 'r') as file:
        for line in file:
            label_list.append(line.split()[0])  # Get the first word of each line as the label

    # make a dictionary with unique labels as keys and their frequency in the label list as values
    label_counter = {}
    for label in label_list:
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1

    # get the most frequent label
    majority_label = max(label_counter, key=label_counter.get)

    print("Percentage of most most frequent label:", label_counter[majority_label]/len(label_list))

    # return the most frequent label
    return majority_label

def baseline_model_majority_label_predict(X_test):
    y_pred = []
    majority_label = return_majority_label(DATA_FILE)
    for utterance in X_test:
        y_pred.append(majority_label)
    return y_pred

def print_metrics(y_test, y_pred):

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate and print the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

def main():
    # Load the data into a DataFrame
    df = create_dataframe(DATA_FILE)

    # Split the data into features (utterances) and labels
    X = df['utterance']
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions on the test set
    y_pred = baseline_model_majority_label_predict(X_test)

    print_metrics(y_test, y_pred)

# Run the main function
main()
