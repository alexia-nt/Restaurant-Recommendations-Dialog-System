# Restaurant-Recommendations-Dialog-System
This is a restaurant recommendations dialog system.

## Installation Requirements

To run this project, you need to have **Python 3.x** installed on your machine.

The following packages are required:
- **pandas**: For data manipulation and analysis.
- **sklearn**: For machine learning models.
- **matplotlib**: For plotting graphs and visualizations.
- **Levenshtein**: For calculating Levenshtein distance.

## Data

The dataset (dialog_acts.dat) has the following format: **dialog_act [space] utterance_content** and is used to train models for dialog act classification.

There are 15 dialog acts in the dataset, listed in the following table:
| Dialog Act | Description                                      | Example Sentence                                |
|------------|--------------------------------------------------|-------------------------------------------------|
| ack        | Acknowledgment                                   | okay um                                         |
| affirm     | Positive confirmation                            | yes right                                       |
| bye        | Greeting at the end of the dialog                | see you good bye                                |
| confirm    | Check if given information confirms to query     | is it in the center of town                     |
| deny       | Reject system suggestion                         | I don't want Vietnamese food                    |
| hello      | Greeting at the start of the dialog              | hi I want a restaurant                          |
| inform     | State a preference or other information          | I'm looking for a restaurant that serves seafood|
| negate     | Negation                                         | no in any area                                  |
| null       | Noise or utterance without content               | cough                                           |
| repeat     | Ask for repetition                               | can you repeat that                             |
| reqalts    | Request alternative suggestions                  | how about Korean food                           |
| reqmore    | Request more suggestions                         | more                                            |
| request    | Ask for information                              | what is the post code                           |
| restart    | Attempt to restart the dialog                    | okay start over                                 |
| thankyou   | Express thanks                                   | thank you good bye                              |

## Part1A.py
The program provides the following options in the terminal:
```
1. Print evaluation metrics
2. Give an utterance for prediction
3. Exit
```

The following four models are trained for dialog act classification:
- **Majority-Label Model**: This model predicts the most common label in the training data for all inputs.
- **Rule-Based Model**: This model uses predefined rules to classify utterances based on keywords.
- **Decision Tree Model**: This is a decision tree machine learning model implemented using sklearn.
- **Logistic Regression Model**: This is a logistic regression machine learning model implemented using sklearn.

If the user chooses the first option, the program print the evaluation metrics for these models.

If the user chooses the second option, the user can enter an utterance and the system provides two dialog act classifications of this utterance based on the decision tree and the logistic regression model respectively.

For example:
```
Enter an utterance (or type '0' to go to menu): i want to find a restaurant
Predicted Label (Decision Tree): inform
Predicted Label (Logistic Regression): inform
```

## Part1B.py
The state diagram for this part of the project is the following:

![part1B](https://github.com/user-attachments/assets/19366b43-edde-41a2-b66f-1ddcee64dc6d)

The system asks the user which model to use for dialog act classification (Linear Regression Model / Decision Tree Model / Rule Based Model).

Then the conversation starts.

## Part1C.py
The state diagram for this part of the project is the following:

![part1C](https://github.com/user-attachments/assets/7627695e-a3f9-48b8-9a52-0097bf8c6d6f)

We implemented the following configurabilities building upon part1B.py:
1. Use one of the baselines for dialog act recognition instead of the machine learning classifier.
2. Levenshtein edit distance for preference extraction.
3. Introduce a delay before showing system responses.
4. Allow users to change their preferences or not.
5. Allow preferences to be stated in a single utterance only.

The system asks the user which model to use for dialog act classification (Linear Regression Model / Decision Tree Model / Rule Based Model).

Then the system asks for the Levenshtein distance threshold (valid range [0,3]).

Then the system asks if the user want a small delay in the system responses so that the conversation feels more natural.

After having extracted the user preferences, the conversation starts.


