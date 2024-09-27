# Restaurant-Recommendations-Dialog-System
This is a restaurant recommendations dialog system.

## Installation Requirements

To run this project, you need to have **Python 3.x** installed on your machine.

The following packages are required:
- **pandas**: For data manipulation and analysis.
- **sklearn**: For machine learning models.
- **matplotlib**: For plotting graphs and visualizations.
- **Levenshtein**: For calculating Levenshtein distance.

# Part1A.py

The system provides the following options in the terminal:
1. Print evaluation metrics
2. Give an utterance for prediction
3. Exit

# Part1B.py

The system asks the user which model to use for dialog act classification (Linear Regression Model / Decision Tree Model / Rule Based Model).

Then the conversation starts.

# Part1C.py
We implemented the following configurabilities building upon part1B.py:
1. Use one of the baselines for dialog act recognition instead of the machine learning classifier.
2. Levenshtein edit distance for preference extraction.
3. Introduce a delay before showing system responses.
4. Allow users to change their preferences or not.
5. Allow preferences to be stated in a single utterance only.

The system asks the user which model to use for dialog act classification (Linear Regression Model / Decision Tree Model / Rule Based Model).

Then the system asks for the Levenshtein distance threshold (valid range [0,3]).

Then the system asks if the user want a small delay in the system responses so that the conversation feels more natural.


