import pandas as pd
import Levenshtein
import pickle

DT_FILE = "models/dt_model.pkl"
LR_FILE = "models/lr_model.pkl"
VECTORIZER_FILE = "models/vectorizer.pkl"

WELCOME_STATE = 1
ASK_INITIAL_PREFERENCES_STATE = 2
ASK_FOOD_STATE = 3
ASK_PRICE_STATE = 4
ASK_AREA_STATE = 5
RECOMMEND_STATE = 6
END_STATE = 7

DONT_CARE_KEYWORDS = ["do not care", "don't care", "dont care",
                      "do not mind", "don't mind", "dont mind",
                      "does not matter", "doesn't matter", "doesnt matter",
                      "any", "anywhere", "whatever"]

def get_keywords(df):
    food_keywords = list(df["food"].dropna().astype(str).unique())
    price_keywords = list(df["pricerange"].dropna().astype(str).unique())
    area_keywords = list(df["area"].dropna().astype(str).unique())
    return food_keywords, price_keywords, area_keywords

def extract_initial_preferences(utterance, keywords, preference_type):

    # Rule to check for "any" + preference_type
    if f"any {preference_type}" in utterance:
        return "any"

    for keyword in keywords:
        if keyword in utterance:
            return keyword

    # Split the utterance into words for checking
    utterance_words = utterance.lower().split()

    # Check for close matches using Levenshtein distance
    for word in utterance_words:
        for keyword in keywords:
            distance = Levenshtein.distance(word.lower(), keyword.lower())
            if distance <= 2:  # Adjust threshold as needed
                return keyword

    return None

def extract_preferences(utterance,keywords):
    """Extracts a preference from the utterance.

    Args:
        utterance: The input utterance.

    Returns:
        preference: The extracted preference (or None if not found).
    """
    for keyword in DONT_CARE_KEYWORDS:
        if keyword in utterance:
            return "any"

    for keyword in keywords:
        if keyword in utterance:
            return keyword
    
    # Split the utterance into words for checking
    utterance_words = utterance.lower().split()

    # Check for close matches using Levenshtein distance
    for word in utterance_words:
        for keyword in keywords:
            distance = Levenshtein.distance(word.lower(), keyword.lower())
            if distance <= 2:  # Adjust threshold as needed
                return keyword

    return None

def get_matching_restaurants(food_type, area, price_range):
    """Returns a list of restaurants based on the given preferences.

    Args:
        df: A Pandas DataFrame containing restaurant information.
        food_type: The desired food type.
        area: The desired area.
        price_range: The desired price range.

    Returns:
        restaurants: A list of dictionaries representing the recommended restaurant, or None if no suitable restaurant is found.
    """

    # Make a copy of the DataFrame to avoid modifying the original
    filtered_df = pd.read_csv("data/restaurant_info.csv")
    
    # Filter restaurants based on preferences
    if food_type != "any":
        filtered_df = filtered_df[filtered_df['food'] == food_type]
    
    if area != "any":
        filtered_df = filtered_df[filtered_df['area'] == area]
    
    if price_range != "any":
        filtered_df = filtered_df[filtered_df['pricerange'] == price_range]

    if not filtered_df.empty:
        restaurants = filtered_df.to_dict(orient='records')
        return restaurants
    else:
        return []
    
# ------------------------------------------ STATE HANDLERS ------------------------------------------

def welcome_state_handler():
    print("Hello, welcome to the restaurant recommdendations dialogue system! You can ask for restaurants by area, price range or food type.")
    user_input = input(">>").lower()
    # check if user input is inform
    return user_input

def ask_initial_preferences_handler(user_input, food_keywords, price_keywords, area_keywords):
    food_preference = extract_initial_preferences(user_input, food_keywords, "food")
    price_preference = extract_initial_preferences(user_input, price_keywords, "price")
    area_preference = extract_initial_preferences(user_input, area_keywords, "area")

    print("Extracted food preference: ", food_preference)
    print("Extracted price preference: ", price_preference)
    print("Extracted area preference: ", area_preference)

    if food_preference == None:
        return ASK_FOOD_STATE, user_input, food_preference, price_preference, area_preference
    
    if price_preference == None:
        return ASK_PRICE_STATE, user_input, food_preference, price_preference, area_preference
    
    if area_preference == None:
        return ASK_AREA_STATE, user_input, food_preference, price_preference, area_preference
    
    return RECOMMEND_STATE, user_input, food_preference, price_preference, area_preference

def ask_food_handler(user_input, food_keywords, food_preference):
    # keep the food preference from initial preferences
    if food_preference != None:
        return user_input, food_preference
    
    while food_preference == None:
        print("What kind of food would you like?")
        user_input = input(">>").lower()
        food_preference = extract_preferences(user_input, food_keywords)

    print("Extracted food preference: ", food_preference)

    return user_input, food_preference

def ask_price_handler(user_input, price_keywords, price_preference):
    # keep the price preference from initial preferences
    if price_preference != None:
        return user_input, price_preference
    
    while price_preference == None:
        print("What price range do you want?")
        user_input = input(">>").lower()
        price_preference = extract_preferences(user_input, price_keywords)

    print("Extracted price preference: ", price_preference)

    return user_input, price_preference

def ask_area_handler(user_input, area_keywords, area_preference):
    # keep the area preference from initial preferences
    if area_preference != None:
        return user_input, area_preference
    
    while area_preference == None:
        print("What part of town do you have in mind?")
        user_input = input(">>").lower()
        area_preference = extract_preferences(user_input, area_keywords)

    print("Extracted area preference: ", area_preference)
    
    return user_input, area_preference

def recommend_handler(food_preference, price_preference, area_preference):
    matching_restaurants = get_matching_restaurants(food_preference, price_preference, area_preference)
    print("Found ", len(matching_restaurants), " restaurants based on your preferences.")
    for restaurant in matching_restaurants:
        print(restaurant)
    return

def handle_transition(state, user_input, food_keywords, price_keywords, area_keywords, food_preference, price_preference, area_preference):
    if state == WELCOME_STATE:
        user_input= welcome_state_handler()
        return ASK_INITIAL_PREFERENCES_STATE, user_input, food_preference, price_preference, area_preference
    
    elif state == ASK_INITIAL_PREFERENCES_STATE:
        state, user_input, food_preference, price_preference, area_preference = ask_initial_preferences_handler(user_input, food_keywords, price_keywords, area_keywords)
        return state, user_input, food_preference, price_preference, area_preference
    
    elif state == ASK_FOOD_STATE:
        user_input, food_preference = ask_food_handler(user_input, food_keywords, food_preference)
        return ASK_PRICE_STATE, user_input, food_preference, price_preference, area_preference
    
    elif state == ASK_PRICE_STATE:
        user_input, price_preference  = ask_price_handler(user_input, price_keywords, price_preference)
        return ASK_AREA_STATE, user_input, food_preference, price_preference, area_preference
    
    elif state == ASK_AREA_STATE:
        user_input, area_preference  = ask_area_handler(user_input, area_keywords, area_preference)
        return RECOMMEND_STATE, user_input, food_preference, price_preference, area_preference
    
    elif state == RECOMMEND_STATE:
        recommend_handler(food_preference, price_preference, area_preference)
        return END_STATE, user_input, food_preference, price_preference, area_preference
    
    else:
        return END_STATE, user_input, food_preference, price_preference, area_preference

def main():
    # Load the restaurant data
    df = pd.read_csv("data/restaurant_info.csv")

    state = WELCOME_STATE
    food_keywords, price_keywords, area_keywords = get_keywords(df)

    print(food_keywords)
    print(price_keywords)
    print(area_keywords)

    user_input = ""
    food_preference = None
    price_preference = None
    area_preference = None

    while state != END_STATE:
        # Handle state transitions
        state, user_input, food_preference, price_preference, area_preference = handle_transition(state, user_input,
                                                                                                  food_keywords, price_keywords, area_keywords,
                                                                                                  food_preference, price_preference, area_preference)
        print("State: ", state)
        print("User input: ", user_input)
        print("food_type: ", food_preference)
        print("price_range: ", price_preference)
        print("area: ", area_preference)

main()