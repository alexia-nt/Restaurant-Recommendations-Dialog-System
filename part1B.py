import pandas as pd
import Levenshtein

class RestaurantRecommendationSystem:
    DATA_FILE = "data/restaurant_info.csv"
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

    DONT_CARE_KEYWORDS = [
        "do not care", "don't care", "dont care",
        "do not mind", "don't mind", "dont mind",
        "does not matter", "doesn't matter", "doesnt matter",
        "any", "anywhere", "whatever"
    ]

    def __init__(self):
        self.df = pd.read_csv(self.DATA_FILE)
        self.food_keywords, self.price_keywords, self.area_keywords = self.get_keywords()
        self.state = self.WELCOME_STATE
        self.user_input = ""
        self.food_preference = None
        self.price_preference = None
        self.area_preference = None

    def get_keywords(self):
        food_keywords = list(self.df["food"].dropna().astype(str).unique())
        price_keywords = list(self.df["pricerange"].dropna().astype(str).unique())
        area_keywords = list(self.df["area"].dropna().astype(str).unique())
        return food_keywords, price_keywords, area_keywords

    def extract_initial_preferences(self, utterance, keywords, preference_type):
        if f"any {preference_type}" in utterance:
            return "any"
        
        for keyword in keywords:
            if keyword in utterance:
                return keyword
        
        utterance_words = utterance.lower().split()
        
        for word in utterance_words:
            for keyword in keywords:
                distance = Levenshtein.distance(word.lower(), keyword.lower())
                if distance <= 1:  # Adjust threshold as needed
                    return keyword
        return None

    def extract_preferences(self, utterance, keywords):
        for keyword in self.DONT_CARE_KEYWORDS:
            if keyword in utterance:
                return "any"
        
        for keyword in keywords:
            if keyword in utterance:
                return keyword
        
        utterance_words = utterance.lower().split()
        
        for word in utterance_words:
            for keyword in keywords:
                distance = Levenshtein.distance(word.lower(), keyword.lower())
                if distance <= 1:  # Adjust threshold as needed
                    return keyword
        return None

    def get_matching_restaurants(self):
        filtered_df = self.df.copy()

        if self.food_preference != "any":
            filtered_df = filtered_df[filtered_df["food"] == self.food_preference]
        
        if self.price_preference != "any":
            filtered_df = filtered_df[filtered_df["pricerange"] == self.price_preference]
        
        if self.area_preference != "any":
            filtered_df = filtered_df[filtered_df["area"] == self.area_preference]

        if not filtered_df.empty:
            restaurants = filtered_df.to_dict(orient='records')
            return restaurants
        else:
            return []

    # ------------------------ State Handlers ------------------------

    def welcome_state_handler(self):
        print("Hello, welcome to the restaurant recommendations dialogue system! You can ask for restaurants by area, price range, or food type.")
        self.user_input = input(">>").lower()
        return

    def ask_initial_preferences_handler(self):
        self.food_preference = self.extract_initial_preferences(self.user_input, self.food_keywords, "food")
        self.price_preference = self.extract_initial_preferences(self.user_input, self.price_keywords, "price")
        self.area_preference = self.extract_initial_preferences(self.user_input, self.area_keywords, "area")

        if self.food_preference is None:
            return self.ASK_FOOD_STATE
        
        if self.price_preference is None:
            return self.ASK_PRICE_STATE
        
        if self.area_preference is None:
            return self.ASK_AREA_STATE

        return self.RECOMMEND_STATE

    def ask_food_handler(self):
        if self.food_preference is not None:
            return

        while self.food_preference is None:
            print("What kind of food would you like?")
            self.user_input = input(">>").lower()
            self.food_preference = self.extract_preferences(self.user_input, self.food_keywords)

        return

    def ask_price_handler(self):
        if self.price_preference is not None:
            return

        while self.price_preference is None:
            print("What price range do you want?")
            self.user_input = input(">>").lower()
            self.price_preference = self.extract_preferences(self.user_input, self.price_keywords)

        return

    def ask_area_handler(self):
        if self.area_preference is not None:
            return

        while self.area_preference is None:
            print("What part of town do you have in mind?")
            self.user_input = input(">>").lower()
            self.area_preference = self.extract_preferences(self.user_input, self.area_keywords)

        return

    def recommend_handler(self):
        matching_restaurants = self.get_matching_restaurants()
        print(f"Found {len(matching_restaurants)} restaurants based on your preferences.")
        for restaurant in matching_restaurants:
            print(restaurant)

    # ------------------------ State Transition ------------------------

    def handle_transition(self):
        if self.state == self.WELCOME_STATE:
            self.welcome_state_handler()
            self.state = self.ASK_INITIAL_PREFERENCES_STATE
            return

        elif self.state == self.ASK_INITIAL_PREFERENCES_STATE:
            self.state = self.ask_initial_preferences_handler()
            return

        elif self.state == self.ASK_FOOD_STATE:
            self.ask_food_handler()
            self.state = self.ASK_PRICE_STATE
            return

        elif self.state == self.ASK_PRICE_STATE:
            self.ask_price_handler()
            self.state = self.ASK_AREA_STATE
            return

        elif self.state == self.ASK_AREA_STATE:
            self.ask_area_handler()
            self.state = self.RECOMMEND_STATE
            return

        elif self.state == self.RECOMMEND_STATE:
            self.recommend_handler()
            self.state = self.END_STATE
            return

    def run(self):
        while self.state != self.END_STATE:
            self.handle_transition()

            print("State: ", self.state)
            print("User input: ", self.user_input)
            print("food_type: ", self.food_preference)
            print("price_range: ", self.price_preference)
            print("area: ", self.area_preference)

# Create an instance and run the system
if __name__ == "__main__":
    system = RestaurantRecommendationSystem()
    system.run()
