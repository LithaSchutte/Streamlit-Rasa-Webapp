from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
from fuzzywuzzy import process

DATA_PATH = "../data/global_health.csv"

try:
    data = pd.read_csv(DATA_PATH)
    print(data.head())
except FileNotFoundError:
    print(f"File not found!")

class ActionGetCountryData(Action):
    def name(self) -> str:
        return "action_country_specific"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        geo_country = tracker.get_slot("GPE")

        if not geo_country:
            dispatcher.utter_message(
                text="I didn't understand the country name. Can you please specify the country that you meant?")
            return []

        if len(geo_country) == 1:
            dispatcher.utter_message(text=f"The selected country is {geo_country}")
        else:
            country_list = ", ".join(geo_country[:-1]) + f", and {geo_country[-1]}"
            dispatcher.utter_message(text=f"The selected countries are: {country_list}.")

        return [SlotSet("GPE", geo_country)]


class ActionGetAverage(Action):
    def name(self) -> str:
        return "action_calculate_column_stat"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        # Get the column name from the slot
        column_name = tracker.get_slot("column_name")
        value_action = tracker.get_slot("value_action")

        if not value_action:
            dispatcher.utter_message(
                text="I need to know whether you want the 'min', 'max', or 'mean' value. Can you specify?")
            return []

        columns = data.columns.tolist()
        match, score = process.extractOne(column_name, columns)

        if value_action == "mean":
            result = data[match].mean()
        elif value_action == "min":
            result = data[match].min()
        elif value_action == "max":
            result = data[match].max()
        else:
            dispatcher.utter_message(
                text=f"I can only calculate 'min', 'max', or 'mean'. You requested '{value_action}'.")
            return []

        if not column_name:
            dispatcher.utter_message(text="I couldn't understand the column name. Can you clarify?")
            return []

        # Simulate processing the column name
        dispatcher.utter_message(text=f"The {value_action} for {match} is {result}")
        return [SlotSet("column_name", column_name), SlotSet("value_action", value_action)]