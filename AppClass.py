import streamlit as st

class RegressionLayout:
    def __init__(self, title="Regression Models"):
        self.title = title
        self.regression_options = [
            "Linear Regression",
            "Lasso Regression",
            "Ridge Regression",
            "Random Forest Regression",
        ]
        if "selected_regression" not in st.session_state:
            st.session_state.selected_regression = "Linear Regression"
        self.selected_regression = st.session_state.selected_regression

    def display_title(self):
        st.title(self.title)

    def display_options(self):
        selected = st.radio(
            "Select a regression method:",
            self.regression_options,
            index=self.regression_options.index(st.session_state.selected_regression),
            horizontal=True,
        )

        if selected != st.session_state.selected_regression:
            st.session_state.selected_regression = selected
        self.selected_regression = st.session_state.selected_regression

    def is_selected(self, regression_type):
        """Check if a specific regression type is currently selected."""
        return self.selected_regression == regression_type

    def handle_selection(self):
        if self.is_selected("Linear Regression"):
            st.write("You selected Linear Regression.")
        elif self.is_selected("Lasso Regression"):
            st.write("You selected Lasso Regression.")
        elif self.is_selected("Ridge Regression"):
            st.write("You selected Ridge Regression.")
        elif self.is_selected("Random Forest Regression"):
            st.write("You selected Random Forest Regression.")

    def run(self):
        self.display_title()
        self.display_options()
        self.handle_selection()