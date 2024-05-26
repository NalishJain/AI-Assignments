# Breakers and Brokers Travel Recommendation System
This Python project provides a travel recommendation system that suggests destinations based on user preferences, budget, and transport type. Users can also add feedback and new destinations to improve the system over time.

## Brief Description
The system performs the following tasks:

### Data Loading:
Loads previously saved destination data from a pickle file (destination_data.pkl).

### Destination and Transport Types:
Defines several dictionaries for categorizing destinations (e.g., religious, camping, mountain) and transport modes (e.g., railways, car/bus, flight, ship).

### Recommendation Function:
1. show_details: Displays recommended destinations based on user preferences, budget, and transport type. It allows users to give feedback or check existing feedback and ratings.
2. recommend_destination: Gathers user preferences (age, budget, destination type, transport type, etc.) and calls show_details to show recommendations.
   
### Additional Functions:
1. add_new_destination: Allows users to add new destinations to the system.
2. add_feedback: Allows users to add feedback and ratings for destinations.
   
### Main Execution:
Greets the user and starts the recommendation process by calling recommend_destination.

# Search Algorithms
Contains the scratch implentation in python of:
1. ## Dijkstra
2. ## A* Search
3. ## Greedy
# Bayesian Networks
# Linear and Logistic Regression
