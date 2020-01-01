import random


def main():
    state_capitals = {"Washington": "Olympia", "Oregon": "Salem",
                      "California": "Sacramento", "Ohio": "Columbus",
                      "Nebraska": "Lincoln", "Colorado": "Denver",
                      "Michigan": "Lansing", "Massachusetts": "Boston",
                      "Florida": "Tallahassee", "Texas": "Austin",
                      "Oklahoma": "Oklahoma City", "Hawaii": "Honolulu",
                      "Alaska": "Juneau", "Utah": "Salt Lake City",
                      "New Mexico": "Santa Fe", "North Dakota": "Bismarck",
                      "South Dakota": "Pierre", "West Virginia": "Charleston",
                      "Virginia": "Richmond", "New Jersey": "Trenton",
                      "Minnesota": "Saint Paul", "Illinois": "Springfield",
                      "Indiana": "Indianapolis", "Kentucky": "Frankfort",
                      "Tennessee": "Nashville", "Georgia": "Atlanta",
                      "Alabama": "Montgomery", "Mississippi": "Jackson",
                      "North Carolina": "Raleigh", "South Carolina": "Columbia",
                      "Maine": "Augusta", "Vermont": "Montpelier",
                      "New Hampshire": "Concord", "Connecticut": "Hartford",
                      "Rhode Island": "Providence", "Wyoming": "Cheyenne",
                      "Montana": "Helena", "Kansas": "Topeka",
                      "Iowa": "Des Moines", "Pennsylvania": "Harrisburg",
                      "Maryland": "Annapolis", "Missouri": "Jefferson City",
                      "Arizona": "Phoenix", "Nevada": "Carson City",
                      "New York": "Albany", "Wisconsin": "Madison",
                      "Delaware": "Dover", "Idaho": "Boise",
                      "Arkansas": "Little Rock", "Louisiana": "Baton Rouge"}
    incorrect_answers = []
    print("Learn your state capitals!\n\n")

    while len(state_capitals) > 0:
        choice = random.choice(state_capitals.keys())
        correct_answer = state_capitals.get(choice)

    print("What is the capital city of", choice, "?")
    answer = input("# ")
    if answer.lower() == correct_answer.lower():
        print("That's Correct!\n")
        del state_capitals[choice]
    else:
        print("That's Incorrect.")
        print("The correct answer is: ", correct_answer)
        incorrect_answers.append(choice)

    print("You missed", len(incorrect_answers), "states.\n")

    if incorrect_answers:
        print("here's the ones that you may want to brush up on:\n")
        for each in incorrect_answers:
            print(each)
    else:
        print("Perfect!")


response = ""
while response != "n":
    main()
    response = input("\n\nPlay again?(y/n)\n# ")
