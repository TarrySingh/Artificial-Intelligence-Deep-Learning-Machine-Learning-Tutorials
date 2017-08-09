import scrabble

# Print all the words containing uu

for word in scrabble.wordlist:
    if "uu" in word:
        print(word)