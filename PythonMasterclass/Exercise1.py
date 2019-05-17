"""
Write a small program to ask for a name and an age. When both values have been entered, check if the person
is the right age to go on an 18-30 holiday (they must be over 18 and under 31). If they are, welcome them
to the holiday, otherwise print a (polite) message refusing them entry.
"""

name = input("Please enter your name: \n")
age = int(input("Please enter your age: \n"))
if (18<=age<31):
    print("Welcome to the holiday {}".format(name))
else:
    print("I am sorry {}, you do not fall between our age limit, thank you".format(name))
