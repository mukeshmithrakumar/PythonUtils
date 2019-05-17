"""
Write a Python program to print a long text, convert the string to a list and print all the words and their frequencies.
"""

string_words = ''' After ratifying the text on July 4, Congress issued the Declaration of
Independence in several forms. It was initially published as the printed Dunlap broadside
that was widely distributed and read to the public. The source copy used for this printing
has been lost, and may have been a copy in Thomas Jefferson's hand.[5] Jefferson's original
draft, complete with changes made by John Adams and Benjamin Franklin, and Jefferson's notes
of changes made by Congress, are preserved at the Library of Congress. The best-known version
of the Declaration is a signed copy that is displayed at the National Archives in Washington,
D.C., and which is popularly regarded as the official document. This engrossed copy was ordered
by Congress on July 19 and signed primarily on August 2.'''

cleaned = []
for word in string_words.split():
    # isalpha returns “True” if all characters in the string are alphabets
    if not word.strip('".,():[]5').isalpha():
        continue
    cleaned.append(word.strip('".,():[]5').lower())

output = []
for word in set(cleaned):
    word_freq = cleaned.count(word)
    output.append(tuple((word, word_freq)))
print(sorted(output, key=lambda x: x[1], reverse=True))
