from char_data import CharDataset, build_char_vocab, encode


s = """
I don’t know if you have had the same experience, but the snag I always come up against when I’m telling a story is this dashed difficult problem of where to begin it. It’s a thing you don’t want to go wrong over, because one false step and you’re sunk. I mean, if you fool about too long at the start, trying to establish atmosphere, as they call it, and all that sort of rot, you fail to grip and the customers walk out on you.

Get off the mark, on the other hand, like a scalded cat, and your public is at a loss. It simply raises its eyebrows, and can’t make out what you’re talking about.

And in opening my report of the complex case of Gussie Fink-Nottle, Madeline Bassett, my Cousin Angela, my Aunt Dahlia, my Uncle Thomas, young Tuppy Glossop and the cook, Anatole, with the above spot of dialogue, I see that I have made the second of these two floaters.

I shall have to hark back a bit. And taking it for all in all and weighing this against that, I suppose the affair may be said to have had its inception, if inception is the word I want, with that visit of mine to Cannes. If I hadn’t gone to Cannes, I shouldn’t have met the Bassett or bought that white mess jacket, and Angela wouldn’t have met her shark, and Aunt Dahlia wouldn’t have played baccarat.

Yes, most decidedly, Cannes was the point d’appui.

Right ho, then. Let me marshal my facts.

I went to Cannes—leaving Jeeves behind, he having intimated that he did not wish to miss Ascot—round about the beginning of June. With me travelled my Aunt Dahlia and her daughter Angela. Tuppy Glossop, Angela’s betrothed, was to have been of the party, but at the last moment couldn’t get away. Uncle Tom, Aunt Dahlia’s husband, remained at home, because he can’t stick the South of France at any price.

So there you have the layout—Aunt Dahlia, Cousin Angela and self off to Cannes round about the beginning of June.

All pretty clear so far, what?

We stayed at Cannes about two months, and except for the fact that Aunt Dahlia lost her shirt at baccarat and Angela nearly got inhaled by a shark while aquaplaning, a pleasant time was had by all.

On July the twenty-fifth, looking bronzed and fit, I accompanied aunt and child back to London. At seven p.m. on July the twenty-sixth we alighted at Victoria. And at seven-twenty or thereabouts we parted with mutual expressions of esteem—they to shove off in Aunt Dahlia’s car to Brinkley Court, her place in Worcestershire, where they were expecting to entertain Tuppy in a day or two; I to go to the flat, drop my luggage, clean up a bit, and put on the soup and fish preparatory to pushing round to the Drones for a bite of dinner.
"""

itos, stoi = build_char_vocab(s)

print(f"{itos=}")
print(f"{itos=}")

tokens = encode(s, stoi)

print(f"{tokens=}")

ds = CharDataset(tokens, T=16)

x_train, y_train = next(iter(ds))

print(f"{x_train=}")
print(f"{y_train=}")
