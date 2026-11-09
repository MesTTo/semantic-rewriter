; What does Alphabet own, directly or through what it owns?
(gloss owns "{x} owns {y}")
(= (owns $X $Z) (let True (owns $X $Y) (owns $Y $Z)))
(query (owns alphabet_inc $C))
