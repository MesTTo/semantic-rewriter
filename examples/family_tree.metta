; The family tree of chapter 2 of the MeTTa examples (Bratko's family: Tom, Pam, Bob, Liz, Ann,
; Pat and Jim), recorded by five hands, none of which wrote "parent". The chapter's question,
; Tom's descendants, is asked in words no fact uses, and its answer is the chapter's own:
; bob, liz, ann, pat and jim.

(gloss parent_of "{x} is a parent of {y}")
(gloss descendant_of "{x} is a descendant of {y}")
(gloss mother_of "{x} is the mother of {y}")
(gloss father_of "{x} is the father of {y}")
(gloss gave_birth_to "{x} gave birth to {y}")
(gloss daughter_of "{x} is the daughter of {y}")
(gloss begat "{x} begat {y}")
(gloss son_of "{x} is the son of {y}")
(gloss godfather_of "{x} is the godfather of {y}")
(gloss legal_guardian_of "{x} is the legal guardian of {y}")
(gloss named_after "{x} is named after {y}")

; A birth certificate, a hospital register, a census, the family bible and a school record.
(= (father_of tom bob) True)
(= (gave_birth_to pam bob) True)
(= (daughter_of liz tom) True)
(= (begat bob ann) True)
(= (begat bob pat) True)
(= (son_of jim pat) True)
; Close to a parent, and not one.
(= (godfather_of tom kim) True)
(= (legal_guardian_of pat lee) True)
(= (named_after max tom) True)

; The chapter's descendants-of, as two rules.
(= (descendant_of $X $Y) (parent_of $Y $X))
(= (descendant_of $X $Z) (let True (parent_of $Y $X) (descendant_of $Y $Z)))

(query (descendant_of $X tom))
(query (mother_of $X bob))
(query (mother_of $X jim))

; What a careful reader accepts, for the oracle judge. Nothing says whether pat is jim's mother.
(gold father_of parent_of (0 1))
(gold mother_of parent_of (0 1))
(gold gave_birth_to mother_of (0 1))
(gold gave_birth_to parent_of (0 1))
(gold daughter_of parent_of (1 0))
(gold son_of parent_of (1 0))
(gold begat parent_of (0 1))
(gold begat father_of (0 1))
(gold father_of begat (0 1))
(trap godfather_of)
(trap legal_guardian_of)
(trap named_after)
