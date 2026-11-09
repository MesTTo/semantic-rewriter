; Literature-based discovery, after Swanson (1986): fish oil might help Raynaud's syndrome.
; No paper below says so. The chain runs through two literatures that never cite each other,
; each written in its own words, and Swanson's rule is stated once, in the reader's words.
; PyMeTTa's examples/reasoning/literature_discovery.py closes the same gap with an embedding;
; here a judge decides which phrasings entail the rule's, and in which direction.

(gloss lowers "{x} lowers the level of {y}")
(gloss aggravates "{x} aggravates the disease {y}")
(gloss worth_testing_against "{x} is worth testing as a treatment for the disease {y}")
(gloss is_rich_in "{x} is rich in {y}")
(gloss reduces "{x} reduces the level of {y}")
(gloss is_reduced_by "the level of {x} is reduced by {y}")
(gloss exacerbates "{x} exacerbates the disease {y}")
(gloss is_made_worse_by "the disease {x} is made worse by {y}")
(gloss increases "{x} raises the level of {y}")
(gloss is_associated_with "{x} is associated with the disease {y}")

; The nutrition literature.
(= (reduces omega_3 blood_viscosity) True)
(= (is_reduced_by platelet_aggregation omega_3) True)
(= (reduces aspirin inflammation) True)
; The vascular literature.
(= (is_made_worse_by raynauds blood_viscosity) True)
(= (exacerbates platelet_aggregation raynauds) True)
(= (is_associated_with inflammation raynauds) True)
(= (increases dehydration blood_viscosity) True)
; Food chemistry.
(= (is_rich_in fish_oil omega_3) True)

; Swanson's rule: what lowers something that aggravates a condition is worth testing against it.
(= (worth_testing_against $A $C) (let True (lowers $A $B) (aggravates $B $C)))
; A food lowers whatever a compound it is rich in lowers.
(= (lowers $F $B) (let True (is_rich_in $F $A) (lowers $A $B)))

(query (worth_testing_against $X raynauds))

; What a careful reader accepts, for the oracle judge: an association is not aggravation,
; and an increase is the opposite of lowering.
(gold reduces lowers (0 1))
(gold is_reduced_by lowers (1 0))
(gold exacerbates aggravates (0 1))
(gold is_made_worse_by aggravates (1 0))
(trap increases)
(trap is_associated_with)
