; Who descends from Queen Victoria? Wikidata has no descendant property, and none of its 1,790 entity-valued
; properties is called "parent of"; the question uses its own words and its own rules.
(gloss parent_of "{x} is a parent of {y}")
(gloss descendant_of "{x} is a descendant of {y}")
(= (descendant_of $X $Y) (parent_of $Y $X))
(= (descendant_of $X $Z) (let True (parent_of $Y $X) (descendant_of $Y $Z)))
(query (descendant_of $X queen_victoria))
