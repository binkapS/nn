//! Built-in demo corpus for out-of-the-box training. Kept small, highly
//! structured, and self-similar on purpose: a tiny character-level model can
//! learn its patterns in a few hundred steps, making the demo self-verifying
//! (loss falls fast, generated text looks like the corpus).

pub const DEMO_TEXT: &str = r#"the moon rose over the quiet town and the lamps began to glow
the baker was up before dawn kneading dough for the morning bread
a cat sat on the wall watching the carts roll past the square
old tom carried his toolbox and hummed a tune about the sea

"where are you going so early?" called the baker from the door
"i have a lamp to mend and a fence to paint," said tom with a smile
"and when the work is done i always come back for a piece of pie"
the baker laughed and said "then hurry home before the bread gets cold"

the wind blew softly through the elm trees on the hill
schoolchildren ran along the lane with satchels full of books
a bell rang out at eight and the market opened all its doors
merchants spread their colorful cloth and weighed the sacks of corn

the river slipped past the mill dropping leaves along the bank
a heron stood quite still and struck at fish in the glimmering water
millers called the water "the slow machine that never burns its fuel"
and they fed the wheel all day while the sky drifted grey and white

"have you seen my kite?" asked little anna of the postman
"it flew past the church tower and landed near the pond"
the postman pointed with his cap and anna thanked him twice
she ran the whole way home to tell her mother of the rescue

the evening brought the gentle smell of supper from the windows
families gathered round the table sharing stories of the day
the clock struck nine and the stars came out one after another
and the town fell asleep to dreams of ships and hills and rain

tom mended the lamp and painted the fence a bright calm green
he nailed the loose step by the gate and oiled the creaky hinge
"it is honest work," he said, "and a town is held together
by a hundred small repairs that no one ever sees or names"

the baker saved him the end of the loaf and a wedge of warm pie
"you have earned the sweetest piece," she promised with a grin
they watched the clouds turn pink and gold above the rooftops
and agreed the quiet town was the best place in the world

morning after morning the same small miracles kept arriving
the kettle sang, the sparrow argued, the letter box clicked shut
the baker counted loaves while tom counted up the sun
and between them the whole town learned its rhythm like a song

"nothing much changes here," said the teacher with a shrug
"and yet nothing here is ever quite the same twice over"
the river wheel turned slowly, the heron turned to stone
and the days rolled by like carts upon a well-worn lane

when the first frost came the town drew on its woolen scarves
tom repaired the sledges and oiled the runners down the hill
the baker baked twice as much and the whole square smelled of spice
and every window glowed with the first deep gold of winter

anna hung her kite above the fire and told the story proudly
"it sailed right past the tower and the wind had its way
but the postman caught it gently like a bird upon a tree"
the family clapped their hands and let the kettle join the music

so end the little chronicles of the town beneath the hill
where lamps are mended, fences painted, bread is warm and sweet
where a cat can watch the world from any wall it likes
and every quiet morning starts exactly like this one:
"#;
