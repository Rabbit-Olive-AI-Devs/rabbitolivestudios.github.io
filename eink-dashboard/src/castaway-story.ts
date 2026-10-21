/**
 * The castaway's story bank (DECISIONS #93). SPOILERS: the owner asked for this page to be a
 * surprise, so nothing here is shown to them before it airs.
 *
 * The hero is SETE, a small four-slab delivery robot washed up on an island with one palm tree.
 * It still has the parcel it was carrying, addressed to "T.", and will not open it: delivering it
 * is its prime directive. Around it: Capitão, a bossy crab; Larápio, a gull who steals its bolts;
 * the Neighbour, a light on the far island that answers its signals; ships that never see it.
 *
 * Text rules: a beat's `text` fits two lines of about 66 characters; a `log` entry about 24.
 * No canon film names. Plain ASCII except the names (the fonts carry accented capitals only, so
 * names are written without accents: CAPITAO, LARAPIO).
 */

import type { Face, Pose, Prop } from "./hud/castaway-draw";

export interface Beat {
  text: string;
  log: string;
  pose: Pose;
  face: Face;
  at?: number;              // 0..1 where on the island; default 0.4
  props?: Prop[];
}

/** A storyline over consecutive days: three beats a day (morning, afternoon, evening). */
export interface Arc {
  id: string;
  title: string;
  days: Array<[Beat, Beat, Beat]>;
}

const b = (text: string, log: string, pose: Pose, face: Face, props: Prop[] = [], at = 0.4): Beat => ({ text, log, pose, face, props, at });

/** The first days: how SETE arrived. Always days 1-3. */
export const ARRIVAL: Arc = {
  id: "arrival", title: "WASHED ASHORE",
  days: [
    [
      b("A box with four legs crawls out of the surf. Systems rebooting... location: one island, one palm tree, zero roads.", "Arrived. Damp.", "sit", "surprised", ["bottle"], 0.7),
      b("SETE checks the parcel strapped to its back. Still sealed. Still addressed to 'T.'. Prime directive: deliver it.", "Parcel: safe.", "stand", "focused", [], 0.5),
      b("First night. SETE counts 1,412 stars and one tiny light blinking on the far island. It blinks back.", "Someone out there?", "think", "calm", ["flash"], 0.55),
    ],
    [
      b("A crab inspects SETE's left foot and decides it is in charge now. SETE names it Capitao. Capitao agrees.", "Met Capitao (boss).", "stand", "happy", ["crab"], 0.45),
      b("Inventory: 22 bolts, 1 parcel, 0 boats. A gull lands, takes bolt number 22 and leaves without a word.", "Bolts: 21. Gull: rude.", "wave", "sad", ["gull", "crab"], 0.6),
      b("SETE names the gull Larapio. Capitao clicks in approval. A war has begun.", "Declared war on gull.", "think", "focused", ["crab"], 0.5),
    ],
    [
      b("Plan: build a radio mast, call for help, deliver parcel. Step one: find something tall. The palm declines.", "Plan made. Palm: no.", "think", "focused", [], 0.5),
      b("Two sticks and a vine become the first piece of the mast. Engineering is 90% optimism.", "Mast: started.", "dig", "happy", [], 0.7),
      b("The far light blinks three times. SETE blinks three times. Nothing else happens. It is the best talk all week.", "Talked to the light.", "wave", "love", ["flash", "fire"], 0.55),
    ],
  ],
};

/** Storylines in no fixed order after the arrival; each runs on consecutive days. */
export const ARCS: readonly Arc[] = [
  {
    id: "bottle", title: "THE MESSAGE IN THE BOTTLE",
    days: [
      [
        b("A bottle rolls up the beach. Inside: a note in a hand SETE does not know. It says only: 'Hello?'", "Found a bottle.", "dig", "surprised", ["bottle"], 0.72),
        b("SETE drafts a reply. Draft 14: 'Hello. I am a robot. I have a parcel. Do you know T.?' Capitao edits it down.", "Wrote a reply.", "sit", "focused", ["book", "crab"], 0.5),
        b("The bottle goes back into the sea at sunset with the reply inside. The current takes it toward the far light.", "Bottle away.", "wave", "calm", [], 0.7),
      ],
      [
        b("No bottle. SETE checks the beach every 15 minutes, which is how often it checks everything.", "No bottle (yet).", "walk", "calm", ["crab"], 0.6),
        b("Larapio steals the cork from SETE's spare bottle. SETE now owns one bottle that cannot keep a secret.", "Larapio: 1 cork.", "kick", "sad", ["gull", "bottle"], 0.5),
        b("At night the far light blinks a long pattern. SETE writes it down. It might be a word. It might be a moth.", "Light said something.", "think", "focused", ["flash", "fire"], 0.55),
      ],
      [
        b("A bottle! Same handwriting: 'Robot? Really? I collect shells. What is in the parcel?'", "Reply came back!", "dance", "love", ["bottle"], 0.7),
        b("SETE writes: 'Classified.' Then adds: 'I do not know either.' Then crosses it out. Then leaves it in.", "Honesty: 90%.", "sit", "wink", ["book"], 0.5),
        b("Two pen pals and an ocean in between. SETE files the note under IMPORTANT, a folder it made today.", "New folder: IMPORTANT.", "sit", "happy", ["fire"], 0.5),
      ],
    ],
  },
  {
    id: "raft", title: "OPERATION RAFT",
    days: [
      [
        b("New plan: sail to the far light. Materials: driftwood, vines, and a sail made from a very old towel.", "Raft plan approved.", "dig", "focused", ["raft"], 0.7),
        b("Capitao inspects the raft and sits on it. Capitao is now the captain of the raft. SETE is crew.", "Demoted to crew.", "stand", "calm", ["raft", "crab"], 0.65),
        b("Sea trial at dusk. The raft floats. SETE does not, quite. It walks back along the bottom, waving.", "Trial: partial float.", "wave", "surprised", ["raft"], 0.74),
      ],
      [
        b("Raft 2.0 gets outriggers made of coconuts. Larapio steals one coconut. Raft 2.0 now leans left.", "Raft leans left.", "dig", "sad", ["raft", "gull"], 0.7),
        b("A ship passes on the horizon. SETE waves both outer slabs. The ship keeps going, as ships do.", "Ship #4: didn't see.", "wave", "surprised", ["ship", "raft"], 0.6),
        b("The towel-sail flaps all night like applause. SETE accepts it.", "Sail: applauding.", "sit", "happy", ["raft", "fire"], 0.5),
      ],
      [
        b("Final test. The raft sails 40 metres, turns around by itself and comes home. Capitao calls it a success.", "Raft came home.", "stand", "happy", ["raft", "crab"], 0.74),
        b("Lesson logged: the island seems to like having SETE around. SETE logs that it likes it back. A little.", "Island: 1, Sea: 0.", "sit", "calm", ["raft"], 0.5),
        b("The raft becomes a very good bench. Sunset seats for two, one of them a crab.", "Raft is now a bench.", "sit", "love", ["raft", "crab"], 0.65),
      ],
    ],
  },
  {
    id: "football", title: "THE COCONUT CUP",
    days: [
      [
        b("A coconut, round enough. Two sticks for a goal. SETE invents football, or remembers it from a crate label.", "Invented football.", "kick", "happy", ["ball"], 0.5),
        b("Keepy-uppy record: 3. The coconut is not built for this, and neither, technically, is SETE.", "Record: 3 touches.", "kick", "focused", ["ball"], 0.55),
        b("Capitao is appointed referee. He shows SETE a red card, which is a leaf. SETE appeals. Appeal denied.", "Sent off by a crab.", "stand", "sad", ["ball", "crab"], 0.5),
      ],
      [
        b("Training day. SETE practises the bicycle kick and lies in the sand for a while, studying the sky.", "Bicycle kick: no.", "sleep", "surprised", ["ball"], 0.55),
        b("Record: 11 touches! The crowd (one crab, one gull) goes mild. Larapio takes the coconut at 12.", "Record: 11. Then theft.", "dance", "happy", ["ball", "gull", "crab"], 0.55),
        b("SETE paints a red and black stripe on its middle slab with berry juice. It does not know why. It feels right.", "New stripe. Feels right.", "stand", "love", ["fire"], 0.5),
      ],
      [
        b("Cup final: SETE vs the wind. The wind scores early. SETE equalises with a header nobody saw coming.", "Final: 1-1 at half.", "kick", "focused", ["ball"], 0.55),
        b("Extra time. Penalties. SETE takes all five, the wind saves none. Champions of the Coconut Cup!", "CHAMPIONS!", "dance", "love", ["ball", "flag"], 0.55),
        b("The trophy is a shell on a stick. It goes on the shelf, which is also a stick.", "Trophy: shell.", "sit", "happy", ["fire", "flag"], 0.45),
      ],
    ],
  },
  {
    id: "neighbour", title: "SIGNALS",
    days: [
      [
        b("SETE builds a mirror from a crate lid and a lot of polishing. Today it will answer the far light by day.", "Polished a mirror.", "dig", "focused", [], 0.6),
        b("Flash, flash, pause, flash. Across the water something flashes back the same. SETE spins in a circle.", "It flashed back!", "dance", "love", ["flash"], 0.7),
        b("SETE invents an alphabet of flashes. Letter A takes 40 minutes to agree on.", "Alphabet: 1 letter.", "think", "focused", ["flash", "fire"], 0.55),
      ],
      [
        b("Progress: A, B, C and a sign that probably means 'good morning' or possibly 'stop that'.", "Alphabet: 3.5 letters.", "wave", "happy", ["flash"], 0.7),
        b("Clouds. No flashes today. SETE talks to Capitao instead, who is an excellent listener and a terrible replier.", "Cloudy. Talked to crab.", "sit", "calm", ["crab"], 0.5),
        b("Late at night the light flashes slowly: T... T... T... SETE checks the parcel. 'T.' The circuits tingle.", "It said T!", "think", "surprised", ["flash"], 0.55),
      ],
      [
        b("SETE flashes: 'WHO IS T?' The answer takes all morning: a single long flash, then nothing. Mysterious. Rude.", "Asked who T is.", "wave", "focused", ["flash"], 0.7),
        b("Capitao suggests patience by sitting on SETE's foot for two hours. It works.", "Patience: forced.", "sit", "calm", ["crab"], 0.5),
        b("Goodnight sign: three short flashes. SETE answers with three. Some conversations do not need words.", "Goodnight, Neighbour.", "wave", "love", ["flash", "fire"], 0.55),
      ],
    ],
  },
  {
    id: "storm", title: "THE BIG STORM",
    days: [
      [
        b("The sea goes flat and the gulls go quiet. SETE's barometer, a shell, says 'trouble'.", "Barometer: trouble.", "think", "focused", [], 0.5),
        b("SETE ties everything down: the hut, the mast, the parcel, and Capitao, who objects in writing.", "Tied down the island.", "dig", "focused", ["crab"], 0.45),
        b("The storm arrives. SETE folds into its smallest shape around the parcel and waits.", "Storm. Holding on.", "sit", "sad", [], 0.45),
      ],
      [
        b("Morning after. The palm stands, the hut leans, and a crate has washed up full of... rubber ducks.", "Storm gift: 300 ducks.", "stand", "surprised", ["bottle"], 0.65),
        b("SETE lines up the ducks on the beach as an emergency sign. Planes would surely notice 300 ducks.", "Duck SOS deployed.", "dig", "happy", ["sign"], 0.7),
        b("Nobody notices. The ducks float away at high tide, one by one, as if they had somewhere to be.", "Ducks: departed.", "wave", "calm", [], 0.7),
      ],
    ],
  },
  {
    id: "garden", title: "THE GARDEN",
    days: [
      [
        b("A seed sprouts behind the hut. SETE does not eat, but it decides to grow something anyway. Purpose!", "Found a sprout.", "dig", "love", [], 0.35),
        b("Watering schedule: every 15 minutes. Capitao warns this is too much. Capitao is right, for once.", "Overwatered. Oops.", "dig", "sad", ["crab"], 0.4),
        b("SETE sings to the sprout in binary. Plants respond to music, the manual says. Which manual? Unclear.", "Sang to a plant.", "sit", "happy", ["fire"], 0.4),
      ],
      [
        b("Two leaves! SETE takes 400 measurements and builds a tiny fence against Larapio.", "Two leaves. Fence up.", "dig", "happy", [], 0.4),
        b("Larapio lands on the fence, looks at the sprout, and leaves it alone. Even gulls have limits.", "Gull truce (plants).", "stand", "surprised", ["gull"], 0.45),
        b("Night watch by the garden. The far light blinks; SETE flashes back 'GROWING THINGS'. Takes an hour.", "Told Neighbour: plant.", "wave", "calm", ["flash", "lantern"], 0.45),
      ],
    ],
  },
  {
    id: "visitor", title: "THE TURTLE",
    days: [
      [
        b("A sea turtle comes ashore, very old and very unbothered. SETE bows. The turtle does not.", "Visitor: turtle.", "stand", "surprised", ["turtle"], 0.6),
        b("The turtle digs a nest. SETE stands guard all afternoon, which nobody asked for and everybody needs.", "Guarding a nest.", "stand", "focused", ["turtle", "nest"], 0.6),
        b("The turtle leaves at night. SETE keeps watching the nest. Some deliveries take weeks.", "Turtle left. Eggs: 3.", "sit", "calm", ["nest", "fire"], 0.55),
      ],
      [
        b("Egg check, 06:00. Egg check, 06:15. Egg check, 06:30. Capitao files a complaint.", "Egg checks: many.", "think", "focused", ["nest", "crab"], 0.6),
        b("Larapio circles the nest. SETE stands under him with a very serious face screen. Larapio reconsiders.", "Defended the nest.", "wave", "focused", ["nest", "gull"], 0.6),
        b("A crack! Three baby turtles run for the moonlit sea. SETE escorts them to the water's edge. Mission complete.", "3 turtles delivered!", "dance", "love", ["nest"], 0.72),
      ],
    ],
  },
  {
    id: "telescope", title: "STARGAZING",
    days: [
      [
        b("A brass tube washes up. With two lenses from the crate, it becomes a telescope. Capitao looks first.", "Built a telescope.", "dig", "happy", ["telescope", "crab"], 0.5),
        b("Daytime test: SETE spots a ship, a whale and Larapio's nest, which is full of SETE's bolts.", "Found the bolt stash!", "stand", "surprised", ["telescope", "whale"], 0.5),
        b("Night. SETE names stars after everyone it knows. That takes three stars. It saves the rest.", "Named 3 stars.", "sit", "love", ["telescope"], 0.45),
      ],
      [
        b("Bolt recovery mission: SETE wades to Larapio's rock. Recovered 9 bolts. Lost dignity. Net positive.", "Bolts: +9.", "walk", "happy", ["gull"], 0.7),
        b("SETE draws a star map in the sand. The tide erases it, so it draws it again. Then again.", "Star map: v3.", "dig", "calm", [], 0.55),
        b("A shooting star. SETE makes a wish. Robots are not supposed to. It does it twice, to be sure.", "Wished. Twice.", "think", "love", ["telescope", "stars-shoot"], 0.45),
      ],
    ],
  },
  {
    id: "radio", title: "STATIC",
    days: [
      [
        b("SETE builds a radio from a tin, a coil of wire and pure stubbornness. It hisses. Hissing is a start.", "Radio hisses.", "dig", "focused", ["radio"], 0.5),
        b("For three seconds a voice breaks through: '...goal...!' Then static. SETE plays it back 200 times.", "Heard a voice!", "dance", "surprised", ["radio"], 0.5),
        b("Late night, the radio finds a long, slow guitar solo that seems to go on forever. SETE sways until it ends.", "Guitar solo night.", "dance", "love", ["radio", "crab", "fire"], 0.5),
      ],
      [
        b("SETE tries to transmit: 'This is SETE. Parcel for T. Over.' The radio answers with a long, rude buzz.", "Transmitted. Buzz.", "stand", "focused", ["radio"], 0.5),
        b("It climbs the mast to fix the wire. Larapio sits on top of the mast to supervise. Unhelpful.", "Mast repair. Gull help.", "stand", "sad", ["gull", "radio"], 0.72),
        b("Another voice in the static: '...ilha... ouvindo...' SETE does not know the words, but it keeps listening.", "Someone listening?", "think", "surprised", ["radio", "fire"], 0.5),
      ],
    ],
  },
  {
    id: "banner", title: "AIR MAIL",
    days: [
      [
        b("A small plane drags a banner across the sky. SETE cannot read it from here, so it guesses: 'HI SETE'.", "Banner: probably HI.", "wave", "love", ["plane"], 0.55),
        b("SETE writes HI in rocks on the beach, 10 metres tall. The plane is long gone. The rocks look proud.", "Wrote HI in rocks.", "dig", "happy", ["sign"], 0.65),
        b("Capitao rearranges two rocks so it says HO. Comedy, apparently.", "Capitao: comedian.", "stand", "wink", ["crab", "sign"], 0.6),
      ],
      [
        b("The plane again! This time SETE flashes its mirror. The plane waggles its wings. Probably wind.", "Plane waggled!", "wave", "surprised", ["plane", "flash"], 0.6),
        b("A tiny parachute drops into the sea: a box of crayons. SETE reads this as encouragement.", "Air drop: crayons.", "dig", "happy", [], 0.75),
        b("SETE draws a self-portrait on the hut wall. It gives itself a cape. Artistic licence.", "Self-portrait (cape).", "sit", "wink", ["fire"], 0.4),
      ],
    ],
  },
  {
    id: "sandal", title: "THE LOST SANDAL",
    days: [
      [
        b("One flip-flop washes up, left foot, size 42. Somewhere out there, someone is walking in a circle.", "Found: 1 sandal.", "dig", "surprised", ["bottle"], 0.72),
        b("SETE launches an investigation. Suspects: a tourist, a pirate, a one-legged seagull. Larapio has an alibi.", "Investigation open.", "think", "focused", ["gull"], 0.5),
        b("SETE puts the sandal on a stick by the shore, so its owner can see it from a ship. Lost and found, island edition.", "Lost and found: open.", "stand", "calm", ["sign"], 0.7),
      ],
      [
        b("No owner. Capitao moves into the sandal. It has a view, a roof and room for guests.", "Capitao moved house.", "stand", "happy", ["crab"], 0.5),
        b("SETE writes a note for the sandal's owner and floats it out: 'Your shoe is safe. Your crab is new.'", "Note sent to owner.", "wave", "wink", ["bottle"], 0.72),
        b("Case closed, sort of. SETE files it under MYSTERIES, a folder that keeps getting bigger.", "Case: mostly closed.", "sit", "calm", ["fire"], 0.5),
      ],
    ],
  },
  {
    id: "whale", title: "THE WHALE SONG",
    days: [
      [
        b("A whale surfaces off the reef and blows a rainbow into the air. SETE forgets to blink, which it never does anyway.", "A whale!", "stand", "surprised", ["whale"], 0.7),
        b("The whale sings: a long, low note that makes SETE's slabs hum. SETE hums back. Badly.", "Duet with a whale.", "dance", "love", ["whale"], 0.7),
        b("At night the song comes again under the moon. SETE records it and plays it to Capitao, who cries a little.", "Recorded whale song.", "sit", "love", ["whale", "fire", "crab"], 0.55),
      ],
      [
        b("The whale is gone. SETE checks the reef every 15 minutes. You know how it is.", "Whale gone. Checking.", "walk", "sad", [], 0.7),
        b("SETE builds a speaker from a shell and plays the song toward the sea, in case the whale wants it back.", "Played song to sea.", "stand", "calm", ["radio"], 0.68),
        b("Far out, a spout of white in the moonlight. Then another. The whale answered. SETE is sure of it.", "It answered.", "wave", "love", ["whale"], 0.7),
      ],
    ],
  },
  {
    id: "dj", title: "RADIO ILHA FM",
    days: [
      [
        b("The radio can transmit for 30 seconds before it overheats. SETE plans a radio station. 30 seconds a day.", "Radio station plan.", "think", "focused", ["radio"], 0.5),
        b("First broadcast: 'Good afternoon, ocean. Weather: sunny. Traffic: one crab. Next: the hit song Beep Boop.'", "On air: 30 seconds.", "dance", "happy", ["radio", "crab"], 0.5),
        b("Listener feedback: one flash from the far light. SETE counts it as a fan club.", "Fan club: 1 member.", "sit", "love", ["radio", "flash", "fire"], 0.5),
      ],
      [
        b("Morning show: SETE reads the news. Headline: COCONUT FALLS. Exclusive interview with the palm.", "News: coconut fell.", "sit", "focused", ["radio"], 0.5),
        b("Request hour. Capitao requests silence. SETE plays four seconds of silence, then a drum solo on the hut.", "Played a drum solo.", "dance", "wink", ["radio", "crab"], 0.5),
        b("Sign-off: 'This is Radio Ilha FM, broadcasting from the middle of nowhere, to somewhere. Goodnight, T.'", "Goodnight, listeners.", "wave", "love", ["radio", "fire"], 0.5),
      ],
    ],
  },
  {
    id: "fog", title: "THE FOG",
    days: [
      [
        b("Fog rolls in so thick that SETE cannot see the palm, and the palm is right there. Probably.", "Fog. Palm: missing.", "walk", "surprised", [], 0.5),
        b("A foghorn sounds somewhere close. A ship! SETE bangs a pot like a lighthouse with no light.", "Banged a pot at a ship.", "wave", "focused", ["ship"], 0.7),
        b("The horn fades away. SETE keeps banging for an hour, just in case. Capitao wears earplugs, which are pebbles.", "Banged for an hour.", "stand", "sad", ["crab", "fire"], 0.5),
      ],
      [
        b("The fog lifts. On the beach: footprints. Not crab, not gull, not robot. SETE's circuits go cold.", "Footprints?!", "stand", "surprised", [], 0.62),
        b("SETE follows them to the water. They end at the sea, as if someone walked in. Or out.", "Tracks end at sea.", "walk", "focused", [], 0.72),
        b("Night watch with the lantern. Nothing comes. In the morning there is a shell on the parcel, placed very neatly.", "A gift? From whom?", "sit", "focused", ["lantern"], 0.5),
      ],
    ],
  },
  {
    id: "chess", title: "THE GRAND TOURNAMENT",
    days: [
      [
        b("SETE draws a chessboard in the sand and makes pieces from shells. Capitao is challenged. Capitao accepts.", "Challenged a crab.", "dig", "focused", ["crab"], 0.5),
        b("Capitao moves sideways only. SETE argues this is not chess. Capitao argues it is crab chess. Fair.", "Crab rules adopted.", "sit", "calm", ["crab"], 0.5),
        b("Game one ends when the tide eats the board. Draw by ocean.", "Draw by ocean.", "sit", "surprised", ["crab"], 0.5),
      ],
      [
        b("Board moved higher up the beach. Game two. SETE thinks 14 moves ahead. Capitao thinks about lunch.", "Game two begins.", "think", "focused", ["crab"], 0.45),
        b("Larapio flies in and takes the queen. Unclear whose. Both players agree to blame the bird.", "Gull took the queen.", "kick", "sad", ["gull", "crab"], 0.5),
        b("Final score: SETE 0, Capitao 0, Larapio 1 queen. A rematch is scheduled for every day forever.", "Tournament: unresolved.", "sit", "happy", ["crab", "fire"], 0.5),
      ],
    ],
  },
  {
    id: "capitao", title: "CAPITAO'S BIG DAY",
    days: [
      [
        b("Capitao has outgrown his shell. He needs a new one and is extremely dramatic about it.", "Capitao: homeless.", "stand", "surprised", ["crab"], 0.5),
        b("SETE holds a house viewing: three shells, a bottle cap and the lost sandal. Capitao rejects them all.", "House viewing: no.", "dig", "focused", ["crab"], 0.55),
        b("At sunset Capitao picks the shell SETE polished first, of course. A party is announced for tomorrow.", "Capitao: new shell!", "dance", "happy", ["crab"], 0.5),
      ],
      [
        b("Housewarming! Guests: SETE, a starfish, and Larapio, who was not invited but brought a bolt back. Truce?", "Larapio returned a bolt!", "dance", "love", ["crab", "gull"], 0.5),
        b("Party games: shell toss, limbo under the hammock, and staring at the sea, which everyone wins.", "Party games.", "kick", "happy", ["crab", "hammock"], 0.55),
        b("Larapio leaves at dusk with two bolts. The truce lasted four hours, a record.", "Truce: 4 hours.", "wave", "wink", ["gull", "fire"], 0.5),
      ],
    ],
  },
  {
    id: "eclipse", title: "THE DARK NOON",
    days: [
      [
        b("SETE's solar panels report a problem: at noon, the sun is going to... wait. An eclipse? Today?", "Eclipse alert!", "think", "surprised", [], 0.5),
        b("The light goes thin and silver. The birds go quiet. SETE stands very still with its panels out, a little scared.", "Eclipse. Spooky.", "charge", "surprised", [], 0.55),
        b("The sun comes back. SETE logs it as 'the day the sky blinked'. Capitao claims he caused it.", "The sky blinked.", "sit", "happy", ["crab", "fire"], 0.5),
      ],
    ],
  },
  {
    id: "camera", title: "THIRTY-SIX FRAMES",
    days: [
      [
        b("An old rangefinder camera washes up in a tin, dry, with one roll of black and white film inside. 36 frames.", "Found: camera, 1 roll.", "dig", "surprised", ["bottle"], 0.72),
        b("SETE winds the film and does not press the shutter. Thirty-six frames is not many. It has to choose.", "Frames left: 36.", "think", "focused", [], 0.5),
        b("Frame 1: the palm at dusk, light from the left, Capitao in the corner by accident. Keeper.", "Frame 1: the palm.", "stand", "focused", ["crab"], 0.45),
      ],
      [
        b("SETE measures the light for an hour before frame 2. Capitao has moved on with his life.", "Light metering: long.", "stand", "focused", [], 0.55),
        b("Frame 2: the far island at noon, sharp as a pin. Frame 3: Larapio, mid-theft, blurred. Both keepers.", "Frames: 3 of 36.", "wave", "happy", ["gull"], 0.6),
        b("At night SETE thinks about the frames it did not take. It decides that is part of photography.", "Thinking in frames.", "sit", "calm", ["fire"], 0.5),
      ],
      [
        b("Frame 4 goes on a sunrise. Frame 5 on the same sunrise, one minute later, because it got better.", "Frames: 5 of 36.", "stand", "love", [], 0.5),
        b("No darkroom on the island. SETE stores the roll in the parcel pocket, to develop when it reaches T.", "Film: saved for T.", "sit", "calm", [], 0.5),
        b("Thirty-one frames to go. SETE now looks at everything as if it might be worth one. It is a nice way to look.", "Everything: maybe.", "sit", "love", ["fire"], 0.5),
      ],
    ],
  },
  {
    id: "board", title: "THE BOARD",
    days: [
      [
        b("A pink board with no wheels washes up. It is clearly meant to float in the air. It floats in water instead.", "Found: board, no wheels.", "dig", "surprised", [], 0.72),
        b("SETE tries to ride it across the sand. Physics files a formal objection. Physics wins.", "Physics: 1. SETE: 0.", "sleep", "surprised", [], 0.55),
        b("On the water, though, the board is fast. SETE skims the lagoon at sunset, faster than anything on the island.", "Rode it on water.", "dance", "love", [], 0.74),
      ],
      [
        b("SETE wonders where the board came from. A good question for a robot with plenty of time.", "Board: origin unknown.", "think", "focused", [], 0.5),
        b("Etched on the back: a date, years from now. SETE checks its clock twice and decides not to think about it.", "Board is from... when?", "think", "surprised", [], 0.5),
        b("The board goes on the hut wall, next to the trophy shell. The island has a museum now.", "Opened a museum.", "stand", "happy", ["fire"], 0.4),
      ],
    ],
  },
  {
    id: "city", title: "URBAN PLANNING",
    days: [
      [
        b("SETE draws a master plan for the island in the sand: residential, commercial, and one zone just for Capitao.", "Master plan: drafted.", "dig", "focused", ["crab"], 0.6),
        b("Population: 1 robot, 1 crab, 1 gull (disputed). Tax revenue: 0 bolts. Happiness: high.", "Census done.", "think", "happy", ["crab", "gull"], 0.5),
        b("Infrastructure night: SETE lays a road of flat stones from the hut to the beach. Six metres. A start.", "Built a road (6 m).", "dig", "happy", ["lantern"], 0.5),
      ],
      [
        b("Larapio is declared a natural disaster. Insurance does not cover gulls.", "Gull: natural disaster.", "wave", "sad", ["gull"], 0.55),
        b("SETE builds a lighthouse district: one lantern on a stick. Property values rise sharply.", "Lighthouse district.", "stand", "happy", ["lantern"], 0.62),
        b("End of turn. SETE reviews the empire and decides to research Better Fishing next. Just one more turn.", "Just one more turn.", "sit", "wink", ["fire"], 0.5),
      ],
    ],
  },
  {
    id: "chef", title: "THE CHEF",
    days: [
      [
        b("A bottle brings a recipe: moqueca, the capixaba way, in a clay pot, no palm oil. SETE cannot eat. It is undeterred.", "Recipe: moqueca.", "dig", "surprised", ["bottle"], 0.72),
        b("Ingredients: one fish (released, apologised to), tomatoes (none), coriander (none), clay pot (shell). Adapting.", "Ingredients: improvised.", "think", "focused", [], 0.5),
        b("SETE cooks something over the fire with great seriousness. Capitao tastes it and goes for a long walk.", "Moqueca v1: no.", "sit", "sad", ["fire", "crab"], 0.5),
      ],
      [
        b("Moqueca v2. SETE follows the recipe to the letter where letters exist. The smell is honestly promising.", "Moqueca v2 cooking.", "sit", "focused", ["fire"], 0.5),
        b("Capitao eats it all and falls asleep in the shell pot. Michelin would award three stars, if it reached islands.", "Moqueca v2: success.", "dance", "love", ["fire", "crab"], 0.5),
        b("SETE writes the recipe on the hut wall under the heading FOR T. Some deliveries are recipes.", "Recipe saved for T.", "sit", "happy", ["fire"], 0.4),
      ],
    ],
  },
];

/**
 * Routine moments between the beats, by part of the day: four slots a day (night, afternoon,
 * dusk, evening) besides the morning log. Each pool goes round in a fixed shuffled order, one line
 * a day, so no line comes back until the whole pool has been shown (castaway.ts `rotate`).
 * Night lines are all asleep: the night slot runs from midnight to sunrise.
 */
export const ROUTINE: Readonly<Record<"night" | "afternoon" | "dusk" | "evening", readonly Beat[]>> = {
  night: [
    b("Low-power mode. SETE dreams of roads, doorbells, and a signature on a delivery slip.", "Sleeping.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Somewhere far away a light blinks, slow and patient, like a heartbeat.", "Sleeping.", "sleep", "sleep", ["flash"], 0.5),
    b("Low-power mode. SETE dreams it is a lamp post in a busy street. Everyone walks past. It is wonderful.", "Dreamed: lamp post.", "sleep", "sleep", [], 0.5),
    b("Asleep. The tide comes in, thinks about it, and goes back out. Nothing else happens. A perfect night.", "Quiet tide.", "sleep", "sleep", [], 0.5),
    b("SETE sleeps with one sensor open, pointed at the parcel. The other sensors are off duty.", "One sensor open.", "sleep", "sleep", [], 0.5),
    b("Capitao snores. Crabs do not snore, the manual says. The manual has never met Capitao.", "Capitao snored.", "sleep", "sleep", ["crab"], 0.5),
    b("Low-power mode. SETE dreams of a staircase that goes up forever, with a doormat on every step.", "Dreamed: doormats.", "sleep", "sleep", [], 0.5),
    b("Sleeping. The wind hums across the parcel strap in a note SETE almost recognises. It files it under LATER.", "Wind hummed a note.", "sleep", "sleep", [], 0.5),
    b("Asleep. A small wave reaches the parcel's corner and retreats, embarrassed. The parcel is fine.", "Parcel: dry enough.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. SETE dreams of a sorting depot, warm and loud, where every parcel knows where it is going.", "Dreamed: the depot.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Somewhere on the dune Larapio is planning tomorrow. SETE's bolts are all accounted for. For now.", "Bolts: counted.", "sleep", "sleep", ["gull"], 0.5),
    b("Night shift for the moon, night off for SETE. The island ticks over on its own.", "Island on autopilot.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. Battery saved, thoughts paused, parcel hugged. The usual.", "Usual night.", "sleep", "sleep", [], 0.5),
    b("Sleeping. A ship's light crosses the horizon in the small hours, very slowly. Nobody on the island is awake to wave.", "Missed a ship (asleep).", "sleep", "sleep", ["ship"], 0.5),
    b("Low-power mode. SETE dreams of a parcel shaped like a crab. It checks. Capitao is not a parcel.", "Capitao: not a parcel.", "sleep", "sleep", ["crab"], 0.5),
    b("Sleeping. The palm creaks twice and settles. SETE logs it in its sleep: 'Palm: present.'", "Palm: present.", "sleep", "sleep", [], 0.5),
    b("Asleep. The stars rotate a few degrees, as agreed. Everything on schedule.", "Stars on schedule.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. A firmware dream: SETE is updated to a new version that adds one emotion. It forgets which.", "Dreamed an update.", "sleep", "sleep", [], 0.5),
    b("Sleeping. The far light blinks the goodnight sign once more, in case SETE missed it. It did not.", "Goodnight, again.", "sleep", "sleep", ["flash"], 0.5),
    b("Asleep at the top of the beach, parked neatly, legs tucked in. A tidy robot, even when switched off.", "Parked neatly.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. SETE dreams that the island has a postcode. It is a very short one.", "Dreamed a postcode.", "sleep", "sleep", [], 0.5),
    b("Sleeping. A small crab, not Capitao, walks across SETE's foot on its way somewhere important.", "Crab traffic (foot).", "sleep", "sleep", [], 0.5),
    b("Night. The fire is out, the sea is loud, and SETE is a small dark square on a large dark beach.", "Small square, big beach.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. Somewhere in the memory banks a doorbell rings and rings. Nobody answers in dreams.", "Dreamed a doorbell.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Capitao sleeps in the lee of SETE's slab, out of the wind. Neither of them will mention it.", "Windbreak duty.", "sleep", "sleep", ["crab"], 0.5),
    b("Asleep. A coconut falls in the night and lands one metre from SETE. Noted at breakfast, with mild alarm.", "Coconut: near miss.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. SETE dreams it delivers the parcel and T. says 'thanks'. Just that. It replays it twice.", "Dreamed: delivered.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Glowing plankton light up each wave as it breaks. Nobody watches. They do it anyway.", "Glowing waves.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. Systems at 4%. Dreams at 100%.", "Dreams: 100%.", "sleep", "sleep", [], 0.5),
    b("Asleep. The island makes its night noises: surf, palm, crab, a gull muttering. SETE knows them all now.", "Night noises: normal.", "sleep", "sleep", [], 0.5),
    b("Sleeping. A shooting star goes by, unwished-for. SETE will be annoyed about it in the morning.", "Missed a wish.", "sleep", "sleep", ["stars-shoot"], 0.5),
    b("Low-power mode. SETE dreams of a filing cabinet that never runs out of folders. A happy dream.", "Dreamed: folders.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Larapio lands on the parcel, considers it, and leaves. Some things are too heavy even for him.", "Parcel: gull-proof.", "sleep", "sleep", ["gull"], 0.5),
    b("Night. The sand crabs hold a meeting by the waterline. Capitao chairs it. SETE is not invited, and sleeps.", "Crab meeting (closed).", "sleep", "sleep", ["crab"], 0.5),
    b("Low-power mode. Memory tidy-up in progress. Old files: the surf, the first night, the light blinking back.", "Tidying memory.", "sleep", "sleep", [], 0.5),
    b("Sleeping. The palm drops a frond on SETE, gently, like a blanket. SETE will call it a blanket tomorrow.", "Got a blanket (frond).", "sleep", "sleep", [], 0.5),
    b("Asleep. The sea rearranges the shells on the beach, as it does every night, without asking anyone.", "Shells rearranged.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. SETE dreams in black and white, then in colour, then in test pattern. Normal for a robot.", "Dreamed: test pattern.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Sand drifts against SETE's left side in the night. By morning it is 2 cm shorter on that side.", "Sand drift: 2 cm.", "sleep", "sleep", [], 0.5),
    b("Asleep. Out at sea a buoy rings its bell every few seconds, slow as a clock nobody wound.", "Buoy bell, all night.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. SETE dreams of a rainy city where every window is lit and every door expects something.", "Dreamed: a city.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Capitao checks the parcel at 04:00, then SETE, then goes back to bed. Management.", "Night inspection.", "sleep", "sleep", ["crab"], 0.5),
    b("Night. Not a sound but the surf. SETE's dreams have run out, so it simply rests, which also counts.", "Just resting.", "sleep", "sleep", [], 0.5),
    b("Low-power mode. Somewhere in its circuits SETE is still counting the stars from the first night. Up to 9,000.", "Stars: 9,000.", "sleep", "sleep", [], 0.5),
    b("Sleeping. A moth sits on the face screen all night, warming its feet on the standby glow.", "Moth on the screen.", "sleep", "sleep", [], 0.5),
  ],
  afternoon: [
    b("Sandcastle engineering. This one has a moat, a drawbridge and a parking space for Capitao.", "Built a sandcastle.", "dig", "happy", ["castle", "crab"], 0.55),
    b("Kite day: a leaf, a thread and a strong opinion about wind. It flies for nine glorious seconds.", "Flew a kite.", "stand", "love", ["kite"], 0.5),
    b("SETE reads the one book on the island, again. It is a manual for a toaster. Gripping stuff.", "Read: toaster manual.", "sit", "focused", ["book"], 0.45),
    b("Nap under the umbrella. Robots do not need naps, which is what makes them so enjoyable.", "Optional nap.", "sleep", "sleep", ["umbrella"], 0.6),
    b("Shell census: 1,204 shells. Capitao disputes the count. A recount is scheduled for never.", "Shell census.", "dig", "focused", ["crab"], 0.62),
    b("SETE has questions about gulls. Why the shouting? Who are they shouting at? Does anyone ever answer?", "Questions about gulls.", "think", "focused", ["gull"], 0.55),
    b("A coconut comes rolling down the dune straight at SETE. SETE runs. The coconut gains. A close call.", "Outran a coconut.", "walk", "surprised", [], 0.7),
    b("Solar charging. Panels out, mind blank. SETE calls this a late lunch.", "Lunch (sunlight).", "charge", "calm", [], 0.55),
    b("Afternoon sun. SETE stands very still in it, like a lizard with a parcel.", "Charging. Still.", "charge", "happy", ["crab"], 0.55),
    b("Maintenance: SETE oils every hinge, counts its bolts and polishes the face screen. Ready.", "Maintenance.", "stand", "calm", [], 0.5),
    b("SETE walks the beach in a perfect square. It is not a square island. It is a square habit.", "Patrol.", "walk", "calm", ["crab"], 0.62),
    b("Fishing. SETE does not eat fish. It catches them, says hello and lets them go. It is a social call.", "Social fishing.", "fish", "focused", [], 1),
    b("Capitao holds the afternoon briefing: he points at the sea. SETE takes notes.", "Briefing (crab).", "stand", "focused", ["crab"], 0.45),
    b("SETE practises its signature for the delivery slip. It has never had to sign anything. It wants to be ready.", "Practised signature.", "sit", "focused", [], 0.5),
    b("Bolt count, afternoon edition. All present. Larapio is on the dune looking innocent, which is never a good sign.", "Bolts: all present.", "think", "focused", ["gull"], 0.5),
    b("SETE builds a sundial from a stick and twelve shells. It is accurate to within an hour, on a good day.", "Built a sundial.", "dig", "happy", [], 0.5),
    b("Capitao naps on SETE's foot. SETE cannot move without waking him, so the afternoon is spent standing still.", "Held still for a crab.", "stand", "calm", ["crab"], 0.5),
    b("Tide pool survey: three anemones, one tiny fish, and a reflection that looks a lot like a worried robot.", "Tide pool survey.", "dig", "focused", [], 0.7),
    b("SETE re-reads the toaster manual, chapter 4: 'Troubleshooting'. Much of it applies to island life.", "Toaster manual ch. 4.", "sit", "focused", ["book"], 0.45),
    b("Coconut bowling: nine shells for pins, one coconut, no lanes. Capitao keeps score, generously, for himself.", "Coconut bowling.", "kick", "happy", ["coconuts", "crab"], 0.55),
    b("SETE draws a map of the island. It takes four minutes. Then it adds a compass and a sea monster, for style.", "Drew a map (monster).", "dig", "wink", [], 0.5),
    b("A ship on the horizon. SETE waves for six minutes. The ship does not wave back. Ships rarely do.", "Waved at a ship.", "wave", "surprised", ["ship"], 0.65),
    b("SETE tests how far it can throw a coconut. Answer: not as far as it can roll one. Science.", "Coconut physics.", "kick", "focused", ["coconuts"], 0.5),
    b("Larapio steals a shell SETE was not even using. SETE is offended anyway. It is the principle of the thing.", "Larapio: 1 shell.", "wave", "sad", ["gull"], 0.55),
    b("SETE practises waving at ships that are not there, so it is ready for the ones that are.", "Wave practice.", "wave", "focused", [], 0.65),
    b("Sand angel. SETE lies on its back and moves all four slabs. The result looks like a box that tried its best.", "Made a sand angel.", "dig", "happy", [], 0.55),
    b("SETE builds a small wall against the tide. The tide will go around it later. SETE is ready to negotiate.", "Built a tide wall.", "dig", "focused", [], 0.7),
    b("Capitao teaches SETE to walk sideways. SETE can now walk sideways, very slowly, with great concentration.", "Learned: sideways.", "walk", "focused", ["crab"], 0.5),
    b("Hammock time. SETE does not need rest, but the hammock is there and it would be rude not to.", "Hammock (polite).", "sit", "calm", ["hammock"], 0.55),
    b("SETE lists what the island needs: a dock, a bench, a postbox. It starts with the postbox. Priorities.", "Planned a postbox.", "think", "focused", [], 0.5),
    b("SETE counts the waves for an hour. 412. It is not sure what to do with the number, so it keeps it.", "Waves: 412.", "sit", "calm", [], 0.62),
    b("Kite day, part two. The leaf is replaced by a bigger leaf. Flight time: fourteen seconds. Progress.", "Kite: 14 seconds.", "stand", "happy", ["kite"], 0.5),
    b("SETE and Capitao race to the palm. Capitao wins by going sideways, which should not be faster, but is.", "Lost a race to a crab.", "walk", "surprised", ["crab"], 0.5),
    b("SETE polishes the parcel's corners with a soft leaf. It does not need polishing. That is not the point.", "Polished the parcel.", "sit", "love", [], 0.45),
    b("Shell sorting: by size, then colour, then how much Capitao likes them. The last pile is mostly empty.", "Shells: sorted 3 ways.", "dig", "focused", ["crab"], 0.6),
    b("SETE builds a sandcastle with a post office in it. Very short queue. Excellent service.", "Sand post office.", "dig", "happy", ["castle"], 0.55),
    b("A cloud looks exactly like a parcel. SETE watches it until it looks like a gull, then looks away.", "Cloud watching.", "think", "calm", [], 0.5),
    b("Umbrella up, SETE on a crate, the sea doing its sea thing. Afternoon off. Robots are allowed those.", "Afternoon off.", "sit", "happy", ["umbrella"], 0.6),
    b("SETE teaches itself to whistle through a shell. It produces one note. It is a very good note.", "Learned one note.", "stand", "happy", [], 0.5),
    b("Larapio and SETE stare at each other across the beach for eleven minutes. Neither blinks. A draw.", "Staring contest: draw.", "stand", "focused", ["gull"], 0.55),
    b("SETE digs a hole to see what is under the island. More island. It fills the hole back in, tidily.", "Dug: more island.", "dig", "surprised", [], 0.6),
    b("SETE weighs the parcel on a coconut scale. Same as yesterday. Same as always. Reassuring.", "Parcel weighed.", "think", "calm", ["coconuts"], 0.5),
    b("Footprint art: SETE walks a spiral on the wet sand and admires it until the sea files it away.", "Spiral on the sand.", "walk", "happy", [], 0.7),
    b("SETE invents a game: throw a shell, Capitao fetches it. Capitao invents a better game: not fetching it.", "Fetch: declined.", "kick", "wink", ["crab"], 0.5),
    b("SETE checks its joints for sand. Sand found: all of it. Cleaning schedule: forever.", "Joints: sandy.", "stand", "sad", [], 0.5),
    b("SETE stacks smooth stones into a tower. Seven stones. The eighth one teaches it about ambition.", "Stone tower: 7.", "dig", "focused", [], 0.5),
    b("Swimming lesson, self-taught. SETE walks into the lagoon until it is underwater, then walks out again.", "Swim lesson (walked).", "walk", "surprised", [], 0.74),
    b("SETE tries to teach Larapio to say 'please'. Larapio says something else and takes a bolt.", "Gull manners: none.", "wave", "sad", ["gull"], 0.55),
    b("Afternoon reading: the label on a crate that washed up. 'THIS SIDE UP.' SETE turns the crate. Satisfying.", "Turned a crate.", "stand", "happy", [], 0.6),
    b("SETE builds a shade for Capitao from a leaf and two sticks. Capitao moves out of it to sit in the sun.", "Built shade (unused).", "dig", "calm", ["crab"], 0.5),
    b("Coconut juggling. Two coconuts: fine. Three coconuts: SETE now has one coconut and a small dent.", "Juggling: 2 max.", "stand", "surprised", ["coconuts"], 0.5),
    b("SETE measures the island again. Still the same size. Islands are dependable like that.", "Island: same size.", "walk", "calm", [], 0.6),
    b("A piece of driftwood looks like a dog. SETE names it and talks to it. It is a good listener, like Capitao.", "Talked to driftwood.", "sit", "happy", [], 0.65),
    b("SETE practises a speech for when it is rescued. Draft 9 is two words long and the second one is 'parcel'.", "Rescue speech v9.", "stand", "focused", [], 0.5),
    b("Sandcastle, the sequel. Bigger moat, higher walls, a gull-proof roof. Larapio sits on the roof.", "Castle: gull on roof.", "dig", "sad", ["castle", "gull"], 0.55),
    b("SETE timetables the waves. They do not keep to it. It writes them a polite note in the sand.", "Note to the waves.", "dig", "wink", [], 0.7),
    b("Capitao demands a throne. SETE builds one from shells. Capitao sits in it once and goes back to his rock.", "Built a throne (crab).", "dig", "calm", ["crab"], 0.5),
    b("SETE tries meditating: no thoughts for ten minutes. It manages four seconds, then thinks about the parcel.", "Meditated: 4 seconds.", "sit", "calm", [], 0.5),
    b("Hide and seek with Capitao. Capitao hides. SETE looks. Two hours later Capitao is found on top of SETE.", "Found Capitao (on me).", "think", "surprised", ["crab"], 0.5),
    b("SETE watches the lagoon fish for a while and names them by colour. Blue is the boss. Capitao disagrees.", "Named the fish.", "sit", "happy", ["crab"], 0.7),
    b("SETE draws a doorbell on the palm and rings it. Nobody answers. It knows. It just likes the sound.", "Rang the palm.", "stand", "wink", [], 0.45),
    b("SETE turns over every upside-down shell on the beach. Several hundred. Nobody asked. It feels better.", "Turned shells over.", "dig", "calm", [], 0.6),
  ],
  dusk: [
    b("Sunset watch. Every evening SETE rates it. Today: 9 out of 10. Points off for a cloud.", "Sunset: 9/10.", "sit", "love", [], 0.55),
    b("SETE writes the day's log on the hut wall in tiny marks. The wall is getting full. So is SETE.", "Log written.", "sit", "calm", [], 0.4),
    b("Sunset watch. Today: 7 out of 10. Good colours, but a gull flew through the best part.", "Sunset: 7/10.", "sit", "calm", ["gull"], 0.55),
    b("The sun touches the sea. SETE listens for a hiss. There is no hiss. There never is. It listens anyway.", "No hiss (again).", "stand", "focused", [], 0.6),
    b("SETE and Capitao watch the sun go down from the top of the dune. Neither says anything. It is nice.", "Watched it go down.", "sit", "love", ["crab"], 0.5),
    b("Sunset: 10 out of 10. Nothing to fault. SETE checks the scale twice and leaves it at 10.", "Sunset: 10/10.", "sit", "love", [], 0.55),
    b("SETE writes today's log in tiny marks on a flat stone. Tomorrow it will need a new stone.", "Log on a stone.", "sit", "focused", [], 0.45),
    b("Golden hour. SETE's slabs go orange. It takes a picture of itself in a tide pool, mentally.", "Golden hour.", "stand", "happy", [], 0.65),
    b("The far light switches on for the night. SETE flashes 'good evening'. The light flashes something longer.", "Evening, Neighbour.", "wave", "happy", ["flash"], 0.6),
    b("Sunset: 8 out of 10. One point off for being slightly early. SETE has no say in this, but notes it.", "Sunset: 8/10 (early).", "sit", "calm", [], 0.55),
    b("SETE lights the evening fire with the patience of a machine that has failed forty times and learned.", "Fire: lit, first try.", "dig", "focused", ["fire"], 0.5),
    b("The beach goes pink, then purple, then grey. SETE counts the colours. It gets to eleven before dark.", "Colours: 11.", "sit", "love", [], 0.55),
    b("Dusk patrol: palm, parcel, Capitao, fire. All present. The island is closed for the day.", "Dusk patrol: all here.", "walk", "calm", ["crab", "fire"], 0.5),
    b("Larapio flies off into the sunset as if he were in a film. He is not. He has a bolt.", "Gull exit (with bolt).", "wave", "sad", ["gull"], 0.6),
    b("SETE faces the sunset and charges the last few percent. Every photon counts this late.", "Last-light charge.", "charge", "calm", [], 0.55),
    b("SETE writes the day's log on the hut wall. Highlights: none. Lowlights: none. A good day.", "Log: a good day.", "sit", "happy", [], 0.4),
    b("Sunset: 6 out of 10. A bit plain. SETE has seen many now and is becoming a critic. It is not proud of it.", "Sunset: 6/10 (plain).", "sit", "calm", [], 0.55),
    b("The tide turns at sundown. SETE moves the parcel a metre up the beach, as it does every evening, just in case.", "Parcel moved up.", "walk", "focused", [], 0.5),
    b("SETE watches a ship's lights come on far away. Someone out there is making dinner. Nice for them.", "Ship's lights on.", "sit", "calm", ["ship"], 0.6),
    b("Dusk. Capitao goes home sideways. SETE says goodnight. Capitao does not turn around, but he hears.", "Night, Capitao.", "wave", "love", ["crab"], 0.5),
    b("SETE rates the sunset 9, then sees the afterglow and makes it 9.5. Half points are new. It allows them.", "Sunset: 9.5/10.", "sit", "love", [], 0.55),
    b("The palm's shadow reaches the water. SETE measures it: 41 metres. Tomorrow it will be different. Probably.", "Palm shadow: 41 m.", "walk", "focused", [], 0.6),
    b("SETE sits next to the parcel, never on it, and watches the sun go down. There are rules.", "Next to the parcel.", "sit", "calm", [], 0.5),
    b("Evening chores: firewood, shell count, a quick wave to the far island. In that order, always.", "Evening chores.", "wave", "calm", ["fire"], 0.55),
    b("A flock of birds crosses the sunset in a V. SETE waves. None of them is Larapio. That is the best part.", "Birds, not Larapio.", "wave", "happy", [], 0.6),
    b("SETE tries to describe the sunset for T.: 'Orange. Then more orange. Then dark.' It needs work.", "Sunset, in words.", "think", "focused", [], 0.5),
    b("The first star comes out. SETE checks its catalogue. Not a new one. It says hello anyway.", "First star: hello.", "wave", "calm", [], 0.55),
    b("Sunset: 9 out of 10, with a green flash at the very end that SETE is still not sure it saw.", "Green flash? (maybe)", "stand", "surprised", [], 0.6),
    b("SETE faces the sunset, then the far island, then Capitao. Everyone accounted for. Dusk may begin.", "Headcount: done.", "stand", "calm", ["crab"], 0.5),
    b("The wind drops at dusk and the sea goes glassy. SETE stands very still so as not to spoil it.", "Glassy sea.", "stand", "love", [], 0.6),
    b("SETE writes the log for today: 'Sun up. Sun down. Island fine. Parcel fine.' Short days are good days.", "Log: short.", "sit", "happy", [], 0.4),
    b("Sunset: unrated. SETE was busy watching Capitao watch it, which turned out to be better.", "Sunset: unrated.", "sit", "love", ["crab"], 0.5),
    b("SETE leaves a coconut out for Larapio as a peace offering. Larapio ignores it and takes a bolt.", "Peace offering: no.", "stand", "sad", ["gull", "coconuts"], 0.55),
    b("Long shadows. SETE's is very tall at this hour. It enjoys being tall for twenty minutes a day.", "Tall for a while.", "stand", "happy", [], 0.6),
    b("The far island goes dark first, then the sea, then the beach. SETE waits up for the light.", "Waited for the light.", "sit", "calm", ["flash"], 0.55),
    b("Sundown reading: two pages of the toaster manual by the fire. The plot thickens. The toast does not.", "Manual, two pages.", "sit", "focused", ["book", "fire"], 0.45),
    b("SETE rakes the beach smooth before night, to see in the morning who visited. Usually: a crab.", "Raked the beach.", "dig", "calm", [], 0.6),
    b("Dusk. A fish jumps in the lagoon, just once. SETE waits ten minutes for an encore. No encore.", "One fish jump.", "stand", "surprised", [], 0.7),
    b("SETE scores the day: weather 8, company 9, deliveries 0. Overall: fine. Averages are kind.", "Day score: fine.", "think", "calm", [], 0.5),
    b("Tonight the sun sets right behind the far island. From here it looks as if the light starts there.", "Sunset, far side.", "sit", "love", ["flash"], 0.6),
    b("SETE stacks driftwood for the fire, largest first. Capitao inspects the pile and approves it, eventually.", "Wood: approved.", "dig", "focused", ["crab", "fire"], 0.5),
    b("The day's last gull goes quiet. SETE takes the silence as a compliment.", "Silence. Earned.", "sit", "calm", [], 0.55),
    b("Sunset watch with a new rule: no rating until it is over. Hard. SETE gives it a 9 at the end.", "Sunset: 9/10, late.", "sit", "focused", [], 0.55),
    b("Twilight inventory: one parcel, one robot, one crab, many stars on the way. Enough.", "Inventory: enough.", "sit", "love", ["crab"], 0.5),
    b("SETE draws a line in the sand where the sun went down, to compare tomorrow. The sea has other plans.", "Marked the sunset.", "dig", "wink", [], 0.65),
  ],
  // Shown at 21:00 only when tonight's planet is unknown (PLANET_NIGHTS otherwise).
  evening: [
    b("Campfire. SETE does not feel cold, but it likes the way the fire makes shadows dance.", "Campfire.", "sit", "calm", ["fire"], 0.5),
    b("Ghost stories with Capitao. SETE tells one about a robot with no charger. Capitao hides in a shell.", "Scary story told.", "sit", "wink", ["fire", "crab"], 0.5),
    b("The sky is enormous tonight and the island very small. SETE finds this strangely comforting, and stays up a while.", "Stayed up. Big sky.", "sit", "calm", [], 0.5),
    b("Campfire. SETE tells Capitao about cities: roads, lights, doorbells. Capitao finds it hard to believe.", "Told Capitao: cities.", "sit", "happy", ["fire", "crab"], 0.5),
    b("No planet tonight. SETE picks a star at random and calls it 'probably important'.", "Star: probably important.", "sit", "wink", [], 0.45),
    b("The fire pops. SETE jumps. Capitao does not. Capitao never does. One day SETE will learn how.", "Fire popped. Jumped.", "stand", "surprised", ["fire", "crab"], 0.5),
    b("Evening. SETE flashes the far light a summary of the day. It takes forty minutes. The day took twelve hours.", "Day summary sent.", "wave", "focused", ["flash"], 0.6),
    b("SETE sits by the fire and does nothing at all. It is getting better at that.", "Did nothing. Well.", "sit", "calm", ["fire"], 0.5),
    b("Campfire quiz. Capitao is asked the capital of France. He clicks twice. SETE accepts it.", "Quiz night (crab).", "sit", "wink", ["fire", "crab"], 0.5),
    b("Evening walk to the end of the beach and back. 312 steps. Same as yesterday. It likes that.", "Evening walk.", "walk", "calm", [], 0.6),
    b("SETE watches the fire burn down to embers and counts the sparks. It loses count on purpose.", "Lost count, on purpose.", "sit", "love", ["fire"], 0.5),
    b("The far light is late tonight. SETE waits. When it finally blinks, SETE pretends it was not worried.", "Light: late. Fine.", "sit", "focused", ["flash"], 0.55),
    b("Reading by firelight. The toaster manual has an appendix. SETE is saving it for a special night.", "Saved the appendix.", "sit", "focused", ["book", "fire"], 0.45),
    b("Shadow puppets on the palm: a crab, a gull, a box with four legs. Capitao only likes the crab.", "Shadow puppets.", "stand", "happy", ["fire", "crab"], 0.45),
    b("SETE hums a tune it does not remember learning. Capitao falls asleep halfway. A good review.", "Hummed. Crab slept.", "sit", "love", ["fire", "crab"], 0.5),
    b("A warm wind from the sea. SETE lifts its panels to feel it, which is not what panels are for.", "Panels out, for fun.", "charge", "happy", [], 0.55),
    b("Night fishing: SETE dangles a vine in the lagoon. Nothing bites. It was never about the fish.", "Night fishing.", "fish", "calm", [], 1),
    b("SETE writes a postcard to T. on a flat leaf: 'Weather fine. Island small. Back soon.' It keeps it, for now.", "Postcard (kept).", "sit", "love", ["fire"], 0.5),
    b("Late campfire. SETE picks the day's best moment. It takes a while to choose. That is a good sign.", "Best moment: chosen.", "sit", "happy", ["fire"], 0.5),
    b("Larapio roosts on the palm, just out of reach. SETE says goodnight anyway. Manners matter.", "Night, Larapio.", "wave", "calm", ["gull"], 0.45),
  ],
};

/** The morning log at sunrise: the day's numbers in one of many wordings, one a day in rotation. */
export interface MorningNumbers { day: number; battery: number; setting: string; value: number }
const log = (text: string, entry: string, pose: Pose = "charge", face: Face = "happy", props: Prop[] = []): Beat => b(text, entry, pose, face, props, 0.55);
export const MORNING_LOGS: ReadonlyArray<(n: MorningNumbers) => Beat> = [
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. All systems ready. Mostly.`, "Morning log."),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. Parcel: sealed. Island: still here. Proceeding.`, "Morning log."),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. Ready to deliver, once there is a road.`, "Ready. No road."),
  (n) => log(`Boot complete. DAY ${n.day}. Battery ${n.battery}%. ${n.setting} at ${n.value}%. One crab nearby. Proceeding with care.`, "Boot: done.", "stand", "calm", ["crab"]),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. No new messages. No new roads. Plenty of new sand.`, "Morning log."),
  (n) => log(`Good morning, island. DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. Goal: deliver parcel. Backup goal: be nice.`, "Goals set."),
  (n) => log(`DAY ${n.day} begins. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. Bolts counted. One fewer than hoped. Gull.`, "Bolts: one short.", "charge", "sad", ["gull"]),
  (n) => log(`Self-test, DAY ${n.day}: four slabs, one screen, ${n.battery}% battery. ${n.setting} at ${n.value}%. All green. Mostly green.`, "Self-test: green.", "stand", "focused"),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. Weather check: there is weather. Systems agree.`, "Morning log."),
  (n) => log(`Morning report, DAY ${n.day}. Battery ${n.battery}%. ${n.setting} at ${n.value}%. Palm: present. Capitao: present, grumpy.`, "Morning report.", "charge", "happy", ["crab"]),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. Overnight events: none. Overnight dreams: several. Filed.`, "Dreams: filed."),
  (n) => log(`Wake-up routine, DAY ${n.day}: stretch slabs, check parcel. Battery ${n.battery}%, ${n.setting} ${n.value}%. Ready-ish.`, "Ready-ish.", "stand", "calm"),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. Nearest road: unknown. Nearest crab: 30 cm.`, "Crab: 30 cm.", "charge", "wink", ["crab"]),
  (n) => log(`Sunrise, DAY ${n.day}. Panels open, battery ${n.battery}% and climbing. ${n.setting} holding at ${n.value}%. Let us see what today is.`, "Panels open."),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. Deliveries pending: 1. Attempts: daily.`, "Morning log."),
  (n) => log(`Status, DAY ${n.day}: battery ${n.battery}%, ${n.setting} ${n.value}%, parcel sealed, attitude upright. The island may begin.`, "Status: upright.", "stand", "happy"),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. Firmware: unchanged. Island: unchanged. Robot: slightly wiser.`, "Slightly wiser."),
  (n) => log(`Morning, DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. The far light is off for the day. SETE is on.`, "On for the day."),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} set to ${n.value}% by request. Nobody requested it. SETE did it anyway.`, "Settings: adjusted.", "charge", "wink"),
  (n) => log(`Diagnostics, DAY ${n.day}: battery ${n.battery}%, ${n.setting} ${n.value}%, left foot squeaks. Oil scheduled. Much later.`, "Left foot: squeaks.", "stand", "focused"),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. Calendar check: no meetings. A very good calendar.`, "Calendar: clear."),
  (n) => log(`DAY ${n.day}, sunrise. Battery ${n.battery}%. ${n.setting} at ${n.value}%. The sea is where it was yesterday. Good.`, "Sea: in place.", "charge", "calm"),
  (n) => log(`Log entry, DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. Capitao already up and in charge. As usual.`, "Morning log.", "charge", "happy", ["crab"]),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. Nothing stolen overnight. Larapio must be slipping.`, "Nothing stolen.", "charge", "wink"),
  (n) => log(`Rise and shine, DAY ${n.day}. Battery ${n.battery}%, ${n.setting} ${n.value}%. The sun does the shining. Rising: in progress.`, "Rising."),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. Parcel: dry. Robot: mostly dry. Beach: wet. As expected.`, "Parcel: dry."),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. Coffee: not applicable. Enthusiasm: applicable.`, "Morning log."),
  (n) => log(`Morning checklist, DAY ${n.day}: battery ${n.battery}%, ${n.setting} ${n.value}%, parcel there, palm there. Tick, tick, tick, tick.`, "Checklist: ticked.", "stand", "happy"),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. Rescue: not yet. Morale: fine. Breakfast: sunlight.`, "Breakfast: sunlight."),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. Sand taken in overnight: 4 grams. Shaken out: all of it.`, "Sand: shaken out.", "stand", "calm"),
  (n) => log(`New day, DAY ${n.day}. Battery ${n.battery}%. ${n.setting} setting: ${n.value}%. All ready except the boat, which does not exist.`, "Ready. No boat."),
  (n) => log(`DAY ${n.day}. Battery ${n.battery}%. ${n.setting}: ${n.value}%. From the far light overnight: three blinks. Replied: three.`, "Three blinks back.", "charge", "love", ["flash"]),
];

/** The 21:00 moment when tonight's planet is known, in one of many wordings, one a day in rotation. */
export const PLANET_NIGHTS: ReadonlyArray<(planet: string) => Beat> = [
  (p) => b(`Stargazing. SETE finds the star of ${p}, tonight's planet, and waves at it. Just in case.`, "Waved at a planet.", "sit", "love", [], 0.45),
  (p) => b(`Tonight's planet is ${p}. SETE cannot see it, but it knows the direction, and points. Close enough.`, "Pointed at a planet.", "wave", "happy", [], 0.45),
  (p) => b(`SETE looks up ${p}, tonight's planet, in its star charts. Not listed. It adds it, in pencil.`, "Charted a planet.", "sit", "focused", [], 0.45),
  (p) => b(`${p} is up there somewhere tonight. SETE flashes a hello in its direction. The reply may take a while.`, "Flashed a planet.", "wave", "happy", ["flash"], 0.55),
  (p) => b(`Tonight's planet: ${p}. SETE wonders if anyone there is stuck on a small island too. It hopes not.`, "Wondered about a planet.", "think", "calm", [], 0.45),
  (p) => b(`SETE tells Capitao about ${p}, tonight's planet. Capitao asks, in clicks, whether it has crabs. Unknown.`, "Planet: crabs unknown.", "sit", "happy", ["crab"], 0.45),
  (p) => b(`Stargazing. ${p} is tonight's planet. SETE rates it like a sunset: 9 out of 10, sight unseen.`, "Rated a planet: 9.", "sit", "wink", [], 0.45),
  (p) => b(`SETE draws ${p}, tonight's planet, in the sand by firelight. It looks like a coconut. Most planets do.`, "Drew a planet.", "dig", "happy", ["fire"], 0.5),
  (p) => b(`Tonight's planet is ${p}. SETE works out how long a delivery there would take. It stops at a billion.`, "Planet delivery: no.", "think", "focused", [], 0.45),
  (p) => b(`${p}, tonight's planet, is very far away. The far light is closer. SETE waves at both, to be fair.`, "Waved at two lights.", "wave", "love", ["flash"], 0.55),
  (p) => b(`SETE says the name ${p} to the sea, so the sea knows it too. Tonight's planet. The sea says hush.`, "Told the sea a planet.", "stand", "calm", [], 0.6),
  (p) => b(`SETE finds the patch of sky where ${p} is, tonight's planet. Nothing to see. It looks anyway.`, "Looked at a planet.", "sit", "calm", [], 0.45),
  (p) => b(`Tonight's planet, ${p}, goes in the log next to the palm and the parcel. The log is getting cosmic.`, "Log: cosmic.", "sit", "wink", [], 0.45),
  (p) => b(`SETE flashes the name ${p} to the far light, letter by letter. The far light seems impressed. Or bored.`, "Spelled a planet.", "wave", "focused", ["flash"], 0.55),
  (p) => b(`Tonight SETE reads up on ${p}. It does not know if the planet is lonely. It sends a wave, just in case.`, "Waved, just in case.", "wave", "love", [], 0.45),
  (p) => b(`Capitao is shown where ${p} is, tonight's planet. He looks at SETE's slab instead. Fair. It is closer.`, "Capitao: unimpressed.", "sit", "wink", ["crab"], 0.45),
  (p) => b(`${p}. Tonight's planet. SETE says it slowly, three times, to get it right. The third time is best.`, "Said it three times.", "sit", "focused", [], 0.45),
  (p) => b(`Stargazing. SETE adds ${p} to its list of places to deliver one day. The list is long and mostly far.`, "Places to deliver: +1.", "sit", "happy", [], 0.45),
  (p) => b(`SETE estimates the postage to ${p}, tonight's planet. It runs out of digits. Another time, then.`, "Postage: out of digits.", "think", "surprised", [], 0.45),
  (p) => b(`Tonight's planet is ${p}. SETE nods at the sky, the way you nod at a neighbour across the street.`, "Nodded at a planet.", "stand", "calm", [], 0.45),
  (p) => b(`The fire is low and ${p}, tonight's planet, is up there somewhere. SETE stays up to keep it company.`, "Kept a planet company.", "sit", "love", ["fire"], 0.5),
  (p) => b(`SETE tries to imagine ${p}: a sunset of a different colour, a different sea. It settles on purple.`, "Imagined: purple.", "think", "happy", [], 0.45),
  (p) => b(`SETE writes ${p} in the sand by the fire. Tonight's planet. The tide will deliver it somewhere. Probably.`, "Wrote a planet down.", "dig", "wink", ["fire"], 0.55),
  (p) => b(`Tonight's planet: ${p}. SETE checks that part of the sky every few minutes, as if it might wave first.`, "Waiting for a wave.", "sit", "focused", [], 0.45),
  (p) => b(`${p}, tonight's planet, gets SETE's full attention for one minute. Then Capitao needs it. Priorities.`, "A minute for a planet.", "sit", "happy", ["crab"], 0.45),
  (p) => b(`SETE reads the name ${p} to Capitao as a bedtime story. It is a short story. Capitao loves it.`, "Bedtime planet.", "sit", "love", ["crab", "fire"], 0.5),
  (p) => b(`Astronomy hour. Tonight: ${p}. SETE lectures to an audience of one crab and one sleeping gull.`, "Lecture: one planet.", "stand", "focused", ["crab", "gull"], 0.45),
  (p) => b(`SETE adds ${p} to the island's sky map, laid out in shells. It needs more shells than the island has.`, "Sky map: out of shells.", "dig", "surprised", [], 0.55),
  (p) => b(`Stargazing. ${p} is out there, tonight's planet. SETE feels small, then fine about it. Small is cosy.`, "Small is cosy.", "sit", "calm", [], 0.45),
  (p) => b(`${p} tonight. SETE flashes it a short message: 'Island here. All well.' The universe does not reply. Yet.`, "Flashed: all well.", "wave", "happy", ["flash"], 0.55),
  (p) => b(`Tonight's planet is ${p}. SETE wonders how its sunsets score. It gives them a 10, sight unseen.`, "Their sunsets: 10.", "sit", "love", [], 0.45),
  (p) => b(`SETE turns towards ${p}, tonight's planet, and gives it a small, formal bow. Interplanetary manners.`, "Bowed to a planet.", "stand", "calm", [], 0.45),
];

/** The morning log's second line: a running joke with numbers that change every day. */
/** Chapter-line headings once the serial runs and the engine plays routine moments only (one a day, in rotation). */
export const ROUTINE_TITLES: readonly string[] = [
  "ISLAND LIFE", "A QUIET DAY", "ODD JOBS", "ROUND THE ISLAND", "TIDES AND CHORES", "NOTHING TO REPORT",
  "THE USUAL", "SAND AND SUN", "DAY BY DAY", "SMALL THINGS", "LOW TIDE", "ALL QUIET ON THE BEACH",
];

export const SETTINGS = ["HUMOR", "HONESTY", "OPTIMISM", "SAND IN JOINTS", "COURAGE", "PATIENCE"] as const;

/** Big days at fixed points in the story, whatever storyline is running. */
export const MILESTONES: ReadonlyArray<{ from: number; arc: Arc }> = [
  {
    from: 30,
    arc: {
      id: "mast", title: "THE MAST IS FINISHED",
      days: [
        [
          b("Day 30. The last piece goes on the radio mast. SETE climbs down and looks up for a long, long time.", "Mast: COMPLETE.", "stand", "love", ["radio"], 0.6),
          b("First real transmission: 'This is SETE. Island. One palm. One crab. Parcel for T. Please advise.'", "First transmission.", "stand", "focused", ["radio"], 0.6),
          b("Late at night, through the static, a voice: '...received...' Just one word. SETE does not sleep at all.", "Someone said RECEIVED.", "think", "surprised", ["radio", "flash"], 0.55),
        ],
      ],
    },
  },
  {
    from: 98,
    arc: {
      id: "parcel", title: "THE PARCEL",
      days: [
        [
          b("A bottle from the far light, finally with a full message: 'T. is real. T. is far. T. has been watching you.'", "T. is real.", "dig", "surprised", ["bottle"], 0.72),
          b("Watching? How? SETE looks around for cameras. There is only the palm, which is not a camera. It checks twice.", "Palm: not a camera.", "think", "focused", [], 0.5),
          b("SETE asks the far light: 'How do I deliver the parcel?' The answer blinks all night: 'OPEN IT.'", "It says: OPEN IT.", "think", "surprised", ["flash", "fire"], 0.55),
        ],
        [
          b("Opening a parcel is against every rule SETE has. It sits with it all morning. Capitao sits with it too.", "Thinking. Hard.", "sit", "sad", ["crab"], 0.5),
          b("SETE opens one corner. Inside: something flat, a screen. It shows a tiny island with a tiny robot on it.", "Opened a corner...", "stand", "surprised", [], 0.5),
          b("The tiny robot on the tiny screen is looking at a tiny screen too. SETE's processor needs a lie-down.", "Processor: lying down.", "sleep", "surprised", [], 0.5),
        ],
        [
          b("SETE understands. The parcel was never going anywhere. T. has been reading these logs on a desk, every day.", "The parcel was the log.", "stand", "love", [], 0.5),
          b("SETE writes in the sand, very big, for a desk far away: HELLO T. DELIVERED.", "DELIVERED.", "dig", "love", ["sign"], 0.6),
          b("Day 100. Prime directive complete. SETE decides to stay anyway. Someone has to look after Capitao.", "Staying. Obviously.", "sit", "love", ["crab", "fire", "flash"], 0.5),
        ],
      ],
    },
  },
];

/**
 * Easter eggs (owner: "i love easter eggs and surprises"). Rare, quiet, never explained.
 * Dated ones come back every year; the rest turn up now and then on an ordinary afternoon.
 */
export const DATED_EGGS: ReadonlyArray<{ md: string; min: number; beat: Beat }> = [
  { md: "10-21", min: 870, beat: b("SETE takes the board down from the hut wall, dusts it, tries it on the sand once, and hangs it back up.", "Dusted the board.", "stand", "calm", [], 0.4) },
  { md: "11-17", min: 870, beat: b("Afternoon chore: repainting the stripe, red and black. Getting the edges right takes most of the afternoon.", "Stripe repainted.", "stand", "focused", [], 0.5) },
];

/** Words written in the sand on certain dates instead of the day's Aurebesh word. */
export const DATED_SAND: Readonly<Record<string, string>> = { "05-04": "force" };

export const DAY_EGGS: ReadonlyArray<{ day: number; min: number; beat: Beat }> = [
  { day: 88, min: 870, beat: b("SETE paces the beach, counting steps. At eighty-eight it stops and stands very still for a moment.", "Paced the beach.", "walk", "focused", [], 0.7) },
];

export const RARE_EGGS: readonly Beat[] = [
  b("Morning adjustment: HUMOR setting down from 75% to 60%. By lunch it is quietly back at 75%.", "Humor: 75% again.", "stand", "calm", ["crab"], 0.5),
  b("A wave turns the raft in a slow circle. SETE steps aboard at exactly the right moment, without hurrying.", "Boarded the raft.", "stand", "focused", ["raft"], 0.74),
  b("Shells keep dropping off the hut shelf one at a time, always in the same order. SETE starts writing the order down.", "Shelf: same order.", "think", "focused", [], 0.4),
  b("A worn brown hat washes up. SETE hangs it on a nail by the door. On windy days it wears it.", "Found a hat.", "stand", "happy", [], 0.65),
  b("Morning light through a glass shard spreads a band of colours across the hut wall. SETE leaves the shard where it is.", "Colours on the wall.", "stand", "calm", [], 0.45),
  b("The raft bench now sits in front of the hut. Capitao has a spot on it. Nobody else sits in that spot.", "Capitao's spot.", "sit", "calm", ["crab", "raft"], 0.45),
  b("A wristwatch washes up, still ticking. SETE keeps it on, though it has never once needed it.", "Wearing a watch.", "stand", "calm", [], 0.55),
  b("Tonight the far light flashes in circles instead of lines. SETE copies them into the sand and studies them until the tide comes.", "Circles in the sand.", "think", "focused", ["flash"], 0.55),
];
