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

/** Routine moments between the beats, by part of the day. Picked by the clock, never repeated in a row. */
export const ROUTINE = {
  morning: [
    b("Morning maintenance: SETE oils every hinge, counts its bolts and polishes the face screen. Ready.", "Morning maintenance.", "stand", "calm", [], 0.5),
    b("SETE walks the beach in a perfect square. It is not a square island. It is a square habit.", "Morning patrol.", "walk", "calm", ["crab"], 0.62),
    b("Fishing. SETE does not eat fish. It catches them, says hello and lets them go. It is a social call.", "Social fishing.", "fish", "focused", [], 1),
    b("Capitao holds the morning briefing: he points at the sea. SETE takes notes.", "Briefing (crab).", "stand", "focused", ["crab"], 0.45),
  ],
  noon: [
    b("Solar charging. Panels out, mind blank. SETE calls this 'lunch'.", "Lunch (sunlight).", "charge", "calm", [], 0.55),
    b("Midday. SETE stands very still in the sun like a lizard with a parcel.", "Charging. Still.", "charge", "happy", ["crab"], 0.55),
  ],
  afternoon: [
    b("Sandcastle engineering. This one has a moat, a drawbridge and a parking space for Capitao.", "Built a sandcastle.", "dig", "happy", ["castle", "crab"], 0.55),
    b("Kite day: a leaf, a thread and a strong opinion about wind. It flies for nine glorious seconds.", "Flew a kite.", "stand", "love", ["kite"], 0.5),
    b("SETE reads the one book on the island, again. It is a manual for a toaster. Gripping stuff.", "Read: toaster manual.", "sit", "focused", ["book"], 0.45),
    b("Nap under the umbrella. Robots do not need naps, which is what makes them so enjoyable.", "Optional nap.", "sleep", "sleep", ["umbrella"], 0.6),
    b("Shell census: 1,204 shells. Capitao disputes the count. A recount is scheduled for never.", "Shell census.", "dig", "focused", ["crab"], 0.62),
    b("SETE has questions about gulls. Why the shouting? Who are they shouting at? Does anyone ever answer?", "Questions about gulls.", "think", "focused", ["gull"], 0.55),
    b("A coconut comes rolling down the dune straight at SETE. SETE runs. The coconut gains. A close call.", "Outran a coconut.", "walk", "surprised", [], 0.7),
  ],
  dusk: [
    b("Sunset watch. Every evening SETE rates it. Today: 9 out of 10. Points off for a cloud.", "Sunset: 9/10.", "sit", "love", [], 0.55),
    b("SETE writes the day's log on the hut wall in tiny marks. The wall is getting full. So is SETE.", "Log written.", "sit", "calm", [], 0.4),
  ],
  evening: [
    b("Campfire. SETE does not feel cold, but it likes the way the fire makes shadows dance.", "Campfire.", "sit", "calm", ["fire"], 0.5),
    b("Ghost stories with Capitao. SETE tells one about a robot with no charger. Capitao hides in a shell.", "Scary story told.", "sit", "wink", ["fire", "crab"], 0.5),
  ],
  night: [
    b("The sky is enormous tonight and the island very small. SETE finds this strangely comforting, and stays up a while.", "Stayed up. Big sky.", "sit", "calm", [], 0.5),
    b("Low-power mode. SETE dreams of roads, doorbells, and a signature on a delivery slip.", "Sleeping.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Somewhere far away a light blinks, slow and patient, like a heartbeat.", "Sleeping.", "sleep", "sleep", ["flash"], 0.5),
  ],
} as const;

/** The morning log's second line: a running joke with numbers that change every day. */
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
