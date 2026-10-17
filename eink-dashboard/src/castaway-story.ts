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
        b("Late night, the radio picks up music from somewhere. SETE and Capitao sway. Mostly SETE.", "Radio dance night.", "dance", "love", ["radio", "crab", "fire"], 0.5),
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
    b("Low-power mode. SETE dreams of roads, doorbells, and a signature on a delivery slip.", "Sleeping.", "sleep", "sleep", [], 0.5),
    b("Sleeping. Somewhere far away a light blinks, slow and patient, like a heartbeat.", "Sleeping.", "sleep", "sleep", ["flash"], 0.5),
  ],
} as const;

/** The morning log's second line: a running joke with numbers that change every day. */
export const SETTINGS = ["HUMOR", "HONESTY", "OPTIMISM", "SAND IN JOINTS", "COURAGE", "PATIENCE"] as const;
