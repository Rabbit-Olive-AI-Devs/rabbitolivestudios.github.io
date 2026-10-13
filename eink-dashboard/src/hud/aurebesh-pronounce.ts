/**
 * How to say each word on the Aurebesh page.
 *
 * Aurebesh has no phonetics of its own: it is a different set of shapes for the
 * Latin alphabet, and the words are plain English ("Basic"). So this is an
 * English respelling, dictionary style, with the stressed syllable in capitals.
 * Written by hand; the unusual names follow how they are said in the films and
 * series as best known, and are the entries most worth a second look.
 */

const SAY: Readonly<Record<string, string>> = {
  // Core
  FORCE: "FORSS", JEDI: "JEH-dye", REBEL: "REH-bul", EMPIRE: "EM-pyre", GALAXY: "GAL-uk-see",
  PLANET: "PLAN-it", DROID: "DROYD", PILOT: "PYE-lut", SABER: "SAY-ber", MASTER: "MASS-ter",
  // Order
  PADAWAN: "PAD-uh-wahn", KNIGHT: "NYTE", TEMPLE: "TEM-pul", COUNCIL: "KOWN-sul", SENATE: "SEN-it",
  ALLIANCE: "uh-LYE-unss",
  // Fleet
  CRUISER: "KROO-zer", FIGHTER: "FYE-ter", FALCON: "FAL-kun", SHUTTLE: "SHUT-ul", HANGAR: "HANG-er",
  // Fringe
  CANTINA: "kan-TEE-nuh", SMUGGLER: "SMUG-ler", BOUNTY: "BOWN-tee", HUNTER: "HUN-ter",
  // Creatures
  WOOKIEE: "WOOK-ee", EWOK: "EE-wok", JAWA: "JAH-wuh", TUSKEN: "TUSS-ken", BANTHA: "BAN-thuh",
  TAUNTAUN: "TAWN-tawn", RANCOR: "RANG-kor", SARLACC: "SAR-lak", KRAYT: "KRATE", PORG: "PORG",
  LOTHCAT: "LOTH-kat",
  // Planets
  ENDOR: "EN-dor", HOTH: "HAWTH", DAGOBAH: "DAY-guh-bah", TATOOINE: "TAT-oo-een", NABOO: "nuh-BOO",
  BESPIN: "BESS-pin", JAKKU: "JAH-koo", ALDERAAN: "AWL-der-ahn", KASHYYYK: "kah-SHEEK",
  MUSTAFAR: "MOOS-tuh-far", SCARIF: "SKAR-if", CRAIT: "KRATE", YAVIN: "YAH-vin", KESSEL: "KESS-ul",
  CORELLIA: "kor-EL-ee-uh", MANDALORE: "MAN-duh-lor", KAMINO: "kuh-MEE-no", GEONOSIS: "jee-oh-NO-sis",
  UTAPAU: "OO-tuh-pow", FELUCIA: "fuh-LOO-shuh", LOTHAL: "LOH-thawl", DATHOMIR: "DATH-oh-meer",
  ILUM: "EYE-lum", EXEGOL: "EK-suh-gol",
  // Virtues
  HOPE: "HOHP", COURAGE: "KUR-ij", PATIENCE: "PAY-shunss", TRUST: "TRUST", BALANCE: "BAL-unss",
  DESTINY: "DESS-tih-nee", WISDOM: "WIZ-dum", HONOR: "ON-er", LOYALTY: "LOY-ul-tee", FREEDOM: "FREE-dum",
  PEACE: "PEESS", JUSTICE: "JUSS-tiss", KINDNESS: "KYND-niss",
  // Kin
  FRIEND: "FREND", FAMILY: "FAM-uh-lee", BROTHER: "BRUH-ther", SISTER: "SISS-ter", FATHER: "FAH-ther",
  MOTHER: "MUH-ther", CHILD: "CHYLD",
  // Space
  STAR: "STAR", MOON: "MOON", COMET: "KOM-it", NEBULA: "NEB-yoo-luh", ORBIT: "OR-bit",
  // Tech
  ROCKET: "ROK-it", ENGINE: "EN-jin", SHIELD: "SHEELD", LASER: "LAY-zer", BLASTER: "BLASS-ter",
  // Lore
  CRYSTAL: "KRISS-tul", KYBER: "KYE-ber", HOLOCRON: "HOL-uh-kron", HOLOGRAM: "HOL-uh-gram",
  // Signals
  BEACON: "BEE-kun", SIGNAL: "SIG-nul", CIPHER: "SYE-fer", SECRET: "SEE-krit",
  // Ranks
  MISSION: "MISH-un", SQUADRON: "SKWOD-run", COMMAND: "kuh-MAND", CAPTAIN: "KAP-tin",
  ADMIRAL: "AD-mer-ul", GENERAL: "JEN-er-ul", TROOPER: "TROO-per", SCOUT: "SKOWT", RANGER: "RAYN-jer",
  GUARDIAN: "GAR-dee-un", SENTINEL: "SEN-tih-nul", WARRIOR: "WOR-ee-er",
  // Terrain
  DESERT: "DEZ-ert", FOREST: "FOR-ist", SWAMP: "SWOMP", CANYON: "KAN-yun", GLACIER: "GLAY-sher",
  VOLCANO: "vol-KAY-no", ASTEROID: "ASS-ter-oyd", HORIZON: "huh-RYE-zun", SUNSET: "SUN-set",
  TWILIGHT: "TWYE-lyte",
  // Cast
  LUKE: "LOOK", LEIA: "LAY-uh", HAN: "HAHN", CHEWIE: "CHOO-ee", YODA: "YOH-duh", VADER: "VAY-der",
  KENOBI: "keh-NOH-bee", ANAKIN: "AN-uh-kin", PADME: "PAD-may", AHSOKA: "ah-SOH-kuh", REY: "RAY",
  FINN: "FIN", POE: "POH", KYLO: "KYE-loh", LANDO: "LAN-doh", BOBA: "BOH-buh", JABBA: "JAB-uh",
  GROGU: "GROH-goo", MANDO: "MAN-doh", ARTOO: "AR-too", THREEPIO: "THREE-pee-oh",
  PALPATINE: "PAL-puh-teen", DOOKU: "DOO-koo", MAUL: "MAWL", QUIGON: "KWY-gon", WINDU: "WIN-doo",
  SOLO: "SOH-loh", SKYWALKER: "SKY-waw-ker", ORGANA: "or-GAH-nuh", AMIDALA: "ah-mih-DAH-luh",
  TARKIN: "TAR-kin", GRIEVOUS: "GREE-vuss", EZRA: "EZ-ruh", HERA: "HAIR-uh", SABINE: "suh-BEEN",
  KANAN: "KAY-nun", ZEB: "ZEB", THRAWN: "THRAWN", CASSIAN: "KASS-ee-un", JYN: "JIN", ANDOR: "AN-dor",
  CHIRRUT: "CHEER-oot", BODHI: "BOH-dee", KRENNIC: "KREN-ik", HUX: "HUKS", PHASMA: "FAZ-muh",
  SNOKE: "SNOHK", ROSE: "ROHZ", HOLDO: "HOL-doh", ACKBAR: "AK-bar", WEDGE: "WEJ", BIGGS: "BIGZ",
  GREEDO: "GREE-doh", WATTO: "WAH-toh", JARJAR: "JAR-jar", BAZE: "BAYZ", SAW: "SAW",
  CHOPPER: "CHOP-er", REX: "REKS", CODY: "KOH-dee",
  // Special days
  SITH: "SITH", STARWARS: "STAR-worz", BRASIL: "brah-ZEEW", MURPH: "MURF", FLAMENGO: "flah-MENG-goo",
  LIFEDAY: "LYFE-day", NATAL: "nah-TOW",
};

/** The respelling for a known word, or null. A name we were never given gets no guess. */
export function pronounce(word: string): string | null {
  return SAY[word] ?? null;
}
