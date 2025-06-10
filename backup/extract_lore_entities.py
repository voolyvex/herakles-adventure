import os
import re
import json
from collections import defaultdict

CHUNKS_DIR = "lore_chunks"
ENTITY_DIR = "lore_entities"
os.makedirs(ENTITY_DIR, exist_ok=True)

# Expanded list of known characters (Greek, Roman, Norse, etc.) - Case Sensitive for Proper Nouns
# Note: This is still not exhaustive, but much larger. Regex might still miss some.
KNOWN_CHARACTERS = set([
    'AEetes','AEneas','AEson','AEolus','AEsculapius','AEthra','Agamemnon','Agave','Agenor','Ajax','Alcestis','Alcides','Alcinous','Alcmena','Alcyone',
    'Amalthea','Amata','Amphion','Amphitrite','Amulius','Amycus','Ancaeus','Anchises','Androgeus','Andromache','Andromeda','Antaeus','Antenor','Anteros','Anticlea',
    'Antigone','Antilochus','Antinous','Antiope','Aphrodite','Apollo','Apollon','Arachne','Arcas','Arete','Arethusa','Argonauts','Argus','Ariadne','Arion','Aristaeus',
    'Artemis','Ascanius','Ascalaphus','Asclepius','Asia','Asteria','Astyanax','Astyoche','Atalanta','Athamas','Athena','Athene','Atlas','Atreus','Atropos','Augeas',
    'Aurora','Autolycus','Bacchus','Battus','Baucis','Bellerophon','Bellona','Boreas','Briareus','Briseis','Busiris','Byblis','Cacus','Cadmus','Caeneus','Calais',
    'Calchas','Calliope','Callirrhoe','Callisto','Calypso','Camilla','Canace','Cassandra','Cassiopeia','Castor','Cecrops','Celaeno','Celeus','Centaurs','Cephalus',
    'Cepheus','Cerberus','Cercyon','Ceres','Ceto','Ceyx','Chaos','Charon','Charybdis','Chimera','Chiron','Chloris','Chryseis','Chrysippus','Chryses','Cilix',
    'Circe','Clio','Clotho','Clymene','Clytemnestra','Clytie','Codrus','Comus','Copreus','Coronis','Corybantes','Creon','Cretheus','Creusa','Cronos','Cronus',
    'Cupid','Cybele','Cyclopes','Cycnus','Cygnus','Cyparissus','Cyrene','Daedalion','Daedalus','Danae','Danaides','Danaus','Daphne','Dardanus','Deianira',
    'Deidamia','Deiphobus','Demeter','Demophoon','Deucalion','Diana','Dido','Diomedes','Dione','Dionysus','Dirce','Dryades','Dryope','Echo','Eetion','Egeria',
    'Electra','Electryon','Endymion','Enyo','Eos','Epaphus','Epeius','Epimetheus','Erato','Erebus','Erechtheus','Erginus','Erichthonius','Erigone','Eris',
    'Eros','Erymanthian Boar','Eryx','Eteocles','Euadne','Eumaeus','Eumenides','Eumolpus','Europa','Eurus','Euryale','Euryalus','Eurycleia','Eurydice','Eurylochus',
    'Eurynome','Eurypylus','Eurystheus','Eurytion','Euterpe','Evadne','Evander','Fama','Faun','Faunus','Faustulus','Flora','Fortuna','Furies','Gaea','Gaia',
    'Galatea','Ganymede','Geryon','Giants','Glauce','Glaucus','Gorgons','Graces','Graeae','Haemon','Hades','Harmonia','Harpies','Hebe','Hecate','Hector','Hecuba',
    'Helen','Helenus','Heliades','Helios','Helle','Hephaestus','Hera','Heracles','Hercules','Hermaphroditus','Hermes','Hermione','Hero','Hesperides','Hesperus',
    'Hestia','Hippocoon','Hippodamia','Hippolyta','Hippolytus','Hippomedon','Hippomenes','Horae','Hyacinthus','Hydra','Hygieia','Hylas','Hyperion','Hypermnestra',
    'Hypnos','Iacchus','Iapetus','Iasion','Icarus','Ida','Idaeus','Idas','Idmon','Idomeneus','Ilus','Inachus','Ino','Io','Iobates','Iolaus','Iole','Ion','Iphicles',
    'Iphigenia','Iphis','Iphitus','Iris','Ismene','Iulus','Ixion','Janus','Jason','Jocasta','Juno','Jupiter','Juturna','Juventas','Lachesis','Laertes','Laestrygones',
    'Laius','Laocoon','Laodamia','Laomedon','Lapithae','Larissa','Latona','Lausus','Lavinia','Leander','Leda','Lelex','Lemures','Leto','Leucippus','Liber','Libitina',
    'Linus','Lotis','Luna','Lycaon','Lycomedes','Lycurgus','Lynceus','Lysithea','Maenads','Maia','Manes','Mars','Marsyas','Medea','Medon','Medusa','Megaera',
    'Megapenthes','Megara','Melampus','Melanthius','Melantho','Meleager','Meliae','Melicertes','Melpomene','Memnon','Menelaus','Menestheus','Menoetius','Mentor',
    'Mercury','Meriones','Merope','Metis','Mezentius','Midas','Minerva','Minos','Minotaur','Minyae','Mnemosyne','Molossus','Momus','Morpheus','Muses','Myrmidons',
    'Myrrha','Myrtilus','Naiads','Napaeae','Narcissus','Nausicaa','Nausithous','Neleus','Nemesis','Neoptolemus','Nephele','Neptune','Nereids','Nereus','Nessus',
    'Nestor','Nike','Niobe','Nisus','Notus','Numitor','Nycteus','Nymphs','Nyx','Oceanids','Oceanus','Odysseus','Oeagrus','Oebalus','Oedipus','Oeneus','Oenomaus',
    'Oenone','Ogyges','Oileus','Olympus','Omphale','Ops','Oreads','Orestes','Orion','Orithyia','Orpheus','Ossa','Othryoneus','Otus','Paean','Palaemon','Palamedes',
    'Pales','Palladium','Pallas','Pan','Pandarus','Pandion','Pandora','Panopeus','Parcae','Paris','Parthenopaeus','Pasiphae','Patroclus','Pegasus','Peirithous',
    'Peleus','Pelias','Pelopia','Pelops','Penates','Penelope','Penthesilea','Pentheus','Perdix','Periboea','Periclymenus','Periphetes','Perse','Persephone',
    'Perseus','Phaedra','Phaethon','Phaethusa','Phantasos','Phaon','Philemon','Philoctetes','Philomela','Phineus','Phlegyas','Phocus','Phobus','Phoebe','Phoebus',
    'Phoenix','Pholus','Phorcys','Phrixus','Phyllis','Pirithous','Pittheus','Pleiades','Pleione','Pluto','Plutus','Podalirius','Podarces','Poena','Pollux','Polybus',
    'Polydamas','Polydectes','Polydorus','Polyhymnia','Polyidus','Polymestor','Polynices','Polyphemus','Polyxena','Pomona','Pontus','Porphyrion','Poseidon',
    'Praxithea','Priam','Priapus','Procne','Procris','Procrustes','Proetus','Prometheus','Proserpina','Protesilaus','Proteus','Psyche','Pterelaus','Pygmalion',
    'Pylades','Pyracmon','Pyramus','Pyrrha','Pyrrhus','Pythia','Python','Quirinus','Remus','Rhadamanthus','Rhea','Rhesus','Rhodope','Romulus','Salmacis',
    'Salmoneus','Sarpedon','Saturn','Satyrs','Scylla','Scyron','Seasons','Selene','Semele','Sibyl','Sileni','Silenus','Sinis','Sinope','Siren','Sisyphus',
    'Sol','Somnus','Sphinx','Sterope','Stheneboea','Sthenelus','Styx','Symplegades','Syrinx','Tages','Talaus','Talos','Tantalus','Tartarus','Taygete','Teiresias',
    'Telamon','Telchines','Telegonus','Telemachus','Telephus','Tellus','Tenes','Tereus','Terminus','Terpsichore','Tethys','Teucer','Teuthras','Thalia','Thamyris',
    'Thanatos','Thaumas','Theia','Themis','Thersites','Theseus','Thespius','Thetis','Thisbe','Thyestes','Thyrsus','Tiphys','Tisiphone','Titans','Tithonus',
    'Tityus','Tmolus','Triton','Triptolemus','Troilus','Trophonius','Turnus','Tyche','Tydeus','Tyndareus','Typhoeus','Typhon','Tyro','Ulysses','Urania','Uranus',
    'Venus','Vertumnus','Vesta','Victoria','Vulcan','Zephyrus','Zetes','Zethus','Zeus',
    # Norse Characters
    'Aesir','Aegir','Andvari','Angerbode','Ask','Aslaug','Audhumla','Balder','Baldr','Baldur','Bestla','Bor','Bragi','Brunhild','Brynhild','Buri','Day',
    'Earth','Einherjar','Eir','Elli','Elves','Embla','Fafnir','Fenrir','Forseti','Frey','Freya','Freyja','Frigg','Frigga','Fulla','Garm','Gefion','Geirrod',
    'Gerd','Gerda','Giants','Gimli','Ginungagap','Gioll','Gladsheim','Gna','Grid','Gullinkambi','Gullveig','Gunnar','Guttorm','Gymir','Heimdall','Heimdallr',
    'Hel','Hela','Hermod','Hlin','Hod','Hodr','Hoenir','Hogni','Hreidmar','Hrungnir','Huginn','Hyrrokkin','Idun','Iduna','Jarnsaxa','Jord','Jormungand','Jotun',
    'Jotunheim','Kari','Laufey','Lif','Lifthrasir','Lofn','Logi','Loki','Magni','Mani','Midgard Serpent','Mimir','Mjollnir','Modi','Muninn','Muspelheim',
    'Naglfar','Nana','Nanna','Narfi','Narve','Nidhogg','Night','Nine Worlds','Njord','Norns','Nott','Odin','Odur','Otr','Ran','Ratatosk','Regin','Rind',
    'Rig','Saga','Sif','Sigmund','Signy','Sigurd','Sigyn','Sjofn','Skadi','Skidbladnir','Skirnir','Skoll','Skrymir','Sleipnir','Snotra','Sol','Sun','Surtr',
    'Suttung','Svadilfari','Syn','Thialfi','Thiassi','Thor','Thrud','Thrudgelmir','Thrym','Tyr','Ull','Urd','Utgard-Loki','Valhalla','Vali','Valkyries','Vanir',
    'Var','Ve','Vidar','Vili','Volund','Volsung','Yggdrasil','Ymir',
    # Egyptian Characters (Less common in Bulfinch, but for broader scope)
    'Amun','Anubis','Apep','Apis','Aten','Atum','Bastet','Bes','Geb','Hathor','Horus','Imhotep','Isis','Khepri','Khnum','Khonsu','Maat','Montu','Mut',
    'Neith','Nekhbet','Nephthys','Nut','Osiris','Ptah','Ra','Sekhmet','Serqet','Set','Seth','Shu','Sobek','Taweret','Tefnut','Thoth', 'Wadjet'
])

# Expanded list of known places (Mythological and Ancient World)
KNOWN_PLACES = set([
    'Acheron','Acropolis','Aeaea','Aegina','AEthiopia','Aetna','Africa','Alpheus','Amazon','Aonia','Arcadia','Areopagus','Argolis','Argos','Asia','Asgard',
    'Athens','Atlas Mountains','Aulis','Avernus','Babylon','Bactra','Bithynia','Boeotia','Byzantium','Cadmea','Caicus','Calydon','Campania','Capitoline Hill',
    'Cappadocia','Caria','Carpathian Sea','Carthage','Castalian Spring','Caucasus','Cayster','Centaurs Island','Cephissus','Cephisus','Cerne','Chalcedon',
    'Chios','Cilicia','Cimmerians','Cithaeron','Cnidus','Cocytus','Colchis','Colonnus','Corinth','Crete','Cumae','Curetes','Cyclades','Cyllene','Cynthus',
    'Cyprus','Cythera','Cytorus','Danube','Dardanelles','Daulis','Delos','Delphi','Dictaean Cave','Dodona','Dolopes','Doris','Earth','Echinades','Egypt',
    'Elysian Fields','Elysium','Emathia','Enipeus','Ephyra','Epidaurus','Epirus','Erebus','Eretria','Erymanthus','Erytheia','Ethiopia','Etruria','Euboea',
    'Euxine Sea','Eurotas','Europa','Forum','Gades','Ganges','Gaul','Germania','Gigas','Greece','Hades','Haemonia','Haemus','Halys','Hellas','Hellespont',
    'Helicon','Heptapylos','Heraclea','Hermione','Hesperia','Hiberia','Hippocrene','Hispania','Hydaspes','Hymettus','Hyperborea','Iberia','Icaria','Ida',
    'Ilion','Ilissus','Ilium','India','Inachus','Iolcus','Ionia','Ithaca','Isthmus','Italy','Jotunheim','Labyrinth','Lacedaemon','Ladon','Lake Tritonis',
    'Larissa','Latium','Laurium','Lebadea','Lemnos','Lenaeum','Leontini','Lerna','Lesbos','Lethe','Leucadia','Libya','Locris','Lycaeus','Lycia','Lydia',
    'Maeander','Maeonia','Malea','Mantinea','Marathon','Media','Mediterranean Sea','Megara','Meliboea','Memphis','Messenia','Miletus','Midgard','Mount Atlas',
    'Mount Cithaeron','Mount Cyllene','Mount Dicte','Mount Eryx','Mount Etna','Mount Helicon','Mount Hymettus','Mount Ida','Mount Latmus','Mount Lycaeus',
    'Mount Olympus','Mount Ossa','Mount Parnassus','Mount Pelion','Mount Pentelicus','Muspelheim','Mycenae','Mygdonia','Myndus','Mysia','Naxos','Nemea',
    'Niflheim','Nile','Nysa','Ocalea','Ocean','Oceanus','Oechalia','Oeta','Ogygia','Olenus','Olympia','Olympus','Ossa','Othrys','Pactolus','Padus',
    'Paeonia','Palatine Hill','Palestine','Pallantium','Pangaeus','Paphos','Parnassus','Paros','Parthenius','Patara','Pelion','Pella','Peloponnesus',
    'Peneus','Pergamum','Perinthus','Persia','Phaistos','Pharos','Pharsalus','Phasis','Pherae','Phlegethon','Phocis','Phoenicia','Phrygia','Phthia',
    'Pieria','Pindus','Piraeus','Pisa','Pityus','Plataea','Pleuron','Pontus','Potniae','Prusa','Psophis','Pylos','Pyramids','Pytho','Rhodes','Rhodope',
    'Rome','Rubicon','Sais','Salamis','Samos','Samothrace','Sardinia','Sardis','Sarmatia','Scylla','Scythia','Seriphos','Sicily','Sicyon','Sidon','Sigeum',
    'Sipylus','Smyrna','Sparta','Spercheus','Strophades','Styx','Sunium','Susa','Syria','Syracuse','Taenarus','Tanagra','Tanais','Tarpeian Rock','Tarentum',
    'Tartarus','Taurus Mountains','Taygetus','Tegea','Tempe','Tenedos','Tentyra','Teos','Thasos','Thaumacia','Thebes','Thermopylae','Thespiae','Thessaly',
    'Thrace','Thria','Thrinacia','Thymbra','Tiber','Tiryns','Tmolus','Trachis','Trezene','Tripolis','Troad','Troas','Troy','Tuscia','Tyre','Underworld',
    'Utgard','Valhalla','Veii','Vesuvius','Zacynthus'
])

# Expanded list of known items, artifacts, weapons, creatures (More specific)
KNOWN_ITEMS = set([
    'Aegis','Ambrosia','Apple of Discord','Apples of Hesperides','Arcanian gear','Argo','Arrows of Hercules','Arrows of Philoctetes','Axe','Belt of Hippolyta',
    'Boar','Bow','Bow of Apollo','Bow of Eros','Bow of Odysseus','Brazen Bulls','Bull','Caduceus','Cap of Invisibility','Cart','Cauldron','Centaurs','Cerberus',
    'Ceryneian Hind','Chariot','Chariot of the Sun','Chimera','Chiton','Chlamys','Club','Club of Hercules','Cornucopia','Creeping Things','Cretan Bull','Crown',
    'Cup','Cyclops','Cymbal','Dart','Delphic Tripod','Diadem','Dirk','Distaff','Dove','Dragon','Dragon Teeth','Draught','Eagle','Earrings','Egret Feathers',
    'Elixir','Embroidered Robe','Falchion','Fleece','Flocks','Flowers','Flute','Ford','Forests','Forge','Fountain','Girdle','Girdle of Venus','Gladius',
    'Goat','Goblet','Golden Apple','Golden Fleece','Gorgoneion','Graeae','Gryphon','Halberd','Hammer','Hammer of Thor','Harp','Harpe','Helmet','Helmet of Hades',
    'Helm of Awe','Herbs','Herds','Hide','Hippocampi','Hoplite Shield','Horn','Horse','Horses of Diomedes','Hound','Hydra','Icarian Sea','Ichor','Image','Incense',
    'Iron','Ivory','Jar','Javelin','Jewels','Keys','Kithara','Knife','Knots','Labrys','Labyrinth','Lance','Laurel Wreath','Leather','Linen','Lion','Lotus',
    'Lyre','Mace','Maenads','Magic Herbs','Mares of Diomedes','Mask','Mead','Medicine','Mirror','Mist','Mistletoe','Mjollnir','Monster','Moon','Mountains',
    'Moly','Nectar','Nemean Lion','Net','Oak Leaves','Oar','Obsidian','Oil','Olive Branch','Olive Tree','Oracle','Orc','Oxen','Palladium','Pan Pipes','Papyri',
    'Parchment','Patera','Pearl','Pegasus','Pelt','Pendant','Peplos','Petasos','Phiale','Pitcher','Plants','Plough','Plumes','Poison','Pomegranate','Poppy',
    'Potions','Pouch','Purple Lock','Pyre','Python','Quiver','Ram','Reed','Relics','Ring','River','Robe','Rocks','Rod','Rose','Ruddy Gold','Rudder','Runes',
    'Sack','Sandal','Sandals of Hermes','Satyr','Scepter','Scroll','Scylla','Scythe','Sea','Serpent','Shackles','Sheep','Shield','Shield of Achilles',
    'Shield of Ajax','Ship','Sickle','Silver','Silver Bow','Siren','Skin','Sling','Snake','Snow','Spear','Spell','Sphinx','Spindle','Spoil','Spring','Staff',
    'Staff of Asclepius','Stag','Stars','Statue','Steed','Stones','Storm Cloud','Stymphalian Birds','Sun','Swan','Sword','Sword of Peleus','Symplegades',
    'Tablet','Talaria','Talismans','Teeth','Thorn','Thread','Thunderbolt','Thyrsus','Tiara','Tiger','Timber','Tinder','Tongs','Torc','Torch','Tortoise Shell',
    'Treasure','Tree','Trident','Tripod','Trumpet','Tunic','Urn','Vase','Vegetables','Veil','Vines','Viola','Viper','Wand','Water','Weapons','Wheat','Wheel',
    'Whip','Whirlpool','Wild Beasts','Wind','Wine','Wings','Wings of Icarus','Wolf','Wolf Skin','Wood','Wool','Wreath','Yarn','Yew','Yoke','Zither'
])


# Regex patterns (Consider refinement if needed, especially for places)
# Basic pattern for capitalized words/phrases, potentially multi-word
CHARACTER_PAT = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b")

# Pattern for places - often capitalized, may be followed by prepositions or commas
# This is tricky and might capture character names too easily.
PLACE_PAT = re.compile(r"\b([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\b(?:,| of| in| on| to| from| near| at| toward)")

# Pattern for items - case-insensitive search for specific known items/objects
# Building a dynamic regex from the large set
item_pattern_string = r"\b(" + "|".join(re.escape(item) for item in KNOWN_ITEMS) + r")\b"
ITEM_PAT = re.compile(item_pattern_string, re.IGNORECASE)

# Function to clean text slightly for better matching (optional)
def clean_text(text):
    # Remove possessive 's which might interfere with matching
    text = re.sub(r"'s\b", "", text)
    return text

# Process each chunk
for fname in os.listdir(CHUNKS_DIR):
    if not fname.endswith('.md'):
        continue

    fpath = os.path.join(CHUNKS_DIR, fname)
    try:
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        # Clean text before extraction
        cleaned_text = clean_text(text)

        # Extract potential entities using regex
        potential_chars = CHARACTER_PAT.findall(cleaned_text)
        potential_places = [m[0] for m in PLACE_PAT.findall(cleaned_text)] # Get the matched group
        potential_items = ITEM_PAT.findall(cleaned_text)

        # Filter against known lists (case-sensitive for characters, places need refinement)
        # Characters: Direct match against the known set
        chars = set(char for char in potential_chars if char in KNOWN_CHARACTERS)

        # Places: Match potential places against known list (case-insensitive for broader matching)
        known_places_lower = {p.lower() for p in KNOWN_PLACES}
        places = set(place for place in potential_places if place.lower() in known_places_lower)
        # Attempt to find capitalized words NOT followed by place indicators as potential places too
        # This needs careful tuning to avoid catching too many characters
        # potential_places_alt = re.findall(r"\b([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\b", cleaned_text)
        # places.update(p for p in potential_places_alt if p in KNOWN_PLACES and p not in chars) # Avoid adding characters here

        # Items: Already filtered by regex using KNOWN_ITEMS list (case-insensitive)
        # Standardize capitalization for output
        items = set(item.capitalize() for item in potential_items if item.lower() in {i.lower() for i in KNOWN_ITEMS})

        # Build final entity dictionary
        entities = defaultdict(list)
        if chars:
            entities['characters'] = sorted(list(chars))
        if places:
            entities['places'] = sorted(list(places))
        if items:
            entities['items'] = sorted(list(items))

        # Write to JSON if any entities found
        if entities:
            base = os.path.splitext(fname)[0]
            outpath = os.path.join(ENTITY_DIR, base + '.json')
            with open(outpath, 'w', encoding='utf-8') as outf:
                json.dump(entities, outf, indent=2)

    except Exception as e:
        print(f"Error processing {fname}: {str(e)}")

print(f"Entity extraction complete. See '{ENTITY_DIR}' directory.")
