from load_utils import load_pretrained_vision_models, get_clip, get_CIFAR10_synsets_info, get_transform,\
    load_pretrained_imagenet_models
import torch.nn as nn
import clip
import torch

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def construct_class_texts_single(orig_prompt, imgNet=True):
    from clip.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    class_texts = []
    vocab = tokenizer.encoder
    inv_clip_vocab = [0] * 49408
    for item in vocab.items():
        inv_clip_vocab[item[1]] = item[0]
    wordid_li = [int(word_id) for word_id in orig_prompt.split(' ')]
    prompt = [inv_clip_vocab[wordid] for wordid in wordid_li]
    # print('prompt: ', ' '.join(prompt))
    stack_tensor_li = []
    for word_id in wordid_li:
        if imgNet:
            tmp = torch.zeros(1000).long()
        else:
            tmp = torch.zeros(10).long()
        tmp.fill_(word_id)
        stack_tensor_li.append(tmp)
    stack_tensor = torch.stack(stack_tensor_li, dim=1)
    if imgNet:
        all_classes= imagenet_classes
    else:
        all_classes = cifar_classes

    for name in all_classes:
        class_texts.append(name)
    class_texts = clip.tokenize(class_texts, context_length=20)
    class_texts = torch.cat((class_texts[:, 0].unsqueeze(1), stack_tensor, class_texts[:, 1:-len(wordid_li)]), dim=1)
    # print(class_texts.shape)
    return class_texts

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

# orig_imagenet_classes = ['tench, Tinca tinca', 'goldfish, Carassius auratus', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark', 'electric ray, crampfish, numbfish, torpedo', 'stingray', 'cock', 'hen', 'ostrich, Struthio camelus', 'brambling, Fringilla montifringilla', 'goldfinch, Carduelis carduelis', 'house finch, linnet, Carpodacus mexicanus', 'junco, snowbird', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'robin, American robin, Turdus migratorius', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel, dipper', 'kite', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'great grey owl, great gray owl, Strix nebulosa', 'European fire salamander, Salamandra salamandra', 'common newt, Triturus vulgaris', 'eft', 'spotted salamander, Ambystoma maculatum', 'axolotl, mud puppy, Ambystoma mexicanum', 'bullfrog, Rana catesbeiana', 'tree frog, tree-frog', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'loggerhead, loggerhead turtle, Caretta caretta', 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'mud turtle', 'terrapin', 'box turtle, box tortoise', 'banded gecko', 'common iguana, iguana, Iguana iguana', 'American chameleon, anole, Anolis carolinensis', 'whiptail, whiptail lizard', 'agama', 'frilled lizard, Chlamydosaurus kingi', 'alligator lizard', 'Gila monster, Heloderma suspectum', 'green lizard, Lacerta viridis', 'African chameleon, Chamaeleo chamaeleon', 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'African crocodile, Nile crocodile, Crocodylus niloticus', 'American alligator, Alligator mississipiensis', 'triceratops', 'thunder snake, worm snake, Carphophis amoenus', 'ringneck snake, ring-necked snake, ring snake', 'hognose snake, puff adder, sand viper', 'green snake, grass snake', 'king snake, kingsnake', 'garter snake, grass snake', 'water snake', 'vine snake', 'night snake, Hypsiglena torquata', 'boa constrictor, Constrictor constrictor', 'rock python, rock snake, Python sebae', 'Indian cobra, Naja naja', 'green mamba', 'sea snake', 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'sidewinder, horned rattlesnake, Crotalus cerastes', 'trilobite', 'harvestman, daddy longlegs, Phalangium opilio', 'scorpion', 'black and gold garden spider, Argiope aurantia', 'barn spider, Araneus cavaticus', 'garden spider, Aranea diademata', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse, partridge, Bonasa umbellus', 'prairie chicken, prairie grouse, prairie fowl', 'peacock', 'quail', 'partridge', 'African grey, African gray, Psittacus erithacus', 'macaw', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser, Mergus serrator', 'goose', 'black swan, Cygnus atratus', 'tusker', 'echidna, spiny anteater, anteater', 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus', 'wallaby, brush kangaroo', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'sea anemone, anemone', 'brain coral', 'flatworm, platyhelminth', 'nematode, nematode worm, roundworm', 'conch', 'snail', 'slug', 'sea slug, nudibranch', 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'chambered nautilus, pearly nautilus, nautilus', 'Dungeness crab, Cancer magister', 'rock crab, Cancer irroratus', 'fiddler crab', 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'isopod', 'white stork, Ciconia ciconia', 'black stork, Ciconia nigra', 'spoonbill', 'flamingo', 'little blue heron, Egretta caerulea', 'American egret, great white heron, Egretta albus', 'bittern', 'crane', 'limpkin, Aramus pictus', 'European gallinule, Porphyrio porphyrio', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'bustard', 'ruddy turnstone, Arenaria interpres', 'red-backed sandpiper, dunlin, Erolia alpina', 'redshank, Tringa totanus', 'dowitcher', 'oystercatcher, oyster catcher', 'pelican', 'king penguin, Aptenodytes patagonica', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'dugong, Dugong dugon', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog, Maltese terrier, Maltese', 'Pekinese, Pekingese, Peke', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound, Afghan', 'basset, basset hound', 'beagle', 'bloodhound, sleuthhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound, Walker foxhound', 'English foxhound', 'redbone', 'borzoi, Russian wolfhound', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound, Ibizan Podenco', 'Norwegian elkhound, elkhound', 'otterhound, otter hound', 'Saluki, gazelle hound', 'Scottish deerhound, deerhound', 'Weimaraner', 'Staffordshire bullterrier, Staffordshire bull terrier', 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier, Sealyham', 'Airedale, Airedale terrier', 'cairn, cairn terrier', 'Australian terrier', 'Dandie Dinmont, Dandie Dinmont terrier', 'Boston bull, Boston terrier', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier, Scottish terrier, Scottie', 'Tibetan terrier, chrysanthemum dog', 'silky terrier, Sydney silky', 'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa, Lhasa apso', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla, Hungarian pointer', 'English setter', 'Irish setter, red setter', 'Gordon setter', 'Brittany spaniel', 'clumber, clumber spaniel', 'English springer, English springer spaniel', 'Welsh springer spaniel', 'cocker spaniel, English cocker spaniel, cocker', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog, bobtail', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'collie', 'Border collie', 'Bouvier des Flandres, Bouviers des Flandres', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard, St Bernard', 'Eskimo dog, husky', 'malamute, malemute, Alaskan malamute', 'Siberian husky', 'dalmatian, coach dog, carriage dog', 'affenpinscher, monkey pinscher, monkey dog', 'basenji', 'pug, pug-dog', 'Leonberg', 'Newfoundland, Newfoundland dog', 'Great Pyrenees', 'Samoyed, Samoyede', 'Pomeranian', 'chow, chow chow', 'keeshond', 'Brabancon griffon', 'Pembroke, Pembroke Welsh corgi', 'Cardigan, Cardigan Welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf, grey wolf, gray wolf, Canis lupus', 'white wolf, Arctic wolf, Canis lupus tundrarum', 'red wolf, maned wolf, Canis rufus, Canis niger', 'coyote, prairie wolf, brush wolf, Canis latrans', 'dingo, warrigal, warragal, Canis dingo', 'dhole, Cuon alpinus', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'hyena, hyaena', 'red fox, Vulpes vulpes', 'kit fox, Vulpes macrotis', 'Arctic fox, white fox, Alopex lagopus', 'grey fox, gray fox, Urocyon cinereoargenteus', 'tabby, tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat, Siamese', 'Egyptian cat', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'lynx, catamount', 'leopard, Panthera pardus', 'snow leopard, ounce, Panthera uncia', 'jaguar, panther, Panthera onca, Felis onca', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'brown bear, bruin, Ursus arctos', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'sloth bear, Melursus ursinus, Ursus ursinus', 'mongoose', 'meerkat, mierkat', 'tiger beetle', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'ground beetle, carabid beetle', 'long-horned beetle, longicorn, longicorn beetle', 'leaf beetle, chrysomelid', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'cricket', 'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'cicada, cicala', 'leafhopper', 'lacewing, lacewing fly', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'damselfly', 'admiral', 'ringlet, ringlet butterfly', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly', 'sulphur butterfly, sulfur butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star', 'sea urchin', 'sea cucumber, holothurian', 'wood rabbit, cottontail, cottontail rabbit', 'hare', 'Angora, Angora rabbit', 'hamster', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'beaver', 'guinea pig, Cavia cobaya', 'sorrel', 'zebra', 'hog, pig, grunter, squealer, Sus scrofa', 'wild boar, boar, Sus scrofa', 'warthog', 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'ox', 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'bison', 'ram, tup', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'ibex, Capra ibex', 'hartebeest', 'impala, Aepyceros melampus', 'gazelle', 'Arabian camel, dromedary, Camelus dromedarius', 'llama', 'weasel', 'mink', 'polecat, fitch, foulmart, foumart, Mustela putorius', 'black-footed ferret, ferret, Mustela nigripes', 'otter', 'skunk, polecat, wood pussy', 'badger', 'armadillo', 'three-toed sloth, ai, Bradypus tridactylus', 'orangutan, orang, orangutang, Pongo pygmaeus', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'gibbon, Hylobates lar', 'siamang, Hylobates syndactylus, Symphalangus syndactylus', 'guenon, guenon monkey', 'patas, hussar monkey, Erythrocebus patas', 'baboon', 'macaque', 'langur', 'colobus, colobus monkey', 'proboscis monkey, Nasalis larvatus', 'marmoset', 'capuchin, ringtail, Cebus capucinus', 'howler monkey, howler', 'titi, titi monkey', 'spider monkey, Ateles geoffroyi', 'squirrel monkey, Saimiri sciureus', 'Madagascar cat, ring-tailed lemur, Lemur catta', 'indri, indris, Indri indri, Indri brevicaudatus', 'Indian elephant, Elephas maximus', 'African elephant, Loxodonta africana', 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'barracouta, snoek', 'eel', 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 'rock beauty, Holocanthus tricolor', 'anemone fish', 'sturgeon', 'gar, garfish, garpike, billfish, Lepisosteus osseus', 'lionfish', 'puffer, pufferfish, blowfish, globefish', 'abacus', 'abaya', "academic gown, academic robe, judge's robe", 'accordion, piano accordion, squeeze box', 'acoustic guitar', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'airliner', 'airship, dirigible', 'altar', 'ambulance', 'amphibian, amphibious vehicle', 'analog clock', 'apiary, bee house', 'apron', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'bakery, bakeshop, bakehouse', 'balance beam, beam', 'balloon', 'ballpoint, ballpoint pen, ballpen, Biro', 'Band Aid', 'banjo', 'bannister, banister, balustrade, balusters, handrail', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel, cask', 'barrow, garden cart, lawn cart, wheelbarrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap, swimming cap', 'bath towel', 'bathtub, bathing tub, bath, tub', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'beacon, lighthouse, beacon light, pharos', 'beaker', 'bearskin, busby, shako', 'beer bottle', 'beer glass', 'bell cote, bell cot', 'bib', 'bicycle-built-for-two, tandem bicycle, tandem', 'bikini, two-piece', 'binder, ring-binder', 'binoculars, field glasses, opera glasses', 'birdhouse', 'boathouse', 'bobsled, bobsleigh, bob', 'bolo tie, bolo, bola tie, bola', 'bonnet, poke bonnet', 'bookcase', 'bookshop, bookstore, bookstall', 'bottlecap', 'bow', 'bow tie, bow-tie, bowtie', 'brass, memorial tablet, plaque', 'brassiere, bra, bandeau', 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'breastplate, aegis, egis', 'broom', 'bucket, pail', 'buckle', 'bulletproof vest', 'bullet train, bullet', 'butcher shop, meat market', 'cab, hack, taxi, taxicab', 'caldron, cauldron', 'candle, taper, wax light', 'cannon', 'canoe', 'can opener, tin opener', 'cardigan', 'car mirror', 'carousel, carrousel, merry-go-round, roundabout, whirligig', "carpenter's kit, tool kit", 'carton', 'car wheel', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello, violoncello', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'chain', 'chainlink fence', 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'chain saw, chainsaw', 'chest', 'chiffonier, commode', 'chime, bell, gong', 'china cabinet, china closet', 'Christmas stocking', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'cleaver, meat cleaver, chopper', 'cliff dwelling', 'cloak', 'clog, geta, patten, sabot', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil, spiral, volute, whorl, helix', 'combination lock', 'computer keyboard, keypad', 'confectionery, confectionary, candy store', 'container ship, containership, container vessel', 'convertible', 'corkscrew, bottle screw', 'cornet, horn, trumpet, trump', 'cowboy boot', 'cowboy hat, ten-gallon hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib, cot', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam, dike, dyke', 'desk', 'desktop computer', 'dial telephone, dial phone', 'diaper, nappy, napkin', 'digital clock', 'digital watch', 'dining table, board', 'dishrag, dishcloth', 'dishwasher, dish washer, dishwashing machine', 'disk brake, disc brake', 'dock, dockage, docking facility', 'dogsled, dog sled, dog sleigh', 'dome', 'doormat, welcome mat', 'drilling platform, offshore rig', 'drum, membranophone, tympan', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan, blower', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso maker', 'face powder', 'feather boa, boa', 'file, file cabinet, filing cabinet', 'fireboat', 'fire engine, fire truck', 'fire screen, fireguard', 'flagpole, flagstaff', 'flute, transverse flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster', 'freight car', 'French horn, horn', 'frying pan, frypan, skillet', 'fur coat', 'garbage truck, dustcart', 'gasmask, respirator, gas helmet', 'gas pump, gasoline pump, petrol pump, island dispenser', 'goblet', 'go-kart', 'golf ball', 'golfcart, golf cart', 'gondola', 'gong, tam-tam', 'gown', 'grand piano, grand', 'greenhouse, nursery, glasshouse', 'grille, radiator grille', 'grocery store, grocery, food market, market', 'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'hand-held computer, hand-held microcomputer', 'handkerchief, hankie, hanky, hankey', 'hard disc, hard disk, fixed disk', 'harmonica, mouth organ, harp, mouth harp', 'harp', 'harvester, reaper', 'hatchet', 'holster', 'home theater, home theatre', 'honeycomb', 'hook, claw', 'hoopskirt, crinoline', 'horizontal bar, high bar', 'horse cart, horse-cart', 'hourglass', 'iPod', 'iron, smoothing iron', "jack-o'-lantern", 'jean, blue jean, denim', 'jeep, landrover', 'jersey, T-shirt, tee shirt', 'jigsaw puzzle', 'jinrikisha, ricksha, rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat, laboratory coat', 'ladle', 'lampshade, lamp shade', 'laptop, laptop computer', 'lawn mower, mower', 'lens cap, lens cover', 'letter opener, paper knife, paperknife', 'library', 'lifeboat', 'lighter, light, igniter, ignitor', 'limousine, limo', 'liner, ocean liner', 'lipstick, lip rouge', 'Loafer', 'lotion', 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', "loupe, jeweler's loupe", 'lumbermill, sawmill', 'magnetic compass', 'mailbag, postbag', 'mailbox, letter box', 'maillot', 'maillot, tank suit', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'matchstick', 'maypole', 'maze, labyrinth', 'measuring cup', 'medicine chest, medicine cabinet', 'megalith, megalithic structure', 'microphone, mike', 'microwave, microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt, mini', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home, manufactured home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter, scooter', 'mountain bike, all-terrain bike, off-roader', 'mountain tent', 'mouse, computer mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook, notebook computer', 'obelisk', 'oboe, hautboy, hautbois', 'ocarina, sweet potato', 'odometer, hodometer, mileometer, milometer', 'oil filter', 'organ, pipe organ', 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle, boat paddle', 'paddlewheel, paddle wheel', 'padlock', 'paintbrush', "pajama, pyjama, pj's, jammies", 'palace', 'panpipe, pandean pipe, syrinx', 'paper towel', 'parachute, chute', 'parallel bars, bars', 'park bench', 'parking meter', 'passenger car, coach, carriage', 'patio, terrace', 'pay-phone, pay-station', 'pedestal, plinth, footstall', 'pencil box, pencil case', 'pencil sharpener', 'perfume, essence', 'Petri dish', 'photocopier', 'pick, plectrum, plectron', 'pickelhaube', 'picket fence, paling', 'pickup, pickup truck', 'pier', 'piggy bank, penny bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate, pirate ship', 'pitcher, ewer', "plane, carpenter's plane, woodworking plane", 'planetarium', 'plastic bag', 'plate rack', 'plow, plough', "plunger, plumber's helper", 'Polaroid camera, Polaroid Land camera', 'pole', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'poncho', 'pool table, billiard table, snooker table', 'pop bottle, soda bottle', 'pot, flowerpot', "potter's wheel", 'power drill', 'prayer rug, prayer mat', 'printer', 'prison, prison house', 'projectile, missile', 'projector', 'puck, hockey puck', 'punching bag, punch bag, punching ball, punchball', 'purse', 'quill, quill pen', 'quilt, comforter, comfort, puff', 'racer, race car, racing car', 'racket, racquet', 'radiator', 'radio, wireless', 'radio telescope, radio reflector', 'rain barrel', 'recreational vehicle, RV, R.V.', 'reel', 'reflex camera', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'revolver, six-gun, six-shooter', 'rifle', 'rocking chair, rocker', 'rotisserie', 'rubber eraser, rubber, pencil eraser', 'rugby ball', 'rule, ruler', 'running shoe', 'safe', 'safety pin', 'saltshaker, salt shaker', 'sandal', 'sarong', 'sax, saxophone', 'scabbard', 'scale, weighing machine', 'school bus', 'schooner', 'scoreboard', 'screen, CRT screen', 'screw', 'screwdriver', 'seat belt, seatbelt', 'sewing machine', 'shield, buckler', 'shoe shop, shoe-shop, shoe store', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule, slipstick', 'sliding door', 'slot, one-armed bandit', 'snorkel', 'snowmobile', 'snowplow, snowplough', 'soap dispenser', 'soccer ball', 'sock', 'solar dish, solar collector, solar furnace', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', "spider web, spider's web", 'spindle', 'sports car, sport car', 'spotlight, spot', 'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch, stop watch', 'stove', 'strainer', 'streetcar, tram, tramcar, trolley, trolley car', 'stretcher', 'studio couch, day bed', 'stupa, tope', 'submarine, pigboat, sub, U-boat', 'suit, suit of clothes', 'sundial', 'sunglass', 'sunglasses, dark glasses, shades', 'sunscreen, sunblock, sun blocker', 'suspension bridge', 'swab, swob, mop', 'sweatshirt', 'swimming trunks, bathing trunks', 'swing', 'switch, electric switch, electrical switch', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'tape player', 'teapot', 'teddy, teddy bear', 'television, television system', 'tennis ball', 'thatch, thatched roof', 'theater curtain, theatre curtain', 'thimble', 'thresher, thrasher, threshing machine', 'throne', 'tile roof', 'toaster', 'tobacco shop, tobacconist shop, tobacconist', 'toilet seat', 'torch', 'totem pole', 'tow truck, tow car, wrecker', 'toyshop', 'tractor', 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'tray', 'trench coat', 'tricycle, trike, velocipede', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus, trolley coach, trackless trolley', 'trombone', 'tub, vat', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle, monocycle', 'upright, upright piano', 'vacuum, vacuum cleaner', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin, fiddle', 'volleyball', 'waffle iron', 'wall clock', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'warplane, military plane', 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'washer, automatic washer, washing machine', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool, woolen, woollen', 'worm fence, snake fence, snake-rail fence, Virginia fence', 'wreck', 'yawl', 'yurt', 'web site, website, internet site, site', 'comic book', 'crossword puzzle, crossword', 'street sign', 'traffic light, traffic signal, stoplight', 'book jacket, dust cover, dust jacket, dust wrapper', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot, hotpot', 'trifle', 'ice cream, icecream', 'ice lolly, lolly, lollipop, popsicle', 'French loaf', 'bagel, beigel', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini, courgette', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber, cuke', 'artichoke, globe artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple, ananas', 'banana', 'jackfruit, jak, jack', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce, chocolate syrup', 'dough', 'meat loaf, meatloaf', 'pizza, pizza pie', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff, drop, drop-off', 'coral reef', 'geyser', 'lakeside, lakeshore', 'promontory, headland, head, foreland', 'sandbar, sand bar', 'seashore, coast, seacoast, sea-coast', 'valley, vale', 'volcano', 'ballplayer, baseball player', 'groom, bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'corn', 'acorn', 'hip, rose hip, rosehip', 'buckeye, horse chestnut, conker', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn, carrion fungus', 'earthstar', 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 'bolete', 'ear, spike, capitulum', 'toilet tissue, toilet paper, bathroom tissue']
# def get_imgNet_class():
#     imagenet_texts = []
#     for class_name in imagenet_classes:
#         imagenet_texts.append('a photo of a ' + class_name)
#     return imagenet_texts


def get_promptsid_li(prompts_id_file):
    def filter(li):
        new_li = {}
        new_li['320 1125 539 320'] = 1
        for item in li:
            if item not in new_li:
                new_li[item] = 1
            else:
                new_li[item] += 1
        return new_li
    with open(prompts_id_file, 'r') as f:
        promptsid_li = filter(f.read().split('\n')[:-1])
    return promptsid_li
# def get_promptsid_li(prompts_id_file):
#     def filter(li):
#         new_li = []
#         for item in li:
#             if item not in new_li:
#                 new_li.append(item)
#         if '320 1125 539 320' not in new_li:
#             new_li.append('320 1125 539 320')
#         return new_li
#     with open(prompts_id_file, 'r') as f:
#         promptsid_li = filter(f.read().split('\n')[:-1])
#     return promptsid_li



from tqdm import tqdm
def construct_class_texts(class_name, stack_prompt_tensor):
    class_texts = [class_name] * stack_prompt_tensor.shape[0]
    class_texts = clip.tokenize(class_texts, context_length=20)  # 23, 77
    # (23, 1) (23, 4) (23, ())
    class_texts = torch.cat((class_texts[:, 0].unsqueeze(1), stack_prompt_tensor, class_texts[:, 1:-4]), dim=1)
    return class_texts.cuda()


def get_zero_shot_weights(img_Net, clip_model, stack_prompt_tensor):
    zeroshot_weights = []
    with torch.no_grad():
        classes = imagenet_classes if img_Net else get_CIFAR10_synsets_info()
        for classname in classes:
            # [a photo of a dog, a picture of a dog, .....]
            # texts = [template.format(classname) for template in imagenet_templates]  # format with class
            # texts shape: (template_number, context_length)
            # texts = clip.tokenize(texts).cuda()  # tokenize
            texts = construct_class_texts(classname, stack_prompt_tensor)
            class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            # print(class_embeddings.shape)  # 80, 512
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights



class VictimModel(nn.Module):
    def __init__(self, **kwargs):
        super(VictimModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_type = kwargs['eval_type']
        self.text = None
        self.kwargs = kwargs
        model = kwargs['model']
        # print(model)
        self.img_Net = kwargs['imgNet']
        class_texts = get_CIFAR10_synsets_info()
        if self.eval_type == 'CLIP':
            if model is not None:
                self.model = model
            else:
                clip_type = kwargs['clip_type']
                self.model = get_clip(clip_type)
                # state_dict = torch.load('clip'+clip_type+'.pkl' if clip_type != 'ViT-B/32' else 'clipvit.pkl', )
                # self.model.load_state_dict(state_dict)

            cons_dict = {'RN50': '23703 1150 316 3628',
                         'RN50x4':'2689 1381 34383 550',
                         'ViT-B/32':'1848 1664 653 912',
                         'RN101':'7395 16694 7811 320' }
            # print(class_texts)
            # print('aa')
            if self.img_Net:
                # self.texts = clip.tokenize(get_imgNet_class(), context_length=20).to(self.device)
                self.texts=construct_class_texts_single(cons_dict[clip_type] ,True).to(self.device)



                self.model.positional_embedding.data = self.model.positional_embedding[:20, :]
            else:
                self.texts = self.text=construct_class_texts_single(cons_dict[clip_type] ,False).to(self.device)
                self.model.positional_embedding.data = self.model.positional_embedding[:20, :]

        elif self.eval_type == 'CLIP_ensemble':
            clip_type = kwargs['clip_type']
            self.model = get_clip(clip_type)
            self.model.positional_embedding.data = self.model.positional_embedding[:20, :]
            zeroshot_weights = []
            with torch.no_grad():
                if self.img_Net:
                    class_texts = imagenet_classes
                for classname in class_texts:
                    texts = [template.format(classname) for template in imagenet_templates]  # format with class
                    texts = clip.tokenize(texts, context_length=20).cuda()  # tokenize
                    class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                    # print(class_embeddings.shape)  # 80, 512
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1).detach()  # 512, 10
                # print(self.zeroshot_weights)
        elif self.eval_type == 'CLIP_AUTO':
            clip_type = kwargs['clip_type']
            sample_p = kwargs['sample_p']
            self.model = get_clip(clip_type)
            self.model.positional_embedding.data = self.model.positional_embedding[:20, :]
            prompts_id_file = kwargs['prompts_id_file']
            # promptids_li = get_promptsid_li(prompts_id_file)
            prompts_li = get_promptsid_li(prompts_id_file)
            sorted_prompts_li = sorted(prompts_li.items(), key=lambda item: item[1], reverse=True)

            # print('template number: ', len(sorted_prompts_li))
            if clip_type == 'ViT-B/32':
                promptids_li = [item[0] for item in sorted_prompts_li]
            else:
                promptids_li = [item[0] for item in sorted_prompts_li[:sample_p]]



            stack_li = []
            for prompt_id in promptids_li:
                wordid_li = [int(word_id) for word_id in prompt_id.split(' ')]  # [333,22,13,24]
                stack_li.append(torch.tensor(wordid_li).long())
            stack_prompt_tensor = torch.stack(stack_li, dim=0)  # template_nums,4
            self.zeroshot_weights = get_zero_shot_weights(img_Net=self.img_Net, clip_model=self.model, stack_prompt_tensor=stack_prompt_tensor)


        elif self.eval_type == 'CLIP_LP':
            clip_type = kwargs['clip_type']
            model = get_clip(clip_type)
            self.vision_model = model.visual
            self.target_type = self.vision_model.conv1.weight.dtype
            hidden_dim_dict = {'ViT-B/32': 512,
                               'RN50': 1024,
                               'RN101': 512,
                               'RN50x4': 640}
            hidden_dim = hidden_dim_dict[clip_type]
            self.prob_layer = nn.Linear(hidden_dim, 10, bias=True)
            state_dict = torch.load('CLIP'+clip_type+'.pkl' if clip_type != 'ViT-B/32' else 'CLIPViT-B32.pkl')
            self.prob_layer.load_state_dict(state_dict)
            self.prob_layer = self.prob_layer.to(self.device)
            self.prob_layer.eval()

        else:
            if self.img_Net:
                self.model = load_pretrained_imagenet_models(kwargs['vision_model'])
            else:
                self.model = load_pretrained_vision_models(kwargs['vision_model'])


        for param in self.model.parameters():
            param.requires_grad = False




    def forward(self, image):
        image = image.to(self.device)
        if self.eval_type == 'CLIP':
            output, _ = self.model(image, self.texts)
        elif self.eval_type == 'CLIP_ensemble':
            image_features = self.model.encode_image(image)  # 32, 512
            # print(image_features.shape)
            image_features_norm = torch.norm(image_features, dim=-1, keepdim=True)
            image_features = image_features / image_features_norm
            output = 100. * image_features @ self.zeroshot_weights  # 32, 10
        elif self.eval_type == 'CLIP_LP':
            img_features = self.vision_model(image.type(self.target_type))  # 32, 1024
            output = self.prob_layer(img_features.float())  # batch_size, 10
        elif self.eval_type == 'CLIP_AUTO':
            # predict
            image_features = self.model.encode_image(image)  # 1, 512
            # print(image_features.shape)
            norm = image_features.norm(dim=-1, keepdim=True)
            image_features = image_features / norm
            output = 100. * image_features @ self.zeroshot_weights  # 32, 1000
        else:
            output = self.model(image)
        # batch_size (1), class_num (1000)
        return output






