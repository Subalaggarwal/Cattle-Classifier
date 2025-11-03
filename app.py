import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import time
import os

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Indian Cattle Breed Classifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== BREED INFORMATION DATABASE =====
BREED_INFO = {
    "Alambandi Cow": {
        "Origin": "Alambadi Village, Tamil Nadu, India",
        "Primary Use": "Labour-intensive farm tasks",
        "Key Traits": "Grey or dark-grey coat, backward-curving horns",
        "Milk Yield": "Approx. 432 kg per lactation",
        "Temperament": "Hardy and active",
        "Advantages": "Strong, resilient, adapted to Indian climate",
        "Limitations": "Low milk productivity",
        "Diet": "Grass and crop residues",
        "Lifespan": "15–18 years",
        "Climate Adaptability": "Hot and semi-arid regions",
        "Fun Fact": "Alambandi cattle are prized in Tamil Nadu for their stamina in ploughing rocky fields."
    },

    "Amritmahal Cow": {
        "Origin": "Karnataka, India",
        "Primary Use": "Draught and military transport work",
        "Key Traits": "Grey shade, long sharp horns, compact body",
        "Milk Yield": "Low producer (150–300 kg per lactation)",
        "Temperament": "Alert and energetic",
        "Advantages": "Exceptional strength and endurance for field work",
        "Limitations": "Poor milk yield",
        "Diet": "Grass, straw, and fodder mix",
        "Lifespan": "14–16 years",
        "Climate Adaptability": "Dry tropical zones",
        "Fun Fact": "Amritmahal bulls once served as the royal army transport animals of Mysore rulers."
    },

    "Ayrshire Cow": {
        "Origin": "Ayrshire County, Scotland",
        "Primary Use": "Dairy",
        "Key Traits": "Medium size, reddish-brown and white coat",
        "Milk Yield": "4500–6000 kg per lactation",
        "Temperament": "Active and alert",
        "Advantages": "Excellent milk quality and feed efficiency",
        "Limitations": "Sensitive to hot climates",
        "Diet": "High-protein grasses and silage",
        "Lifespan": "15–20 years",
        "Climate Adaptability": "Cool temperate regions",
        "Fun Fact": "Ayrshires are known for their balanced milk fat and protein, ideal for butter production."
    },

    "Banni Buffalo": {
        "Origin": "Kutch district, Gujarat",
        "Primary Use": "Dairy buffalo",
        "Key Traits": "Black coat with white markings and massive body",
        "Milk Yield": "12–18 litres per day",
        "Temperament": "Strong grazer, adapted to harsh terrain",
        "Advantages": "High fat milk and tolerance to heat and scarcity",
        "Limitations": "Requires open grazing fields",
        "Diet": "Pasture grass and dry fodder",
        "Lifespan": "18–20 years",
        "Climate Adaptability": "Hot and arid regions",
        "Fun Fact": "The Banni breed was developed by Maldhari pastoralists through centuries of selective breeding."
    },

    "Bargur Cow": {
        "Origin": "Bargur hills, Tamil Nadu",
        "Primary Use": "Draught animal for hill terrains",
        "Key Traits": "Reddish-brown coat with white patches and curved horns",
        "Milk Yield": "Low (150–300 kg per lactation)",
        "Temperament": "Hardy and agile",
        "Advantages": "Excellent hill climber and disease resistant",
        "Limitations": "Low milk production",
        "Diet": "Forest grasses and crop by-products",
        "Lifespan": "15 years",
        "Climate Adaptability": "Hilly and semi-arid zones",
        "Fun Fact": "Bargur cattle are so active that they were once used for mountain patrols."
    },

    "Bhadwari Buffalo": {
        "Origin": "Agra–Gwalior region of Uttar Pradesh and Madhya Pradesh",
        "Primary Use": "Dairy buffalo",
        "Key Traits": "Copper-coloured skin with black horns",
        "Milk Yield": "1000–1200 kg per lactation",
        "Temperament": "Calm and gentle",
        "Advantages": "Produces high-fat milk (>8%)",
        "Limitations": "Low overall volume of milk",
        "Diet": "Green fodder and crop residues",
        "Lifespan": "18 years",
        "Climate Adaptability": "Hot and humid plains",
        "Fun Fact": "Bhadwari milk is famous for making traditional Indian ghee due to its rich fat content."
    },

    "Brown Swiss Cow": {
        "Origin": "Switzerland",
        "Primary Use": "Dairy and cross-breeding",
        "Key Traits": "Solid brown coat, robust build, large frame",
        "Milk Yield": "4500–5000 kg per lactation",
        "Temperament": "Docile and steady",
        "Advantages": "Excellent feed conversion and long productive life",
        "Limitations": "Requires cooler climate for best yield",
        "Diet": "High-protein grasses, hay, and silage",
        "Lifespan": "18–22 years",
        "Climate Adaptability": "Cool temperate regions",
        "Fun Fact": "Brown Swiss milk is ideal for cheese-making because of its perfect protein-fat balance."
    },

    "Dangi Cow": {
        "Origin": "Nashik and Ahmednagar districts, Maharashtra and Gujarat border",
        "Primary Use": "Draught and milk production",
        "Key Traits": "Oily skin resisting rain, grey spotted coat",
        "Milk Yield": "175–800 kg per lactation",
        "Temperament": "Docile and gentle",
        "Advantages": "Performs well in heavy rainfall areas",
        "Limitations": "Slow worker in dry regions",
        "Diet": "Tropical grasses and wetland forage",
        "Lifespan": "14–16 years",
        "Climate Adaptability": "Heavy rainfall and humid zones",
        "Fun Fact": "Dangi cattle can plough fields even during monsoons thanks to their oily skin."
    },

    "Deoni Cow": {
        "Origin": "Deoni Taluk, Maharashtra",
        "Primary Use": "Dual purpose – milk and draught",
        "Key Traits": "Strong frame, white body with black patches",
        "Milk Yield": "Approx. 868 kg per lactation",
        "Temperament": "Calm and docile",
        "Advantages": "Adaptable and good for cross-breeding programs",
        "Limitations": "Moderate milk production",
        "Diet": "Grass, legume fodder, and crop residues",
        "Lifespan": "15–18 years",
        "Climate Adaptability": "Hot and semi-arid zones",
        "Fun Fact": "Deoni is believed to be the ancestor of the popular Ongole breed."
    },

    "Gir Cow": {
        "Origin": "Gir Forest, Gujarat",
        "Primary Use": "Dairy",
        "Key Traits": "Reddish or spotted coat, long ears, arched forehead",
        "Milk Yield": "900–3000 kg per lactation",
        "Temperament": "Gentle and social",
        "Advantages": "High milk fat content and heat tolerance",
        "Limitations": "Slow growth rate in calves",
        "Diet": "Grass and dry fodder mixtures",
        "Lifespan": "20 years",
        "Climate Adaptability": "Hot and dry zones",
        "Fun Fact": "Gir cows are one of the oldest known zebu breeds and ancestors to many international dairy cattle."
    },

    "Guernsey Cow": {
        "Origin": "Channel Islands, United Kingdom",
        "Primary Use": "Dairy",
        "Key Traits": "Reddish-fawn coat with white patches",
        "Milk Yield": "6000–6500 kg per lactation",
        "Temperament": "Docile and friendly",
        "Advantages": "Produces golden-colored milk rich in beta-carotene",
        "Limitations": "Less adapted to tropical heat",
        "Diet": "Cool-climate grasses and silage",
        "Lifespan": "15–20 years",
        "Climate Adaptability": "Temperate regions",
        "Fun Fact": "Guernsey milk naturally contains a higher amount of vitamin A, giving it a golden hue."
    },

    "Hallikar Cow": {
        "Origin": "Mysuru region, Karnataka",
        "Primary Use": "Draught and agricultural work",
        "Key Traits": "Grey or white coat, muscular build",
        "Milk Yield": "227–1134 kg per lactation",
        "Temperament": "Vigorous and active",
        "Advantages": "High stamina and disease resistance",
        "Limitations": "Low milk production",
        "Diet": "Dry fodder, paddy straw, and green grass",
        "Lifespan": "14–16 years",
        "Climate Adaptability": "Tropical and semi-arid regions",
        "Fun Fact": "Hallikar is considered one of the foundations of the renowned Amritmahal breed."
    },

    "Hariana Cow": {
        "Origin": "Haryana, Punjab and Uttar Pradesh regions of India",
        "Primary Use": "Dual purpose – milk and draught",
        "Key Traits": "White to light grey color, strong legs",
        "Milk Yield": "600–800 kg per lactation",
        "Temperament": "Alert and sometimes aggressive",
        "Advantages": "Hardy and tolerant to heat and disease",
        "Limitations": "Lower milk fat percentage",
        "Diet": "Grass, fodder, and grains",
        "Lifespan": "15 years",
        "Climate Adaptability": "Hot plains of North India",
        "Fun Fact": "Hariana bulls are often used in cross-breeding to improve draught strength in other breeds."
    },

    "Holstein Friesian Cow": {
        "Origin": "Netherlands and West Friesland region",
        "Primary Use": "High-yield dairy",
        "Key Traits": "Distinct black-and-white patches, large body",
        "Milk Yield": "6000–7000 kg per lactation",
        "Temperament": "Gentle and docile",
        "Advantages": "World’s highest average milk production",
        "Limitations": "Sensitive to heat and tropical diseases",
        "Diet": "Balanced fodder and high-energy grains",
        "Lifespan": "15–18 years",
        "Climate Adaptability": "Temperate zones only",
        "Fun Fact": "Holstein Friesians are the global standard for milk yield records and dairy efficiency."
    },
    "Jaffrabadi Buffalo": {
        "Origin": "Kutch, Gujarat",
        "Primary Use": "Dual purpose – dairy and draught",
        "Key Traits": "Massive black body with white forehead markings and heavy horns",
        "Milk Yield": "2200–3000 kg per lactation",
        "Temperament": "Docile yet strong",
        "Advantages": "High milk fat content and strong body for field work",
        "Limitations": "Slow breeder, needs space for movement",
        "Diet": "Green fodder and coarse grass",
        "Lifespan": "18–22 years",
        "Climate Adaptability": "Hot, coastal, and semi-arid regions",
        "Fun Fact": "The Jaffrabadi is one of India’s heaviest buffalo breeds and a key ancestor of Mehsana buffaloes."
    },

    "Jersey Cow": {
        "Origin": "Island of Jersey, United Kingdom",
        "Primary Use": "Dairy",
        "Key Traits": "Light to dark brown coat, large eyes, compact body",
        "Milk Yield": "5000–8000 kg per lactation",
        "Temperament": "Intelligent, docile, and friendly",
        "Advantages": "High butter-fat milk and easy calving",
        "Limitations": "Prone to heat stress in tropics",
        "Diet": "Green fodder, hay, and supplements",
        "Lifespan": "18 years",
        "Climate Adaptability": "Cool to moderate climates",
        "Fun Fact": "Jersey milk’s high fat makes it perfect for rich ice-cream and cheese production."
    },

    "Kangayan Cow": {
        "Origin": "Kangayam Taluk, Tamil Nadu",
        "Primary Use": "Draught animal",
        "Key Traits": "Grey or white coat, short curved horns",
        "Milk Yield": "540–700 kg per lactation",
        "Temperament": "Active, fearless, and steady",
        "Advantages": "Excellent draught capability and heat tolerance",
        "Limitations": "Low dairy output",
        "Diet": "Grass, crop residues, and oil-cake feed",
        "Lifespan": "14–16 years",
        "Climate Adaptability": "Dry southern plains",
        "Fun Fact": "Kangayam bulls are locally called the ‘tractors of Tamil Nadu’ for their endurance in ploughing."
    },

    "Kankrej Cow": {
        "Origin": "Rann of Kutch and Jodhpur region",
        "Primary Use": "Dual purpose – milk and draught",
        "Key Traits": "Silver-grey to iron-grey coat, large hump",
        "Milk Yield": "1700–1800 kg per lactation",
        "Temperament": "Vigorous and alert",
        "Advantages": "Heat tolerant and disease resistant",
        "Limitations": "Late maturing breed",
        "Diet": "Dry fodder and desert grasses",
        "Lifespan": "16–18 years",
        "Climate Adaptability": "Hot and arid zones",
        "Fun Fact": "Kankrej cattle were among the first Indian zebu breeds exported to the USA in the 1800s."
    },

    "Kasaragod Cow": {
        "Origin": "Kasaragod district, Kerala",
        "Primary Use": "Low-input dairy and manure production",
        "Key Traits": "Dwarf size, compact body",
        "Milk Yield": "400–700 kg per lactation",
        "Temperament": "Friendly and docile",
        "Advantages": "Requires minimal feed and adapts to small farms",
        "Limitations": "Low milk yield",
        "Diet": "Household scraps and pasture grass",
        "Lifespan": "13–15 years",
        "Climate Adaptability": "Humid coastal belt",
        "Fun Fact": "Kasaragod is one of the few naturally miniature cattle breeds of South India."
    },

    "Kenkatha Cow": {
        "Origin": "Bundelkhand region, Madhya Pradesh and Uttar Pradesh",
        "Primary Use": "Draught",
        "Key Traits": "Grey to black coat, small head, compact body",
        "Milk Yield": "500–600 kg per lactation",
        "Temperament": "Active and sturdy",
        "Advantages": "Adapted to rocky, dry terrains",
        "Limitations": "Limited milk use",
        "Diet": "Dry grass and field residues",
        "Lifespan": "15 years",
        "Climate Adaptability": "Semi-arid regions",
        "Fun Fact": "Kenkatha oxen are famous for hauling heavy loads on rough village roads."
    },

    "Kherigarh Cow": {
        "Origin": "Lakhimpur Kheri district, Uttar Pradesh",
        "Primary Use": "Draught",
        "Key Traits": "White or light-grey coat, lyre-shaped horns",
        "Milk Yield": "300–500 kg per lactation",
        "Temperament": "Active and energetic",
        "Advantages": "Excellent cart-pulling power",
        "Limitations": "Poor dairy capacity",
        "Diet": "Green fodder and straw",
        "Lifespan": "14–16 years",
        "Climate Adaptability": "Gangetic plains",
        "Fun Fact": "Kherigarh cattle are locally prized for their speed in bullock races."
    },

    "Khillari Cow": {
        "Origin": "Karnataka and Maharashtra border regions",
        "Primary Use": "Draught",
        "Key Traits": "Greyish-white color, long backward horns",
        "Milk Yield": "240–515 kg per lactation",
        "Temperament": "Moderate and docile",
        "Advantages": "Extremely hardy and fast trotter",
        "Limitations": "Low milk productivity",
        "Diet": "Dry fodder, coarse grass, and grains",
        "Lifespan": "15 years",
        "Climate Adaptability": "Drought-prone areas",
        "Fun Fact": "Khillari bulls are famous for their endurance in traditional South Indian bull races."
    },

    "Krishna Valley Cow": {
        "Origin": "Krishna Valley region, Karnataka",
        "Primary Use": "Draught and moderate dairy",
        "Key Traits": "White coat, small curved horns, powerful legs",
        "Milk Yield": "750–916 kg per lactation",
        "Temperament": "Docile and friendly",
        "Advantages": "High stamina and strength for cart work",
        "Limitations": "Limited dairy capacity",
        "Diet": "Fodder, legumes, and field residues",
        "Lifespan": "15–17 years",
        "Climate Adaptability": "Hot tropical regions",
        "Fun Fact": "Krishna Valley cattle were developed by crossing local draught breeds for stronger muscle power."
    },

    "Malnad Gidda Cow": {
        "Origin": "Western Ghats of Karnataka",
        "Primary Use": "Low-input dairy and manure",
        "Key Traits": "Dwarf body, sure-footed grazer",
        "Milk Yield": "Approx. 220 kg per lactation",
        "Temperament": "Active and intelligent",
        "Advantages": "Consumes little feed yet survives harsh forest conditions",
        "Limitations": "Low milk yield",
        "Diet": "Wild grasses and shrubs",
        "Lifespan": "14–15 years",
        "Climate Adaptability": "Humid, hilly regions",
        "Fun Fact": "Malnad Gidda cows can thrive entirely on forest grazing without human care."
    },

    "Mehsana Buffalo": {
        "Origin": "Mehsana district, Gujarat",
        "Primary Use": "Dairy",
        "Key Traits": "Mostly black coat, sickle-shaped horns",
        "Milk Yield": "598–3597 kg per lactation",
        "Temperament": "Docile and cooperative",
        "Advantages": "High milk yield and fat content",
        "Limitations": "Needs regular watering and shade",
        "Diet": "Fodder, cottonseed cake, and grass",
        "Lifespan": "20 years",
        "Climate Adaptability": "Dry and semi-arid zones",
        "Fun Fact": "Mehsana buffaloes are a genetic blend of Murrah and Surti breeds."
    },

    "Murrah Buffalo": {
        "Origin": "Haryana and Punjab regions of India",
        "Primary Use": "High-volume dairy",
        "Key Traits": "Jet-black coat, short tightly-curved horns",
        "Milk Yield": "2500–3600 kg per lactation",
        "Temperament": "Calm and intelligent",
        "Advantages": "Produces rich 7–8 % fat milk",
        "Limitations": "Sensitive to cold climates",
        "Diet": "Green fodder and cottonseed",
        "Lifespan": "20 years",
        "Climate Adaptability": "Hot and humid plains",
        "Fun Fact": "Murrah buffaloes are nicknamed the ‘Black Gold’ of India for their prolific milk yield."
    },

    "Nagori Cow": {
        "Origin": "Nagaur district, Rajasthan",
        "Primary Use": "Draught",
        "Key Traits": "Light grey coat, long narrow face, long legs",
        "Milk Yield": "479–905 kg per lactation",
        "Temperament": "Active and agile",
        "Advantages": "Fast walker and efficient plough animal",
        "Limitations": "Moderate milk producer",
        "Diet": "Dry fodder and grasses",
        "Lifespan": "14–16 years",
        "Climate Adaptability": "Hot arid regions",
        "Fun Fact": "Nagori cattle were once essential for transporting goods across the Thar Desert."
    },

    "Nagpuri Buffalo": {
        "Origin": "Vidarbha region, Maharashtra",
        "Primary Use": "Dual purpose – draught and dairy",
        "Key Traits": "Black coat with white facial patches",
        "Milk Yield": "700–1200 kg per lactation",
        "Temperament": "Docile and mild",
        "Advantages": "Good working capacity and milk yield",
        "Limitations": "Performs poorly in humid areas",
        "Diet": "Coarse grass and hay",
        "Lifespan": "17–18 years",
        "Climate Adaptability": "Hot dry plateau",
        "Fun Fact": "Nagpuri buffaloes are famed for pulling heavy carts in central India’s heat."
    },

    "Nili Ravi Buffalo": {
        "Origin": "Punjab region of India and Pakistan",
        "Primary Use": "Dairy",
        "Key Traits": "Small horns, ‘wall-eyes’, deep black coat",
        "Milk Yield": "1500–2500 kg per lactation",
        "Temperament": "Docile and social",
        "Advantages": "Excellent milk fat and calm nature",
        "Limitations": "Needs ample water and shade",
        "Diet": "Green fodder, sugarcane tops, and concentrates",
        "Lifespan": "18–20 years",
        "Climate Adaptability": "Hot and humid plains",
        "Fun Fact": "Nili Ravi buffaloes hold world records for highest milk yield among buffalo breeds."
    },
    "Nimari Cow": {
        "Origin": "Nimar region, Madhya Pradesh",
        "Primary Use": "Labour and mild dairy",
        "Key Traits": "Red or copper coat, muscular body, strong hooves",
        "Milk Yield": "600–954 kg per lactation",
        "Temperament": "Active and agile",
        "Advantages": "Hardy and disease-resistant",
        "Limitations": "Low milk productivity",
        "Diet": "Dry grass and crop residues",
        "Lifespan": "14–16 years",
        "Climate Adaptability": "Hot and semi-arid zones",
        "Fun Fact": "Nimari cattle are descendants of Gir and local breeds, prized for their endurance."
    },

    "Ongole Cow": {
        "Origin": "Prakasam district, Andhra Pradesh",
        "Primary Use": "Labour and dairy",
        "Key Traits": "Glossy white coat, broad forehead, muscular frame",
        "Milk Yield": "900–1200 kg per lactation",
        "Temperament": "Trainable and alert",
        "Advantages": "Strong and heat-tolerant, exported worldwide",
        "Limitations": "Slow milking process",
        "Diet": "Grass, oil cakes, and grains",
        "Lifespan": "16–18 years",
        "Climate Adaptability": "Tropical and semi-arid regions",
        "Fun Fact": "Ongole bulls contributed to the famous ‘Brahman’ breed in the USA."
    },

    "Pulikulam Cow": {
        "Origin": "Tamil Nadu",
        "Primary Use": "Farm labour and sports",
        "Key Traits": "Small size, long curved horns, agile",
        "Milk Yield": "600–800 kg per lactation",
        "Temperament": "Furious and active",
        "Advantages": "Highly alert and suited for rugged terrain",
        "Limitations": "Difficult to handle during heat cycles",
        "Diet": "Wild grass and dry fodder",
        "Lifespan": "14 years",
        "Climate Adaptability": "Hot southern plains",
        "Fun Fact": "Pulikulam cattle are popularly used in Tamil Nadu’s traditional Jallikattu sport."
    },

    "Red Dane Cow": {
        "Origin": "Denmark",
        "Primary Use": "Dairy and beef",
        "Key Traits": "Solid red coat, strong hooves, muscular build",
        "Milk Yield": "3000–4000 kg per lactation",
        "Temperament": "Calm and hardy",
        "Advantages": "Dual purpose and cold-resistant",
        "Limitations": "Requires nutrient-rich feed",
        "Diet": "Fodder, hay, and legumes",
        "Lifespan": "15–18 years",
        "Climate Adaptability": "Temperate and cool regions",
        "Fun Fact": "Red Dane cows helped shape many modern hybrid dairy breeds."
    },

    "Red Sindi Cow": {
        "Origin": "Karachi and Hyderabad regions",
        "Primary Use": "Dairy",
        "Key Traits": "Deep red coat, compact body, broad forehead",
        "Milk Yield": "1100–2600 kg per lactation",
        "Temperament": "Extremely docile",
        "Advantages": "High-fat milk and heat tolerance",
        "Limitations": "Slow milker",
        "Diet": "Grass and oil cakes",
        "Lifespan": "15–17 years",
        "Climate Adaptability": "Hot dry plains",
        "Fun Fact": "Red Sindhi cattle were the first Indian dairy breed exported globally for crossbreeding."
    },

    "Sahiwal Cow": {
        "Origin": "Sahiwal district, Pakistan",
        "Primary Use": "Dairy",
        "Key Traits": "Reddish-brown coat, long drooping ears",
        "Milk Yield": "1400–2500 kg per lactation",
        "Temperament": "Docile and calm",
        "Advantages": "Excellent dairy output and disease resistance",
        "Limitations": "Slower breeder",
        "Diet": "Green fodder, oil cakes, and legumes",
        "Lifespan": "18 years",
        "Climate Adaptability": "Hot tropical regions",
        "Fun Fact": "Sahiwal is known as the ‘Holstein of the Tropics’ for its high milk yield in harsh climates."
    },

    "Surti Buffalo": {
        "Origin": "Charotar region, Gujarat",
        "Primary Use": "Dairy",
        "Key Traits": "Silver-grey coat, sickle-shaped horns",
        "Milk Yield": "1600–1800 kg per lactation",
        "Temperament": "Docile and gentle",
        "Advantages": "High milk fat and early maturity",
        "Limitations": "Low draught ability",
        "Diet": "Grass, maize stalks, and grains",
        "Lifespan": "17–18 years",
        "Climate Adaptability": "Humid plains",
        "Fun Fact": "Surti buffalo milk is famous for producing rich, creamy ghee."
    },

    "Tharparkar Cow": {
        "Origin": "Thar Desert, Sindh and Rajasthan",
        "Primary Use": "Dairy and draught",
        "Key Traits": "White or grey coat, medium frame, long horns",
        "Milk Yield": "913–2147 kg per lactation",
        "Temperament": "Docile and gentle",
        "Advantages": "Highly heat-tolerant and disease-resistant",
        "Limitations": "Needs open grazing area",
        "Diet": "Desert grass and crop residues",
        "Lifespan": "15–18 years",
        "Climate Adaptability": "Arid and semi-arid",
        "Fun Fact": "Tharparkar cattle are nicknamed the ‘White Sindhi’ due to their bright coat."
    },

    "Toda Buffalo": {
        "Origin": "Nilgiri Hills, Tamil Nadu",
        "Primary Use": "Dairy and ceremonial use",
        "Key Traits": "Ash-grey coat, crescent-shaped horns",
        "Milk Yield": "800–1200 kg per lactation",
        "Temperament": "Aggressive yet loyal to owners",
        "Advantages": "Produces high-fat milk suited for butter",
        "Limitations": "Limited distribution and low breeding rate",
        "Diet": "Forest grass and herbs",
        "Lifespan": "16–18 years",
        "Climate Adaptability": "Cool hilly regions",
        "Fun Fact": "Toda buffaloes are sacred to the Toda tribe, and their milk is used in temple rituals."
    },

    "Umblachery Cow": {
        "Origin": "Cauvery delta region, Tamil Nadu",
        "Primary Use": "Draught",
        "Key Traits": "White star on forehead, white socks on legs",
        "Milk Yield": "470–725 kg per lactation",
        "Temperament": "Docile and calm",
        "Advantages": "Excellent for ploughing in wet fields",
        "Limitations": "Low dairy output",
        "Diet": "Grass and crop residues",
        "Lifespan": "15 years",
        "Climate Adaptability": "Tropical coastal regions",
        "Fun Fact": "Umblachery cattle are called ‘Jersey of Tamil Nadu’ for their graceful appearance."
    },

    "Vechur Cow": {
        "Origin": "Kottayam district, Kerala",
        "Primary Use": "Dairy",
        "Key Traits": "Smallest cattle breed in the world, dwarf size",
        "Milk Yield": "1062–2810 kg per lactation",
        "Temperament": "Gentle and docile",
        "Advantages": "Requires minimal feed, easy to maintain",
        "Limitations": "Small frame limits draught use",
        "Diet": "Grass, banana stems, and paddy straw",
        "Lifespan": "14–15 years",
        "Climate Adaptability": "Humid tropical regions",
        "Fun Fact": "The Vechur cow holds the Guinness World Record for being the world’s smallest cattle breed."
    }
}

# ===== MODEL LOADING =====
@st.cache_resource
def load_model(model_path):
    """Load the pre-trained ResNet-50 model"""
    try:
        if not os.path.exists(model_path):
            return None, None

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_state = checkpoint['model_state']
        classes = checkpoint['classes']

        # Define model structure
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(model_state)
        model.eval()

        return model, classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# ===== PREDICTION FUNCTION =====
def predict(image, model_path='best_model.pth'):
    """Predict cattle breed from image"""
    try:
        # Load model + class names
        model, classes = load_model(model_path)
        if model is None or classes is None:
            return {"error": "Model file not found or corrupted."}

        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = image.convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_breed = classes[predicted_idx.item()]

        # Top predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(5, len(classes)))
        top_predictions = [
            {"breed": classes[idx.item()], "score": prob.item()}
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]

        return {
            "prediction": predicted_breed,
            "confidence": confidence.item(),
            "top_n_predictions": top_predictions
        }

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


# ===== CUSTOM CSS STYLING =====
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: fadeIn 0.8s ease-in;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        animation: slideUp 0.6s ease-out;
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        animation: scaleIn 0.5s ease-out;
    }
    
    .prediction-box h1 {
        font-size: 2.5rem;
        margin: 1rem 0;
        font-weight: 700;
    }
    
    .prediction-box h2 {
        font-size: 1.5rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .prediction-box h3 {
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        opacity: 0.8;
        font-weight: 400;
    }
    
    /* Top Prediction Cards */
    .top-pred-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    /* Fun Fact Card */
    .fun-fact-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.3);
        animation: slideUp 0.6s ease-out;
    }
    
    .fun-fact-card h4 {
        color: #333;
        margin: 0 0 0.8rem 0;
        font-size: 1.2rem;
    }
    
    .fun-fact-card p {
        color: #555;
        margin: 0;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        height: 4rem;
        font-size: 1.3rem;
        font-weight: 600;
        border-radius: 15px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(102, 126, 234, 0.5);
    }
    
    /* Info Box Styles */
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .info-box h4 {
        color: #333;
        margin-top: 0;
        font-weight: 600;
    }
    
    .info-box p {
        color: #555;
        margin-bottom: 0;
        line-height: 1.6;
    }
    
    /* Breed Info Tabs */
    .breed-detail-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        color: #333;
        font-size: 2rem;
        margin: 0.5rem 0;
        font-weight: 700;
    }
    
    .metric-card p {
        color: #666;
        margin: 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input Method Selector */
    .input-method-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File Uploader */
    .uploadedFile {
        border-radius: 15px;
        border: 2px dashed #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: black;
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Image Container */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        margin: 1rem 0;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ===== INITIALIZE SESSION STATE =====
if 'result' not in st.session_state:
    st.session_state['result'] = None

# ===== HEADER =====
st.markdown("""
    <div class='main-header'>
        <h1>🐄 Indian Cattle Breed Classifier 🐃</h1>
        <p>AI-Powered Recognition of Indigenous Indian Cattle Breeds</p>
    </div>
""", unsafe_allow_html=True)

# ===== MAIN CONTENT =====
col1, col2 = st.columns([1, 1], gap="large")

# LEFT COLUMN - IMAGE INPUT
with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    # ===== INPUT METHOD SELECTION =====
    st.markdown("""
        <div class='input-method-card'>
            <h3 style='margin: 0; font-size: 1.3rem;'>📷 Choose Input Method</h3>
        </div>
    """, unsafe_allow_html=True)
    
    input_method = st.radio(
        "Select how you want to provide the image:",
        ["📤 Upload Image File", "📸 Use Live Webcam"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    image = None
    
    # ===== WEBCAM INPUT =====
    if input_method == "📸 Use Live Webcam":
        st.markdown("### 📸 Capture from Webcam")
        st.info("💡 Position the cattle in frame and click 'Take Photo'")
        
        camera_photo = st.camera_input("Take a photo", label_visibility="collapsed")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, caption='📸 Captured Image', use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Image info
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.markdown(f"**Width:** {image.size[0]}px")
            with col_i2:
                st.markdown(f"**Height:** {image.size[1]}px")
            with col_i3:
                st.markdown(f"**Format:** {image.format if image.format else 'JPEG'}")
    
    # ===== FILE UPLOAD INPUT =====
    else:
        st.markdown("### 📤 Upload Cattle Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, well-lit image for best results",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, caption='📸 Uploaded Image', use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Image info
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.markdown(f"**Width:** {image.size[0]}px")
            with col_i2:
                st.markdown(f"**Height:** {image.size[1]}px")
            with col_i3:
                st.markdown(f"**Format:** {image.format}")
    
    # ===== FUN FACT DISPLAY =====
    if image is not None and 'result' in st.session_state and st.session_state['result']:
        predicted_breed = st.session_state['result']['prediction']
        if predicted_breed in BREED_INFO:
            st.markdown(f"""
                <div class='fun-fact-card'>
                    <h4>💡 Fun Fact about {predicted_breed}</h4>
                    <p>{BREED_INFO[predicted_breed]['Fun Fact']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # ===== PREDICT BUTTON =====
    if image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔍 Identify Breed Now", type="primary")
        
        if predict_button:
            with st.spinner('🔄 Analyzing image with AI... Please wait...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                try:
                    # Call prediction function
                    result = predict(image, model_path='best_model.pth')
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.session_state['result'] = result
                        st.success("✅ Analysis Complete!")
                        time.sleep(0.5)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Please ensure 'best_model.pth' is in the same directory.")
    else:
        st.markdown("""
            <div style='text-align: center; padding: 3rem 1rem;'>
                <img src='https://cdn-icons-png.flaticon.com/512/3588/3588592.png' width='200'>
                <p style='font-size: 1.1rem; color: #666; margin-top: 1.5rem;'>
                    👆 Select an input method and provide an image to get started
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT COLUMN - RESULTS
with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Results")
    
    if 'result' in st.session_state and st.session_state['result']:
        result = st.session_state['result']
        
        # Main prediction display
        st.markdown(f"""
            <div class='prediction-box'>
                <h3>🎯 Identified Breed</h3>
                <h1>{result['prediction']}</h1>
                <h2>Confidence: {result['confidence']*100:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence indicator with badges
        confidence_level = result['confidence']
        if confidence_level > 0.8:
            st.markdown("<div class='confidence-badge confidence-high'>🎯 High Confidence Prediction</div>", unsafe_allow_html=True)
        elif confidence_level > 0.6:
            st.markdown("<div class='confidence-badge confidence-medium'>⚠️ Moderate Confidence</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='confidence-badge confidence-low'>❗ Low Confidence</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Top 3 predictions
        st.markdown("#### 🏆 Top 3 Predictions")
        
        for i, pred in enumerate(result['top_n_predictions'][:3], 1):
            confidence_pct = pred['score'] * 100
            
            # Medal emojis
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            
            st.markdown(f"""
                <div class='top-pred-card'>
                    <h4 style='margin: 0 0 0.5rem 0; color: white;'>
                        {medals[i]} {pred['breed']}
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.progress(pred['score'])
            with col_b:
                st.markdown(f"**{confidence_pct:.1f}%**")
            
            if i < 3:
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='text-align: center; padding: 3rem 1rem;'>
                <img src='https://cdn-icons-png.flaticon.com/512/3588/3588592.png' width='180'>
                <p style='font-size: 1.1rem; color: #666; margin-top: 1.5rem;'>
                    🔍 Results will appear here after prediction
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===== BREED INFORMATION SECTION =====
if 'result' in st.session_state and st.session_state['result']:
    st.markdown("<br>", unsafe_allow_html=True)
    
    predicted_breed = st.session_state['result']['prediction']
    
    st.markdown(f"""
        <div class='glass-card'>
            <h2 style='color: #667eea; margin-top: 0;'>📖 Detailed Information: {predicted_breed}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if predicted_breed in BREED_INFO:
        breed_data = BREED_INFO[predicted_breed]
        
        # Key Metrics at Top
        st.markdown("<br>", unsafe_allow_html=True)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);'>
                    <p>🥛 MILK YIELD</p>
                    <h3>{breed_data["Milk Yield"].split()[0] if breed_data["Milk Yield"] != "N/A" else "N/A"}</h3>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);'>
                    <p>⏳ LIFESPAN</p>
                    <h3>{breed_data["Lifespan"].split()[0] if breed_data["Lifespan"] != "N/A" else "N/A"}</h3>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            animal_type = "Buffalo" if "Buffalo" in predicted_breed else "Cow"
            st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);'>
                    <p>🐄 TYPE</p>
                    <h3>{animal_type}</h3>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);'>
                    <p>😊 NATURE</p>
                    <h3>{breed_data["Temperament"].split()[0]}</h3>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs for organized information
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Overview", "🎯 Characteristics", "📊 Detailed Metrics", "🌟 Special Info"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div class='breed-detail-card'>
                        <h4 style='color: #667eea; margin-top: 0;'>🌍 Origin</h4>
                        <p style='color: #555; line-height: 1.8;'>{breed_data["Origin"]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='breed-detail-card'>
                        <h4 style='color: #667eea; margin-top: 0;'>🎯 Primary Use</h4>
                        <p style='color: #555; line-height: 1.8;'>{breed_data["Primary Use"]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='breed-detail-card'>
                        <h4 style='color: #667eea; margin-top: 0;'>😊 Temperament</h4>
                        <p style='color: #555; line-height: 1.8;'>{breed_data["Temperament"]}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='breed-detail-card'>
                        <h4 style='color: #667eea; margin-top: 0;'>🌡️ Climate Adaptability</h4>
                        <p style='color: #555; line-height: 1.8;'>{breed_data["Climate Adaptability"]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='breed-detail-card'>
                        <h4 style='color: #667eea; margin-top: 0;'>🍽️ Diet</h4>
                        <p style='color: #555; line-height: 1.8;'>{breed_data["Diet"]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='breed-detail-card'>
                        <h4 style='color: #667eea; margin-top: 0;'>⏳ Lifespan</h4>
                        <p style='color: #555; line-height: 1.8;'>{breed_data["Lifespan"]}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown(f"""
                <div class='breed-detail-card' style='background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); color: white;'>
                    <h4 style='color: white; margin-top: 0;'>🔍 Key Physical Traits</h4>
                    <p style='color: white; line-height: 1.8; font-size: 1.05rem;'>{breed_data['Key Traits']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div class='breed-detail-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white;'>
                        <h4 style='color: white; margin-top: 0;'>✅ Advantages</h4>
                        <p style='color: white; line-height: 1.8;'>{breed_data["Advantages"]}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='breed-detail-card' style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white;'>
                        <h4 style='color: white; margin-top: 0;'>⚠️ Limitations</h4>
                        <p style='color: white; line-height: 1.8;'>{breed_data["Limitations"]}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            # Detailed table
            st.markdown("#### 📋 Complete Breed Profile")
            
            info_data = {
                "🌍 Origin": breed_data["Origin"],
                "🎯 Primary Use": breed_data["Primary Use"],
                "🥛 Milk Yield": breed_data["Milk Yield"],
                "😊 Temperament": breed_data["Temperament"],
                "🌡️ Climate": breed_data["Climate Adaptability"],
                "🍽️ Diet": breed_data["Diet"],
                "⏳ Lifespan": breed_data["Lifespan"],
                "🔍 Key Traits": breed_data["Key Traits"],
                "✅ Advantages": breed_data["Advantages"],
                "⚠️ Limitations": breed_data["Limitations"]
            }
            
            for key, value in info_data.items():
                st.markdown(f"""
                    <div class='breed-detail-card'>
                        <h5 style='color: #667eea; margin: 0 0 0.5rem 0;'>{key}</h5>
                        <p style='color: #555; margin: 0; line-height: 1.6;'>{value}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            color: white; padding: 3rem 2rem; border-radius: 20px; 
                            box-shadow: 0 10px 40px rgba(240, 147, 251, 0.4);'>
                    <h2 style='margin-top:0; text-align: center;'>💡 Did You Know?</h2>
                    <p style='margin-bottom:0; line-height: 1.8; font-size: 1.15rem; text-align: center;'>
                        {breed_data['Fun Fact']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Additional interesting section
            st.markdown("""
                <div class='breed-detail-card'>
                    <h4 style='color: #667eea; margin-top: 0;'>🌾 Cultural Significance</h4>
                    <p style='color: #555; line-height: 1.8;'>
                        Indian cattle breeds are deeply rooted in the country's agricultural heritage 
                        and hold significant cultural importance. They are adapted to local climatic 
                        conditions and play a crucial role in sustainable farming practices.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Detailed information not available for this breed.")

# ===== FOOTER =====
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 20px; text-align: center; color: white;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);'>
        <h3 style='margin: 0 0 1rem 0; font-size: 1.8rem;'>
            🐄 Indian Cattle Breed Classifier 🐃
        </h3>
        <p style='margin: 0; opacity: 0.9; font-size: 1.05rem;'>
            Powered by Deep Learning • ResNet-50 Architecture • PyTorch Framework
        </p>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.8;'>
            Preserving Indian Agricultural Heritage through AI Technology
        </p>
    </div>
""", unsafe_allow_html=True)