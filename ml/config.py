# domain vocab + priors

CUISINES = [
    "Italian","Mexican","Japanese","Indian","Greek","American","Thai","Chinese",
    "Mediterranean","Korean","French","Vietnamese","Lebanese","Turkish",
    "Spanish","Moroccan","Ethiopian","Brazilian","Nordic","Caribbean"
]
FLAVORS = [
    "sweet","spicy","savory","sour","bitter","umami",
    "salty","fresh","acidic","smoky","herbal","earthy"
]
TEXTURES = [
    "crispy","chewy","creamy","crunchy","silky","tender","flaky",
    "juicy","firm","smooth","gooey","fatty","fibrous","brittle"
]

REGIONS = ["USA","Europe","SouthAsia","EastAsia","MiddleEast","LatAm"]

# Priors encode correlations you believe are plausible
PRIORS = {

    # -------------------------------
    # 1️⃣ Lifestyle / dietary correlations
    # -------------------------------
    "vegan_cuisine_boost": {
        "Indian": 0.25, "Mediterranean": 0.25, "Thai": 0.2,
        "Vietnamese": 0.2, "Ethiopian": 0.20, "Nordic": 0.10,
        "Lebanese": 0.25, "Turkish": 0.15
    },

    "vegan_flavor_adjust": {
        "umami": -0.10, "savory": +0.05, "spicy": +0.05,
        "fresh": +0.10, "earthy": +0.10
    },

    "low_carb_cuisine_boost": {
        "Mediterranean": 0.25, "Nordic": 0.20, "Japanese": 0.15, "American": 0.10
    },

    "health_focus_flavor_adjust": {
        "sweet": -0.10, "fresh": +0.15, "herbal": +0.10, "acidic": +0.05
    },

    # -------------------------------
    # 2️⃣ Age-based perceptual shifts
    # -------------------------------
    "age>50_texture_adjust": {
        "crispy": -0.15, "tender": +0.10, "creamy": +0.05, "smooth": +0.05, "chewy": -0.10
    },

    "age>50_flavor_adjust": {
        "savory": +0.10, "sweet": -0.05, "spicy": -0.05, "umami": +0.05, "bitter": -0.05
    },

    "age<25_flavor_adjust": {
        "spicy": +0.10, "sweet": +0.10, "bitter": -0.05, "earthy": -0.05, "salty": +0.10
    },

    # -------------------------------
    # 3️⃣ Region preference biases
    # -------------------------------
    "region_bias": {
        "USA": {"American": +0.20, "Mexican": +0.15, "Italian": +0.10, "Japanese": +0.05, "Chinese": +0.10},
        "Europe": {"Italian": +0.20, "French": +0.15, "Mediterranean": +0.15, "Nordic": +0.10, "Greek": +0.10, "Turkish": +0.05},
        "SouthAsia": {"Indian": +0.30, "Thai": +0.10, "Chinese": +0.05},
        "EastAsia": {"Japanese": +0.15, "Korean": +0.15, "Chinese": +0.20, "Vietnamese": +0.10},
        "MiddleEast": {"Lebanese": +0.25, "Turkish": +0.25, "Moroccan": +0.15, "Mediterranean": +0.10},
        "LatAm": {"Mexican": +0.30, "Brazilian": +0.20, "Caribbean": +0.15}
    },

    # -------------------------------
    # 4️⃣ Cuisines' internal flavor priors
    # -------------------------------
    "cuisine_flavor_profile": {
        "Indian": {"spicy": .85, "savory": .75, "sweet": .30, "umami": .50, "earthy": .45},
        "Mediterranean": {"savory": .75, "fresh": .55, "umami": .45, "sour": .35, "herbal": .55},
        "Italian": {"savory": .80, "umami": .65, "sweet": .45, "acidic": .35},
        "Mexican": {"spicy": .75, "savory": .70, "sour": .40, "smoky": .40, "sweet": .35},
        "Japanese": {"umami": .85, "savory": .70, "sweet": .30, "fresh": .45, "acidic": .25},
        "Thai": {"spicy": .80, "sweet": .60, "sour": .60, "savory": .60, "fresh": .40},
        "Chinese": {"umami": .70, "savory": .70, "sweet": .40, "spicy": .45},
        "Korean": {"spicy": .65, "umami": .65, "savory": .70, "sweet": .35},
        "French": {"savory": .75, "umami": .50, "sweet": .50, "buttery": .35},
        "Vietnamese": {"sour": .55, "savory": .65, "sweet": .45, "fresh": .60, "herbal": .55},
        "American": {"savory": .60, "sweet": .55, "smoky": .45, "salty": .55},
        "Lebanese": {"savory": .65, "fresh": .55, "herbal": .60, "acidic": .40, "spicy": .30},
        "Turkish": {"savory": .70, "sweet": .45, "spicy": .45, "umami": .55},
        "Moroccan": {"spicy": .55, "sweet": .45, "earthy": .60, "savory": .65, "herbal": .35},
        "Ethiopian": {"spicy": .80, "earthy": .60, "savory": .70},
        "Brazilian": {"savory": .65, "smoky": .55, "sweet": .40, "spicy": .35},
        "Nordic": {"fresh": .65, "savory": .50, "acidic": .45, "earthy": .35},
        "Caribbean": {"spicy": .70, "sweet": .60, "smoky": .50, "fresh": .40},
        "Greek": {"savory": .70, "fresh": .60, "herbal": .55, "acidic": .45, "sweet": .35},
        "Spanish": {"savory": .70, "smoky": .50, "sweet": .40, "fresh": .45, "spicy": .35}
    },

    
    # -------------------------------
    # 5️⃣ Cuisine → texture tendencies
    # -------------------------------
    "cuisine_texture_profile": {
        "Japanese": {"tender": .55, "silky": .50, "smooth": .45},
        "Indian": {"creamy": .55, "tender": .50, "juicy": .45},
        "French": {"creamy": .70, "silky": .60, "tender": .55},
        "American": {"crispy": .60, "chewy": .50, "juicy": .45, "smooth": .40},
        "Mexican": {"crispy": .55, "chewy": .55, "juicy": .45, "tender": .40},
        "Thai": {"tender": .55, "juicy": .50, "crunchy": .45},
        "Mediterranean": {"smooth": .50, "tender": .50, "flaky": .40},
        "Korean": {"chewy": .60, "crispy": .55, "juicy": .45},
        "Chinese": {"chewy": .55, "smooth": .50, "juicy": .45},
        "Nordic": {"firm": .55, "tender": .50, "smooth": .45},
        "Greek": {"flaky": .50, "tender": .50, "creamy": .45},
        "Spanish": {"crispy": .55, "tender": .50, "juicy": .45},
        "Lebanese": {"tender": .50, "flaky": .45, "creamy": .40},
        "Turkish": {"tender": .50, "flaky": .45, "smooth": .40},
        "Moroccan": {"tender": .50, "juicy": .45, "fibrous": .35},
        "Ethiopian": {"smooth": .50, "creamy": .45, "tender": .45},
        "Brazilian": {"juicy": .55, "chewy": .50, "crispy": .45},
        "Caribbean": {"juicy": .55, "crispy": .50, "tender": .45}
    },


    # -------------------------------
    # 6️⃣ Cross-flavor and texture correlations
    # -------------------------------
    "flavor_cooccurrence": {
        "spicy": {"savory": +0.35, "umami": +0.25, "sweet": +0.10},
        "sweet": {"creamy": +0.25, "smooth": +0.15, "spicy": +0.10, "smoky": +0.15},
        "umami": {"creamy": +0.20, "savory": +0.40},
        "fresh": {"herbal": +0.40, "acidic": +0.30, "sweet": +0.05},
        "smoky": {"savory": +0.35, "juicy": +0.25, "sweet": +0.15}
    },

    "texture_cooccurrence": {
        "crispy": {"crunchy": +0.40, "chewy": -0.20},
        "creamy": {"smooth": +0.35, "fatty": +0.20},
        "tender": {"juicy": +0.30, "flaky": +0.20},
        "chewy": {"firm": +0.25, "crispy": -0.15}
    }
}