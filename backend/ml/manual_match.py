import json, sys
from service import match
profile = {
  "vegan": True, "region": "SouthAsia", "age": 42,
  "cuisines": {"Indian": .8, "Japanese": .2},
  "flavors": {"spicy": .9, "umami": .7},
  "textures": {"tender": .7, "crispy": .3},
  "top_k": 5
}
print(json.dumps(match(profile, top_k=5), indent=2))    