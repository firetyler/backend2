import random
import json

# Alternativ och kategorier
alternativ = {
    1: ["positiv", "negativ", "neutral"],
    2: ["Elektronik", "Kläder", "Böcker", "Hem & Trädgård", "Sport & Fritid"],
    3: [],  # Fråga-svar-par (skapas nedan)
    4: ["Retur", "Leverans", "Produktfråga", "Betalning", "Support"],
    5: ["svenska", "engelska", "spanska", "tyska", "franska"],
    6: ["spam", "ej spam"]
}

# Exempeldata
texter = [
    "Det här är den bästa produkten jag någonsin köpt.",
    "Varför fungerar inte min orderstatuslänk?",
    "Hur lång är leveranstiden till Sverige?",
    "Fantastisk service, tack så mycket!",
    "Produkten gick sönder efter en vecka.",
    "Hej, kan jag få tillbaka mina pengar?",
    "Where is my package?",
    "Este producto es increíble.",
    "Ich brauche Hilfe mit meiner Bestellung.",
    "Livraison rapide et sans problème."
]

svar = [
    "Din beställning behandlas just nu.",
    "Du kan returnera produkten inom 30 dagar.",
    "Vi beklagar besväret, vi återkommer så snart vi kan.",
    "Produkten bör levereras inom 3–5 arbetsdagar.",
    "Vänligen kontakta vår support för vidare hjälp.",
    "Tack för att du kontaktade oss!"
]

# Skapa 10 000 dataposter
data = []
for _ in range(10000):
    item = {}

    # 1. Textklassificering
    item["text_classification"] = {
        "text": random.choice(texter),
        "label": random.choice(alternativ[1])
    }

    # 2. Produktdata
    item["product_data"] = {
        "title": f"Produkt {random.randint(1000,9999)}",
        "category": random.choice(alternativ[2]),
        "price": round(random.uniform(10, 5000), 2),
        "description": random.choice(texter)
    }

    # 3. Chattdata
    item["chat_data"] = {
        "question": random.choice(texter),
        "answer": random.choice(svar)
    }

    # 4. FAQ
    item["faq"] = {
        "question": random.choice(texter),
        "category": random.choice(alternativ[4])
    }

    # 5. Språkidentifiering
    item["language_detection"] = {
        "text": random.choice(texter),
        "language": random.choice(alternativ[5])
    }

    # 6. Spamklassificering
    item["spam_detection"] = {
        "message": random.choice(texter),
        "label": random.choice(alternativ[6])
    }

    data.append(item)

# Spara till JSON-fil
with open("training_data_10000.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("JSON-fil skapad: training_data_10000.json")
