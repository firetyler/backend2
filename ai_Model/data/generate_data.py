import random
import json

# Options and Categories
options = {
    1: ["positive", "negative", "neutral"],
    2: ["Electronics", "Clothing", "Books", "Home & Garden", "Sports & Leisure"],
    3: [],  # Question-answer pairs (created below)
    4: ["Return", "Delivery", "Product Inquiry", "Payment", "Support"],
    5: ["Swedish", "English", "Spanish", "German", "French"],
    6: ["spam", "not spam"]
}

# Sample data
texts = [
    "This is the best product I have ever bought.",
    "Why isn't my order status link working?",
    "How long is the delivery time to Sweden?",
    "Fantastic service, thank you so much!",
    "The product broke after a week.",
    "Hi, can I get my money back?",
    "Where is my package?",
    "This product is amazing.",
    "I need help with my order.",
    "Fast and hassle-free delivery."
]

answers = [
    "Your order is currently being processed.",
    "You can return the product within 30 days.",
    "We apologize for the inconvenience, we will get back to you as soon as we can.",
    "The product should be delivered within 3–5 business days.",
    "Please contact our support for further assistance.",
    "Thank you for reaching out to us!"
]

# Create 10,000 data entries
data = []
for _ in range(10000):
    item = {}

    # 1. Text classification
    item["text_classification"] = {
        "text": random.choice(texts),
        "label": random.choice(options[1])
    }

    # 2. Product data
    item["product_data"] = {
        "title": f"Product {random.randint(1000,9999)}",
        "category": random.choice(options[2]),
        "price": round(random.uniform(10, 5000), 2),
        "description": random.choice(texts)
    }

    # 3. Chat data
    item["chat_data"] = {
        "question": random.choice(texts),
        "answer": random.choice(answers)
    }

    # 4. FAQ
    item["faq"] = {
        "question": random.choice(texts),
        "category": random.choice(options[4])
    }

    # 5. Language detection
    item["language_detection"] = {
        "text": random.choice(texts),
        "language": random.choice(options[5])
    }

    # 6. Spam classification
    item["spam_detection"] = {
        "message": random.choice(texts),
        "label": random.choice(options[6])
    }

    data.append(item)

# Save to JSON file
with open("training_data_10000_en.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("JSON file created: training_data_10000_en.json")
