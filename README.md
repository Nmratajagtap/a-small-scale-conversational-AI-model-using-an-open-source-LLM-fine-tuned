# a-small-scale-conversational-AI-model-using-an-open-source-LLM-fine-tuned
Build a small-scale conversational AI model using an open-source LLM fine-tuned for one Indian language + English, demonstrating intent detection and context retention. Submit code, README, and sample outputs.
# 9. Test the function with example conversational inputs

if model and tokenizer:
    print("\n--- Testing the response generation function ---")

    # Test Case 1: Simple greeting (Intent detection)
    print("\nTest Case 1: Simple greeting")
    history1 = []
    new_input1 = "Hello!"
    response1 = generate_response(history1, new_input1, model, tokenizer, device)
    print(f"User: {new_input1}")
    print(f"Assistant: {response1}")

    # Test Case 2: Train ticket booking (Intent detection + Context retention - Destination/Origin)
    print("\nTest Case 2: Train ticket booking (Intent + Context)")
    history2 = [
        {"role": "user", "text": "मुझे दिल्ली से मुंबई के लिए ट्रेन टिकट बुक करना है।"}
    ]
    new_input2 = "किस तारीख के लिए?" # Hindi: For which date?
    response2 = generate_response(history2, new_input2, model, tokenizer, device)
    print(f"Conversation History: {history2}")
    print(f"User: {new_input2}")
    print(f"Assistant: {response2}")

    # Test Case 3: Train ticket booking (Context retention - Date)
    print("\nTest Case 3: Train ticket booking (Context - Date)")
    history3 = [
        {"role": "user", "text": "मुझे दिल्ली से मुंबई के लिए ट्रेन टिकट बुक करना है।"},
        {"role": "assistant", "text": "ज़रूर, किस तारीख के लिए टिकट चाहिए?"}
    ]
    new_input3 = "अगले सोमवार के लिए।" # Hindi: For next Monday.
    response3 = generate_response(history3, new_input3, model, tokenizer, device)
    print(f"Conversation History: {history3}")
    print(f"User: {new_input3}")
    print(f"Assistant: {response3}")

    # Test Case 4: Weather query (Intent detection)
    print("\nTest Case 4: Weather query (Intent)")
    history4 = []
    new_input4 = "What is the weather like in Paris today?"
    response4 = generate_response(history4, new_input4, model, tokenizer, device)
    print(f"User: {new_input4}")
    print(f"Assistant: {response4}")

    # Test Case 5: Weather query (Context retention - City)
    print("\nTest Case 5: Weather query (Context - City)")
    history5 = [
        {"role": "user", "text": "What is the weather like in London today?"},
        {"role": "assistant", "text": "Let me check the weather for London. It is currently cloudy with a temperature of 15 degrees Celsius."}
    ]
    new_input5 = "And tomorrow?" # Context: Still asking about London's weather
    response5 = generate_response(history5, new_input5, model, tokenizer, device)
    print(f"Conversation History: {history5}")
    print(f"User: {new_input5}")
    print(f"Assistant: {response5}")

    # Test Case 6: Food ordering (Intent detection + Context retention - Item)
    print("\nTest Case 6: Food ordering (Intent + Context)")
    history6 = [
        {"role": "user", "text": "I want to order a pizza."},
        {"role": "assistant", "text": "What kind of pizza would you like?"}
    ]
    new_input6 = "A large pepperoni pizza."
    response6 = generate_response(history6, new_input6, model, tokenizer, device)
    print(f"Conversation History: {history6}")
    print(f"User: {new_input6}")
    print(f"Assistant: {response6}")

    # Test Case 7: Food ordering (Context retention - Adding item)
    print("\nTest Case 7: Food ordering (Context - Adding item)")
    history7 = [
        {"role": "user", "text": "I want to order a pizza."},
        {"role": "assistant", "text": "What kind of pizza would you like?"},
        {"role": "user", "text": "A large pepperoni pizza."},
        {"role": "assistant", "text": "Okay, a large pepperoni pizza. Anything else?"}
    ]
    new_input7 = "Yes, add a coke." # Context: Adding to the previous order
    response7 = generate_response(history7, new_input7, model, tokenizer, device)
    print(f"Conversation History: {history7}")
    print(f"User: {new_input7}")
    print(f"Assistant: {response7}")

    # Test Case 8: Directions (Intent detection + Context retention - Destination)
    print("\nTest Case 8: Directions (Intent + Context)")
    history8 = [
        {"role": "user", "text": "मुझे लाल किले तक जाने का रास्ता बताओ।"} # Hindi: Tell me the way to Red Fort.
    ]
    new_input8 = "आप अभी कहाँ हैं?" # Hindi: Where are you now?
    response8 = generate_response(history8, new_input8, model, tokenizer, device)
    print(f"Conversation History: {history8}")
    print(f"User: {new_input8}")
    print(f"Assistant: {response8}")

    # Test Case 9: Directions (Context retention - Starting point)
    print("\nTest Case 9: Directions (Context - Starting point)")
    history9 = [
        {"role": "user", "text": "मुझे लाल किले तक जाने का रास्ता बताओ।"}, # Hindi: Tell me the way to Red Fort.
        {"role": "assistant", "text": "आप अभी कहाँ हैं?"} # Hindi: Where are you now?
    ]
    new_input9 = "कनॉट प्लेस में।" # Hindi: In Connaught Place.
    response9 = generate_response(history9, new_input9, model, tokenizer, device)
    print(f"Conversation History: {history9}")
    print(f"User: {new_input9}")
    print(f"Assistant: {response9}")

else:
    print("\nSkipping testing as model or tokenizer failed to load.")
