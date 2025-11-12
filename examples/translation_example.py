#!/usr/bin/env python3
"""
Example usage of the Speaches Translation API.

This script demonstrates how to use the translation endpoints with various options.
"""

import asyncio

from openai import AsyncOpenAI, OpenAI


def basic_translation_sync():
    """Basic synchronous translation example."""
    print("\n=== Basic Translation (Sync) ===")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # Set this if you configured an API key
    )

    # Translate English to Chinese
    translation = client.audio.translations.create(
        file=("text.txt", "Hello, how are you today?".encode()),
        model="Helsinki-NLP/opus-mt-en-zh",
    )

    print(f"Original: Hello, how are you today?")
    print(f"Translation: {translation.text}")
    print(f"Model: Helsinki-NLP/opus-mt-en-zh")


async def basic_translation_async():
    """Basic asynchronous translation example."""
    print("\n=== Basic Translation (Async) ===")

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    # Translate English to Spanish
    translation = await client.audio.translations.create(
        file=("text.txt", "Welcome to the translation service!".encode()),
        model="Helsinki-NLP/opus-mt-en-es",
    )

    print(f"Original: Welcome to the translation service!")
    print(f"Translation: {translation.text}")
    print(f"Model: Helsinki-NLP/opus-mt-en-es")

    await client.close()


async def multiple_translations():
    """Translate the same text to multiple languages."""
    print("\n=== Multiple Translations ===")

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    text = "Good morning! Have a nice day."
    models = [
        ("Spanish", "Helsinki-NLP/opus-mt-en-es"),
        ("French", "Helsinki-NLP/opus-mt-en-fr"),
        ("German", "Helsinki-NLP/opus-mt-en-de"),
        ("Chinese", "Helsinki-NLP/opus-mt-en-zh"),
    ]

    print(f"Original (English): {text}\n")

    tasks = []
    for lang_name, model in models:
        task = client.audio.translations.create(
            file=("text.txt", text.encode()),
            model=model,
        )
        tasks.append((lang_name, task))

    for lang_name, task in tasks:
        translation = await task
        print(f"{lang_name}: {translation.text}")

    await client.close()


async def batch_translation():
    """Translate multiple texts to the same language."""
    print("\n=== Batch Translation ===")

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    texts = [
        "Hello, how are you?",
        "Thank you for your help.",
        "Have a great day!",
        "See you tomorrow.",
    ]

    model = "Helsinki-NLP/opus-mt-en-es"

    print(f"Translating {len(texts)} texts to Spanish...\n")

    tasks = [client.audio.translations.create(file=("text.txt", text.encode()), model=model) for text in texts]

    translations = await asyncio.gather(*tasks)

    for original, translation in zip(texts, translations, strict=False):
        print(f"EN: {original}")
        print(f"ES: {translation.text}\n")

    await client.close()


def using_requests_library():
    """Example using the requests library instead of OpenAI client."""
    print("\n=== Using Requests Library ===")

    import requests

    response = requests.post(
        "http://localhost:8000/v1/audio/translations",
        data={
            "text": "The weather is beautiful today.",
            "model": "Helsinki-NLP/opus-mt-en-fr",
        },
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Original: The weather is beautiful today.")
        print(f"Translation: {data['text']}")
        print(f"Source: {data['source_language']}")
        print(f"Target: {data['target_language']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def error_handling():
    """Example of error handling."""
    print("\n=== Error Handling ===")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    try:
        # Try with invalid model
        translation = client.audio.translations.create(
            file=("text.txt", "Hello".encode()),
            model="invalid-model-id",
        )
        print(f"Translation: {translation.text}")
    except Exception as e:
        print(f"Error caught: {type(e).__name__}: {e}")

    try:
        # Try with empty text
        translation = client.audio.translations.create(
            file=("text.txt", "".encode()),
            model="Helsinki-NLP/opus-mt-en-zh",
        )
        print(f"Empty text translation: '{translation.text}'")
    except Exception as e:
        print(f"Error caught: {type(e).__name__}: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Speaches Translation API Examples")
    print("=" * 60)
    print("\nMake sure the server is running:")
    print("  uvicorn speaches.main:create_app --factory --port 8000")
    print("=" * 60)

    # Synchronous examples
    try:
        basic_translation_sync()
    except Exception as e:
        print(f"Error in basic_translation_sync: {e}")

    try:
        using_requests_library()
    except Exception as e:
        print(f"Error in using_requests_library: {e}")

    try:
        error_handling()
    except Exception as e:
        print(f"Error in error_handling: {e}")

    # Asynchronous examples
    try:
        asyncio.run(basic_translation_async())
    except Exception as e:
        print(f"Error in basic_translation_async: {e}")

    try:
        asyncio.run(multiple_translations())
    except Exception as e:
        print(f"Error in multiple_translations: {e}")

    try:
        asyncio.run(batch_translation())
    except Exception as e:
        print(f"Error in batch_translation: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
