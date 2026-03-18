from deep_translator import GoogleTranslator

LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "kannada": "kn",
    "telugu": "te"
}

def translate_text(text: str, target_language: str) -> str:
    """
    Translates text to target language.
    If translation fails, returns original English text.
    """
    lang_code = LANGUAGE_CODES.get(target_language.lower(), "en")

    # No need to translate if already English
    if lang_code == "en":
        return text

    try:
        translated = GoogleTranslator(
            source="en",
            target=lang_code
        ).translate(text)
        return translated
    except Exception:
        # If translation fails, return original English
        return text


def translate_list(items: list[str], target_language: str) -> list[str]:
    """
    Translates a list of strings to target language.
    Used for translating precautions list.
    """
    lang_code = LANGUAGE_CODES.get(target_language.lower(), "en")

    if lang_code == "en":
        return items

    translated_items = []
    for item in items:
        try:
            translated = GoogleTranslator(
                source="en",
                target=lang_code
            ).translate(item)
            translated_items.append(translated)
        except Exception:
            translated_items.append(item)

    return translated_items