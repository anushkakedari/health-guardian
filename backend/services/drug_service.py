import httpx
from services.translation_service import translate_text

SEVERITY_KEYWORDS = {
    "severe": [
        "severe", "fatal", "death", "life-threatening",
        "serious", "toxic", "dangerous", "contraindicated"
    ],
    "moderate": [
        "moderate", "caution", "monitor", "avoid",
        "increase", "decrease", "reduce", "affect"
    ]
}


def determine_severity(interaction_text: str) -> str:
    """
    Determine severity of interaction from FDA text
    """
    text_lower = interaction_text.lower()
    for keyword in SEVERITY_KEYWORDS["severe"]:
        if keyword in text_lower:
            return "severe"
    for keyword in SEVERITY_KEYWORDS["moderate"]:
        if keyword in text_lower:
            return "moderate"
    return "mild"


def build_drug_advice(interactions: list, overall_risk: str) -> str:
    """
    Build simple advice text based on interactions found
    No external API needed
    """
    if overall_risk == "safe":
        return (
            "No dangerous interactions found between your medicines. "
            "However, always take medicines as prescribed by your doctor. "
            "Do not change your doses without consulting a doctor."
        )
    elif overall_risk == "caution":
        return (
            "Some mild interactions were found between your medicines. "
            "These are not immediately dangerous but you should inform "
            "your doctor about all medicines you are taking. "
            "Do not stop any medicine without asking your doctor first."
        )
    else:
        drugs_involved = []
        for i in interactions:
            if i["severity"] == "severe":
                drugs_involved.append(f"{i['drug1']} and {i['drug2']}")
        drug_text = ", ".join(drugs_involved) if drugs_involved else "some of your medicines"
        return (
            f"WARNING: Dangerous interaction found between {drug_text}. "
            "This combination can be harmful to your health. "
            "Please stop taking these medicines together and "
            "consult a doctor immediately."
        )


async def check_drug_interactions(medicines: list, language: str) -> dict:
    """
    Check drug interactions using OpenFDA API.
    Generates advice using rule based system.
    No external AI API needed.
    """
    fda_interactions = []

    async with httpx.AsyncClient() as http:
        for i in range(len(medicines)):
            for j in range(i + 1, len(medicines)):
                drug1 = medicines[i].strip()
                drug2 = medicines[j].strip()

                try:
                    # Search FDA for interaction between drug1 and drug2
                    url = (
                        f"https://api.fda.gov/drug/label.json"
                        f"?search=drug_interactions:{drug2}"
                        f"+AND+openfda.generic_name:{drug1}&limit=1"
                    )
                    r = await http.get(url, timeout=8)

                    if r.status_code == 200:
                        data = r.json()
                        results = data.get("results", [])
                        if results:
                            interaction_text = results[0].get(
                                "drug_interactions", [""]
                            )[0][:300]

                            if interaction_text:
                                severity = determine_severity(interaction_text)
                                fda_interactions.append({
                                    "drug1": drug1,
                                    "drug2": drug2,
                                    "severity": severity,
                                    "description": interaction_text
                                })
                    else:
                        # FDA has no data — mark as needs checking
                        fda_interactions.append({
                            "drug1": drug1,
                            "drug2": drug2,
                            "severity": "mild",
                            "description": (
                                f"No specific interaction data found for "
                                f"{drug1} and {drug2}. "
                                "Consult your doctor to be safe."
                            )
                        })

                except Exception:
                    # Network issue — flag it
                    fda_interactions.append({
                        "drug1": drug1,
                        "drug2": drug2,
                        "severity": "mild",
                        "description": (
                            f"Could not check interaction between "
                            f"{drug1} and {drug2}. "
                            "Please consult a doctor or pharmacist."
                        )
                    })

    # Calculate overall risk
    severities = [i["severity"] for i in fda_interactions]
    if "severe" in severities:
        overall_risk = "danger"
    elif "moderate" in severities:
        overall_risk = "caution"
    else:
        overall_risk = "safe"

    # Build advice in English
    advice_english = build_drug_advice(fda_interactions, overall_risk)

    # Translate advice
    translated_advice = translate_text(advice_english, language)

    # Translate interaction descriptions if not English
    if language.lower() != "english":
        for interaction in fda_interactions:
            interaction["description"] = translate_text(
                interaction["description"], language
            )

    return {
        "interactions": fda_interactions,
        "overall_risk": overall_risk,
        "advice": advice_english,
        "translated_advice": translated_advice
    }

