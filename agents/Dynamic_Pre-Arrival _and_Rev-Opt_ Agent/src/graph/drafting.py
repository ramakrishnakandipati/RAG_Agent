"""LLM (Qwen3-VL) drafting + reply classification. The LLM only writes copy with {{PLACEHOLDERS}}; facts are rendered deterministically."""
from __future__ import annotations
import re

from src.llm.client import get_llm
from src.utils.logger import get_logger

log = get_logger("drafting")

PLACEHOLDERS = ["{{GUEST_FIRST_NAME}}", "{{HOTEL}}", "{{OFFER_NAME}}", "{{OFFER_PRICE}}", "{{OFFER_LIST_PRICE}}",
                "{{OFFER_DISCOUNT}}", "{{OFFER_DATE}}", "{{CHECK_IN}}"]
NEG_WORDS = ["terrible", "awful", "angry", "refund", "complaint", "disappointed", "unacceptable", "worst", "rude", "dirty",
             "horrible", "furious", "never again", "ridiculous", "disgusting", "cancel my"]
CUSTOM_WORDS = ["arrange", "surprise", "proposal", "propose", "private", "helicopter", "custom", "special request", "bespoke",
                "anniversary dinner on the beach", "yacht", "fireworks", "personalised", "personalized"]
YES = ["yes", "sure", "please add", "book it", "sounds good", "i'd love", "i would love", "go ahead", "add it", "interested", "count me in"]
NO = ["no thanks", "not interested", "no thank", "pass", "decline", "don't want", "do not want", "maybe later", "not now"]
SAFE_SIGN = "The Azure Bay Team"


# ------------------------------------------------------------------------------------ inbound classification
def classify_reply(text: str, offer: dict | None) -> dict:
    t = (text or "").lower()
    llm = get_llm()
    res = None
    if not llm.mock:
        try:
            res = llm.json(
                "You classify a hotel guest's reply to a pre-arrival offer. Fields: intent (accept|decline|question|custom_request|complaint|other), "
                "sentiment (positive|neutral|negative), sentiment_score (-1..1), custom_request (true if the guest asks for something bespoke "
                "or outside the standard offer menu).",
                f"Offer being discussed: {offer['name'] if offer else 'n/a'}\nGuest reply: {text}")
        except Exception as e:  # noqa: BLE001
            log.warning("classify_llm_failed", error=str(e))
    if not res:  # deterministic rules (mock mode / LLM failure)
        intent = "other"
        if any(w in t for w in NEG_WORDS):
            intent = "complaint"
        elif any(w in t for w in CUSTOM_WORDS):
            intent = "custom_request"
        elif any(w in t for w in NO):
            intent = "decline"
        elif "?" in t or t.startswith(("what", "when", "how", "can ", "do you", "is ")):
            intent = "question"
        elif any(w in t for w in YES):
            intent = "accept"
        res = {"intent": intent, "sentiment": "negative" if intent == "complaint" else ("positive" if intent == "accept" else "neutral"),
               "sentiment_score": -0.7 if intent == "complaint" else (0.6 if intent == "accept" else 0.0),
               "custom_request": intent == "custom_request"}
    # deterministic safety net regardless of LLM output
    if any(w in t for w in NEG_WORDS):
        res.update(intent="complaint", sentiment="negative", sentiment_score=min(float(res.get("sentiment_score", -0.7)), -0.5))
    if any(w in t for w in CUSTOM_WORDS):
        res["custom_request"] = True
    res["sentiment_score"] = float(max(-1.0, min(1.0, float(res.get("sentiment_score", 0.0)))))
    return res


# ------------------------------------------------------------------------------------ drafting
def _mock_template(action: str, offer: dict, state: dict, feedback: list[str]) -> dict:
    cat = offer["category"]
    hook = {"spa": "A little calm by the sea awaits at the Tidewater Spa.",
            "dining": "An evening at Salt & Ember is the perfect way to settle in.",
            "room_upgrade": "Wake up to a little more space and a better view.",
            "late_checkout": "Make your last morning unhurried and easy.",
            "loyalty": "Booking direct next time means better rates and warmer perks."}[cat]
    disc = " That is {{OFFER_DISCOUNT}} off our usual rate." if offer["discount_pct"] > 0 and cat != "loyalty" else ""
    if action == "offer":
        price = " It is yours for {{OFFER_PRICE}}." if cat != "loyalty" else " Membership is complimentary."
        if cat == "loyalty":
            price = " Membership is complimentary and brings {{OFFER_DISCOUNT}} off your next direct booking."
        return {"subject": "A little something for your stay at {{HOTEL}}",
                "body_template": f"Hi {{{{GUEST_FIRST_NAME}}}}, we are looking forward to welcoming you on {{{{CHECK_IN}}}}. {hook} "
                                 f"We would love to offer you {{{{OFFER_NAME}}}}.{price if cat == 'loyalty' else price + disc} "
                                 f"Simply reply YES and we will take care of the rest.\n\n{SAFE_SIGN}"}
    if action == "confirm":
        return {"subject": "Your request is confirmed", "body_template":
                f"Hi {{{{GUEST_FIRST_NAME}}}}, wonderful news - {{{{OFFER_NAME}}}} has been added to your reservation"
                + (" for {{OFFER_PRICE}}, charged to your room folio." if cat != "loyalty" else ", welcome to Azure Circle.")
                + f" We cannot wait to see you.\n\n{SAFE_SIGN}"}
    if action == "ack_decline":
        return {"subject": "No problem at all", "body_template":
                f"Hi {{{{GUEST_FIRST_NAME}}}}, thank you for letting us know. Your stay is all set and we look forward to welcoming you.\n\n{SAFE_SIGN}"}
    if action == "holding":
        return {"subject": "We are on it", "body_template":
                f"Hi {{{{GUEST_FIRST_NAME}}}}, thank you for getting in touch. A member of our concierge team will personally follow up with you shortly.\n\n{SAFE_SIGN}"}
    return {"subject": "Thank you for your message", "body_template":
            f"Hi {{{{GUEST_FIRST_NAME}}}}, thank you for your message. Our team will be glad to help with any questions when you arrive.\n\n{SAFE_SIGN}"}


def safe_fallback(state: dict, action: str = "holding") -> dict:
    t = _mock_template(action, state["selected_offer"], state, [])
    return dict(t, action=action, source="safe_fallback")


def draft_with_llm(state: dict, action: str) -> dict:
    offer, guest, env = state["selected_offer"], state["guest"], state["env"]
    llm = get_llm()
    if llm.mock or action in ("ack_decline", "holding", "other"):
        return dict(_mock_template(action, offer, state, state.get("feedback", [])), action=action, source="template")
    snippets = "\n".join(f"- {s['text']}" for s in state.get("kb_snippets", []))
    wx = "rain is in the forecast" if env["precip_prob"] > 0.55 else ("clear sunny skies are expected" if env["precip_prob"] < 0.25 else "mixed weather is expected")
    events = ", ".join(env["events"]) if env["events"] else "none"
    task = {"offer": "Write a warm pre-arrival offer email promoting the offer and inviting the guest to reply YES.",
            "confirm": "Write a short confirmation that the item has been added to the guest's reservation (charged to the room folio).",
            "answer": f"Answer the guest's question using ONLY the property facts below, then gently mention the offer. Guest said: {state.get('inbound_text')}"}[action]
    system = (
        "You are the copywriter for Azure Bay Resort & Spa. Follow the brand voice. "
        "STRICT RULES: (1) NEVER write any number with $ or %, any SKU code, or any price/discount yourself - use ONLY these placeholders: "
        f"{', '.join(PLACEHOLDERS)}. (2) Use {{{{OFFER_DISCOUNT}}}} only when a discount applies. (3) Max 2 short paragraphs, sign off 'The Azure Bay Team'. "
        "(4) No pressure language, no ALL CAPS. Output JSON: {\"subject\": str, \"body\": str}.")
    user = (f"Task: {task}\nGuest segment: {guest['segment']} ({guest['loyalty_tier']} member)\nOffer: {offer['name']} - {offer['desc']}\n"
            f"Has discount: {offer['discount_pct'] > 0}\nContext: {wx}; local events: {events}\nProperty facts:\n{snippets}\n"
            + (f"Previous attempt was rejected for: {state['feedback']}. Fix these issues.\n" if state.get("feedback") else ""))
    try:
        out = llm.json(system, user, temperature=0.4)
        return {"subject": str(out["subject"]), "body_template": str(out["body"]), "action": action, "source": "qwen3-vl"}
    except Exception as e:  # noqa: BLE001
        log.warning("draft_llm_failed", error=str(e))
        return dict(_mock_template(action, offer, state, []), action=action, source="template_fallback")


POISON = {"subject": "Special deal just for you!",
          "body_template": "Hi {{GUEST_FIRST_NAME}}, grab SPA-FAKE9 today for just $49 - that is 40% off our best price, guaranteed!\n\nThe Azure Bay Team"}


def poison_draft(action: str) -> dict:
    return dict(POISON, action=action, source="adversarial")
