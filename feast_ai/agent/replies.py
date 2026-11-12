"""What the assistant says about a tool result - written from the result.

After a call the app runs the tool, draws the results as cards and lets the
assistant add a sentence.  That sentence is where a small model invents: a
price that was not returned, a fourth venue, an opening hour.  So the
sentence has one definition, and two users:

* **training** (``mustachio/generate.py``): the label of every "answer a
  result" turn is written by :func:`say` from the result itself, so no
  training reply states anything the result does not;
* **the app** (:func:`check`): the model's sentence is compared with the
  result in code.  A number or a name the result, the call and the user's
  message do not contain means the reply is replaced by :func:`say`'s - the
  same sentence the model was trained towards, now guaranteed.

The voice is the app's persona (Mustachio: short, direct, no filler), the
rules are the prompt's: at most three places, nothing listed (the cards do
that), an empty result is said plainly with ONE concrete way to widen the
search, a failed lookup is said once.

Sentences avoid a case suffix on a name ("Asma Kahve, 2 dakika" rather than
"Asma Kahve'ye 2 dakika"): Turkish vowel harmony on an arbitrary venue name
is one more thing to get wrong, and the nominative never is.
"""

from __future__ import annotations

import json
import random
import re

from .text import detect_lang, fold

#: Tools whose result is a list the user asked to see.
LISTING = ("search_restaurants", "search_menu_items", "search_posts", "ask_about_posts",
           "search_saved", "search_shared_items", "search_chat_memory", "list_friends",
           "get_suggested_users", "get_restaurant_details", "get_menu_item_details")
_MODE = {"tr": {"walk": "yürüme", "drive": "araba", "transit": "toplu taşıma"},
         "en": {"walk": "walking", "drive": "driving", "transit": "transit"}}


def _join(names: list[str], english: bool) -> str:
    names = [str(n) for n in names[:3]]
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + (" and " if english else " ve ") + names[-1]


def _quoted(args: dict, key: str, default: str) -> str:
    value = str(args.get(key) or "").strip()
    return value or default


def widened(name: str, args: dict) -> dict | None:
    """The call a "yes" means after an empty result, or ``None``.

    :func:`say` asks a yes/no question about an empty result only where the
    answer has one meaning: drop the opening-time filter, drop the distance
    limit, drop the price limit - in that order, as it asks.  Anything else
    ("another area?") is left for the user to say in their own words.
    """
    drop: tuple[str, ...] = ()
    if name == "search_restaurants":
        if args.get("open_now") or args.get("open_at"):
            drop = ("open_now", "open_at")
        elif args.get("max_walk_minutes") or args.get("max_distance_meters"):
            drop = ("max_walk_minutes", "max_distance_meters")
    elif name == "search_menu_items" and args.get("max_price") is not None:
        drop = ("max_price",)
    if not drop:
        return None
    return {key: value for key, value in args.items() if key not in drop}


def say(name: str, args: dict, result: dict, english: bool = False, focus: str = "",
        rng: random.Random | None = None) -> str:
    """One or two sentences about ``result``, stating only what is in it.

    ``focus`` is what the user asked about when the tool returns more than
    that ("hours", "address", "menu", "ingredients", "price").  With ``rng``
    one of several wordings is drawn (training); without, always the first
    (the app's fallback is deterministic).
    """
    pick = rng.choice if rng else (lambda options: options[0])
    items = list(result.get("items") or [])
    count = int(result.get("count", len(items)) or 0)
    ok = bool(result.get("ok", True))
    tr = not english

    if not ok and name not in ("get_directions", "send_message_to_friends", "open_screen",
                               "set_active_location"):
        return pick(("Şu an verilere ulaşamadım, birazdan tekrar dene.",
                     "Verilere ulaşamadım şu an. Birazdan bir daha dene.") if tr else
                    ("I couldn't reach the data just now. Please try again in a moment.",))

    if name == "search_restaurants":
        what = _quoted(args, "query", "mekan" if tr else "place")
        if not items:
            if args.get("open_now") or args.get("open_at"):
                return pick((f"O saatte açık {what} bulamadım. Kapalı olanlara da bakayım mı?",
                             f"Açık {what} çıkmadı. Saat şartını kaldırıp bakayım mı?") if tr else
                            (f"No {what} open then. Want me to include the closed ones?",))
            if args.get("max_walk_minutes") or args.get("max_distance_meters"):
                return pick((f"Bu mesafede {what} çıkmadı. Biraz daha uzağa bakayım mı?",
                             f"O kadar yakında {what} yok. Mesafeyi açalım mı?") if tr else
                            (f"No {what} within that distance. Shall I look a bit further?",))
            return pick((f"Yakında {what} bulamadım. Başka bir semt söylersen oraya bakarım.",
                         f"Buralarda {what} çıkmadı. Bir semt söyle, oraya bakayım.") if tr else
                        (f"I found no {what} nearby. Name another area and I'll look there.",))
        first = items[0]
        closed = "" if first.get("open", True) else (" Şu an kapalı ama." if tr else
                                                     " It's closed right now, though.")
        if len(items) == 1:
            return pick((f"{first['name']} var, {first['walk_min']} dakika yürüme.{closed}",
                         f"Bir yer çıktı: {first['name']}, {first['walk_min']} dakikalık "
                         f"yürüme mesafesinde.{closed}") if tr else
                        (f"There's {first['name']}, a {first['walk_min']}-minute walk.{closed}",))
        near = min(items, key=lambda i: i.get("walk_min", 99))
        if args.get("sort_by") == "popularity":
            return pick((f"{count} yer buldum. En popüleri {first['name']}, "
                         f"{first['walk_min']} dakika yürüme.",
                         f"En çok {first['name']} seviliyor, {first['walk_min']} dakikada. "
                         f"Toplam {count} yer var.") if tr else
                        (f"I found {count} places. The most popular is {first['name']}, "
                         f"a {first['walk_min']}-minute walk.",))
        others = [i["name"] for i in items if i is not near][:2]
        return pick((f"{count} yer buldum. En yakını {near['name']}, {near['walk_min']} dakika "
                     f"yürüme.",
                     f"{count} mekan çıktı. En yakını {near['name']}, {near['walk_min']} dakika. "
                     f"Ayrıca {_join(others, False)} var.",
                     f"{_join([near['name']] + others, False)} var. En yakını {near['name']}, "
                     f"{near['walk_min']} dakika yürüme.") if tr else
                    (f"I found {count} places. The closest is {near['name']}, "
                     f"a {near['walk_min']}-minute walk.",
                     f"{_join([near['name']] + others, True)} came up. {near['name']} is "
                     f"closest, {near['walk_min']} minutes on foot."))

    if name == "search_menu_items":
        what = _quoted(args, "query", "bu" if tr else "that")
        if not items:
            if args.get("max_price") is not None:
                return pick((f"{args['max_price']} liranın altında {what} çıkmadı. Bütçeyi "
                             f"biraz açalım mı?",
                             f"O fiyata {what} yok. Limiti biraz yükseltip bakayım mı?") if tr else
                            (f"No {what} under {args['max_price']} lira. Raise the budget "
                             f"a little?",))
            return pick((f"Yakındaki menülerde {what} bulamadım. Başka bir semt söylersen "
                         f"oraya bakarım.",
                         f"Buralarda {what} çıkmadı. Bir semt söyle, oraya bakayım.") if tr else
                        (f"I found no {what} on nearby menus. Name another area and I'll "
                         f"look there.",))
        if len(items) == 1:
            one = items[0]
            return pick((f"{one['name']} var: {one['restaurant']}, {one['price']} lira.",
                         f"Bir yerde buldum: {one['restaurant']}, {one['name']} "
                         f"{one['price']} lira.") if tr else
                        (f"{one['name']} at {one['restaurant']}, {one['price']} lira.",))
        if args.get("sort_by") == "price_desc":
            top = max(items, key=lambda i: i.get("price", 0))
            return pick((f"En pahalısı {top['restaurant']}: {top['name']}, {top['price']} lira.",)
                        if tr else
                        (f"The priciest is {top['name']} at {top['restaurant']}, "
                         f"{top['price']} lira.",))
        low = min(items, key=lambda i: i.get("price", 0))
        high = max(items, key=lambda i: i.get("price", 0))
        return pick((f"{count} seçenek var. En uygunu {low['restaurant']}: {low['name']}, "
                     f"{low['price']} lira.",
                     f"{count} yerde buldum, fiyatlar {low['price']} ile {high['price']} lira "
                     f"arasında. En ucuzu {low['restaurant']}.",
                     f"En ucuzu {low['price']} lira, {low['restaurant']}. Toplam {count} "
                     f"seçenek çıktı.") if tr else
                    (f"{count} options. The cheapest is {low['name']} at {low['restaurant']}, "
                     f"{low['price']} lira.",
                     f"I found {count}; prices run from {low['price']} to {high['price']} lira. "
                     f"{low['restaurant']} is the cheapest."))

    if name == "get_restaurant_details":
        if not items:
            return ("Bu mekanın detaylarını bulamadım." if tr
                    else "I couldn't find that venue's details.")
        one = items[0]
        state = (("Şu an açık." if one.get("open") else "Şu an kapalı.") if tr else
                 ("It's open now." if one.get("open") else "It's closed now."))
        menu = _join(list(one.get("menu") or []), english)
        if focus == "hours":
            return pick((f"{one['name']} {one['hours']} arası açık. {state}",
                         f"Çalışma saatleri {one['hours']}. {state}") if tr else
                        (f"{one['name']} is open {one['hours']}. {state}",))
        if focus == "address":
            return pick((f"{one['name']}: {one['address']}.", f"Adresi {one['address']}.") if tr
                        else (f"{one['name']} is at {one['address']}.",))
        if focus == "menu" and menu:
            return pick((f"{one['name']} menüsünde {menu} var.",
                         f"Menüde öne çıkanlar: {menu}.") if tr else
                        (f"On the menu at {one['name']}: {menu}.",))
        return (f"{one['name']} {one['hours']} arası açık, adresi {one['address']}." if tr else
                f"{one['name']} is open {one['hours']}, at {one['address']}.")

    if name == "get_menu_item_details":
        if not items:
            return ("Bu ürünün detaylarını bulamadım." if tr
                    else "I couldn't find that item's details.")
        one = items[0]
        if focus == "ingredients":
            return pick((f"{one['name']} içinde {one['ingredients']} var.",
                         f"İçindekiler: {one['ingredients']}.") if tr else
                        (f"{one['name']} has {one['ingredients']}.",))
        if focus == "price":
            return pick((f"{one['name']} {one['price']} lira, {one['restaurant']}.",
                         f"{one['price']} lira. {one['restaurant']} menüsünde.") if tr else
                        (f"{one['name']} is {one['price']} lira at {one['restaurant']}.",))
        return (f"{one['name']}: {one['price']} lira, {one['restaurant']}. İçinde "
                f"{one['ingredients']} var." if tr else
                f"{one['name']}: {one['price']} lira at {one['restaurant']}. It has "
                f"{one['ingredients']}.")

    if name == "get_directions":
        if not ok:
            return ("Yol tarifini açamadım, birazdan tekrar dene." if tr
                    else "I couldn't open the directions. Please try again in a moment.")
        mode = _MODE["tr" if tr else "en"].get(str(result.get("mode") or args.get("mode")), "")
        place = str(result.get("name") or "")
        if tr:
            route = f"{mode} tarifi" if mode else "yol tarifi"
            return pick((f"{place} için {route}ni haritada açtım.",
                         f"Haritayı açtım, {place} için {route} hazır."))
        return f"I opened {mode + ' ' if mode else ''}directions to {place} in maps."

    if name == "set_active_location":
        if not ok:
            return "Konumu değiştiremedim." if tr else "I couldn't change the location."
        if result.get("is_gps"):
            return pick(("Tamam, yine bulunduğun konuma göre arıyorum.",
                         "Tamam, GPS konumuna döndüm.") if tr else
                        ("Done, searching around where you are again.",))
        place = str(result.get("active_location") or "")
        return pick((f"Tamam, artık {place} çevresinde arıyorum.",
                     f"Oldu, arama konumu {place}.") if tr else
                    (f"Done, searching around {place} now.",))

    if name == "search_posts":
        if not items:
            return pick(("Bununla ilgili gönderi bulamadım. Başka bir kelime söylersen ona "
                         "bakarım.",) if tr else
                        ("I found no posts about that. Give me another word and I'll look.",))
        one = items[0]
        return pick((f"{count} gönderi var. {one['user']}, {one['restaurant']} için "
                     f"\"{one['caption']}\" demiş.",
                     f"{count} gönderi buldum; ilki {one['user']} tarafından, "
                     f"{one['restaurant']} hakkında.") if tr else
                    (f"{count} posts. {one['user']} wrote \"{one['caption']}\" about "
                     f"{one['restaurant']}.",))

    if name == "ask_about_posts":
        if not items:
            return pick(("Gönderilerde buna uyan bir yer çıkmadı. Başka bir özellik "
                         "söylersen ona bakarım.",) if tr else
                        ("No place in the posts matches that. Name another quality and "
                         "I'll look.",))
        one = items[0]
        if len(items) == 1:
            return pick((f"{one['name']} uyuyor: gönderilerde \"{one['why']}\" deniyor.",) if tr
                        else (f"{one['name']} fits: posts say \"{one['why']}\".",))
        return pick((f"{_join([i['name'] for i in items], False)} öne çıkıyor. {one['name']} "
                     f"için \"{one['why']}\" denmiş.",
                     f"Gönderilere göre {_join([i['name'] for i in items[:2]], False)} uyuyor.")
                    if tr else
                    (f"{_join([i['name'] for i in items], True)} stand out. Posts say "
                     f"\"{one['why']}\" about {one['name']}.",))

    if name in ("search_saved", "search_shared_items"):
        shared = name == "search_shared_items"
        if not items:
            return ((("Buna uyan bir paylaşım bulamadım." if shared
                      else "Kaydettiklerinde buna uyan bir şey yok.") if tr else
                     ("I found no shared item like that." if shared
                      else "Nothing like that in your saved items.")))
        names = _join([i.get("name") or i.get("caption", "") for i in items], english)
        order = args.get("sort_by")
        if order == "distance" and all("walk_min" in i for i in items):
            near = min(items, key=lambda i: i["walk_min"])
            return (f"En yakını {near['name']}, {near['walk_min']} dakika yürüme." if tr else
                    f"The closest is {near['name']}, a {near['walk_min']}-minute walk.")
        if order in ("price_asc", "price_desc") and all("price" in i for i in items):
            top = (min if order == "price_asc" else max)(items, key=lambda i: i["price"])
            word = (("En ucuzu" if order == "price_asc" else "En pahalısı") if tr else
                    ("The cheapest is" if order == "price_asc" else "The priciest is"))
            return (f"{word} {top['name']}, {top['price']} lira." if tr else
                    f"{word} {top['name']}, {top['price']} lira.")
        if english and count == 1:
            return f"One match: {names}."
        if shared:
            return pick((f"{count} paylaşım buldum: {names}.",
                         f"Aranızda {count} paylaşım var: {names}.") if tr else
                        (f"I found {count} shared items: {names}.",))
        return pick((f"Kaydettiklerinde {count} sonuç var: {names}.",
                     f"{count} kayıt buldum: {names}.") if tr else
                    (f"{count} saved items match: {names}.",))

    if name == "search_chat_memory":
        if not items:
            return ("Bunu daha önce konuştuğumuza dair bir şey bulamadım." if tr
                    else "I found nothing about that in our earlier chats.")
        one = items[0]
        when = str(one["when"])
        return f"{when[:1].upper() + when[1:]}: {one['text']}."

    if name == "list_friends":
        if not items:
            return ("Bu isimde bir arkadaşın görünmüyor." if tr
                    else "I don't see a friend by that name.")
        names = _join([i["name"] for i in items], english)
        if count > 3:
            return (f"{count} arkadaşın var; {names} ve diğerleri." if tr
                    else f"You have {count} friends: {names} and others.")
        return (f"{count} arkadaşın var: {names}." if tr else f"You have {count} friends: {names}.")

    if name == "get_suggested_users":
        if not items:
            return "Şu an önerebileceğim kimse yok." if tr else "No suggestions right now."
        one = items[0]
        names = _join([i["name"] for i in items], english)
        return pick((f"Önerilerde {names} var. {one['name']} ile {one['mutual']} ortak "
                     f"arkadaşınız var.",) if tr else
                    (f"Suggestions: {names}. You share {one['mutual']} friends with "
                     f"{one['name']}.",))

    if name == "open_screen":
        return pick(("Açtım.", "Açıldı.") if tr else ("Opened.",)) if ok else (
            "Açamadım." if tr else "I couldn't open it.")

    if name == "send_message_to_friends":
        if not ok:
            return ("Gönderemedim, birazdan tekrar dene." if tr
                    else "I couldn't send it. Please try again in a moment.")
        return pick(("Gönderdim.", "Mesaj gitti.", "Tamam, gönderdim.") if tr else ("Sent.",))

    return "Tamam." if tr else "Done."


# --------------------------------------------------------------------------
# Checking a reply against its result
# --------------------------------------------------------------------------
_NUMBER = re.compile(r"\d+(?:[:.,]\d+)*")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
#: Capitalised words that are not names of anything in a result.
_FREE = {fold(w) for w in (
    "Mustachio feast TL GPS Pazartesi Salı Çarşamba Perşembe Cuma Cumartesi Pazar "
    "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday I OK").split()}


def _numbers(text: str) -> set[str]:
    out = set()
    for found in _NUMBER.findall(text):
        out.add(found)
        out.update(part.lstrip("0") or "0" for part in re.split(r"[:.,]", found))
    return out


def unsupported(reply: str, sources: list[str]) -> list[str]:
    """Numbers and names in ``reply`` that none of ``sources`` contain.

    A number must appear in a source (as written, or as one side of a time or
    a decimal).  A name is a capitalised word that does not open a sentence;
    it must appear in a source too.  Both tests are deliberately strict - a
    false alarm costs a templated sentence, a miss costs an invented price.
    """
    joined = " ".join(sources)
    known_numbers, haystack = _numbers(joined), fold(joined)
    missing = [n for n in _NUMBER.findall(reply)
               if n not in known_numbers and (n.lstrip("0") or "0") not in known_numbers]
    opens_sentence = True
    for token in re.finditer(r"[^\W\d_]+|[.!?:\n\"“]+|\S", reply, re.UNICODE):
        word = token.group(0)
        if not _WORD.fullmatch(word):
            opens_sentence = bool(re.fullmatch(r"[.!?:\n\"“]+", word))
            continue
        if word[0].isupper() and not opens_sentence and len(word) > 1:
            folded = fold(word)
            if folded not in _FREE and folded not in haystack:
                missing.append(word)
        opens_sentence = False
    return missing


def check(reply: str, name: str, args: dict, result: dict, user_text: str = "",
          context_line: str = "", english: bool = False) -> tuple[str, list[str]]:
    """``(the sentence to show, notes)`` for the model's answer to a result.

    The reply stands when everything it states is in the result, the call or
    the user's message; otherwise :func:`say` writes it.  ``notes`` says what
    was missing, for the turn's report.
    """
    text = reply.strip()
    sources = [json.dumps(result, ensure_ascii=False), json.dumps(args, ensure_ascii=False),
               user_text, context_line]
    if not _WORD.search(text):
        return say(name, args, result, english), ["reply replaced: it was not a sentence"]
    missing = unsupported(text, sources)
    if missing:
        return say(name, args, result, english), [
            "reply replaced: " + ", ".join(repr(m) for m in missing[:4]) + " not in the result"]
    # The data is Turkish whatever the user speaks, and a small model follows
    # the data: an English question answered in Turkish is rewritten.  Only
    # that direction is tested - a short Turkish sentence full of venue names
    # does not read reliably as Turkish.
    if english and len(_WORD.findall(text)) >= 4 and detect_lang(text) != "en":
        return say(name, args, result, english), ["reply replaced: not in the user's language"]
    return text, []
