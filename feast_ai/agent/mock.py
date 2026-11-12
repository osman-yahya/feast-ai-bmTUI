"""A mock of the phone's local data, so a tool call can be answered.

feast AI's benchmarks stop at the call: the tools are not executed, and a
chat in the TUI ends with ``search_menu_items(...)`` on screen.  The mobile
app does not stop there - it runs the tool against its local cache, shows the
results as cards and lets the assistant say a sentence about them.  This
module is that cache, invented: a few dozen venues and dishes around the
active location, the user's saved items, friends and shares.  It is fixed
data, not a simulation - the same call always returns the same result - and
none of it is real.

Two users, one definition of what a result looks like:

* the TUI and the benchmark execute calls with :func:`run`;
* the training generator (``mustachio/generate.py``) builds its tool results
  with the same item builders (:func:`venue_item`, :func:`dish_item`, ...),
  so the model is trained on exactly the shape it is served.

Every result is ``{"ok": bool, "count": int, "items": [...]}``.  Keys are
short because the result is prompt: each one is prefilled on the phone's CPU
before the assistant can answer.  At most :data:`MAX_ITEMS` items go back to
the model - the app shows the rest as cards anyway.
"""

from __future__ import annotations

import hashlib

from .context import AgentContext
from .text import fold

#: The assistant may name three places; a fourth and fifth give it a choice.
MAX_ITEMS = 5

# --------------------------------------------------------------------------
# Item shapes (shared with the training generator)
# --------------------------------------------------------------------------


def venue_item(venue_id: str, name: str, walk_min: int, is_open: bool = True) -> dict:
    return {"id": venue_id, "name": name, "walk_min": walk_min, "open": is_open}


def dish_item(dish_id: str, name: str, price: int, restaurant: str) -> dict:
    return {"id": dish_id, "name": name, "price": price, "restaurant": restaurant}


def post_item(user: str, caption: str, restaurant: str) -> dict:
    return {"user": user, "caption": caption, "restaurant": restaurant}


def opinion_item(name: str, why: str) -> dict:
    """A venue that posts speak for, and the phrase that does."""
    return {"name": name, "why": why}


def result(items: list[dict], ok: bool = True, **extra) -> dict:
    out = {"ok": ok, "count": len(items), "items": items[:MAX_ITEMS]}
    out.update(extra)
    return out


def done(ok: bool = True, **extra) -> dict:
    """The result of an action that returns nothing to list."""
    out = {"ok": ok}
    out.update(extra)
    return out


# --------------------------------------------------------------------------
# The invented neighbourhood
# --------------------------------------------------------------------------
#: ``(id, name, kinds it answers to, walk minutes, open now, popularity)``.
VENUES = (
    ("r_101", "Çınaraltı Kahvaltı", "kahvaltı kahvalti kafe", 4, True, 9),
    ("r_102", "Rıhtım Balık", "balık balıkçı deniz ürünleri meyhane", 7, True, 8),
    ("r_103", "Usta Dönerci", "döner dönerci dürüm sokak lezzetleri", 3, True, 7),
    ("r_104", "Köşe Pide", "pide pideci lahmacun", 6, True, 6),
    ("r_105", "Nar Kebap", "kebap kebapçı ocakbaşı adana", 9, True, 9),
    ("r_106", "Asma Kahve", "kahve kahveci kafe tatlı", 2, True, 8),
    ("r_107", "Mavi Burger", "burger burgerci hamburger", 8, False, 7),
    ("r_108", "Fırın Napoli", "pizza pizzacı italyan makarna", 11, True, 8),
    ("r_109", "Yeşil Kase", "vegan salata bowl glutensiz", 12, True, 5),
    ("r_110", "Avlu Meyhanesi", "meyhane meze rakı", 14, False, 9),
    ("r_111", "Tatlıcı Hanım", "tatlı tatlıcı baklava künefe", 5, True, 6),
    ("r_112", "Ev Yemekleri Sofrası", "ev yemekleri lokanta esnaf çorba", 10, True, 7),
    ("r_113", "Sushi Ada", "sushi japon uzak doğu ramen", 16, True, 6),
    ("r_114", "Çiğ Köfteci Ali", "çiğ köfte sokak lezzetleri dürüm", 4, True, 5),
)
#: ``(id, name, price, venue id)``.
DISHES = (
    ("m_2001", "Serpme Kahvaltı", 450, "r_101"), ("m_2002", "Menemen", 140, "r_101"),
    ("m_2003", "Izgara Levrek", 520, "r_102"), ("m_2004", "Kalamar Tava", 340, "r_102"),
    ("m_2005", "Et Döner Dürüm", 190, "r_103"), ("m_2006", "Tavuk Döner Dürüm", 130, "r_103"),
    ("m_2007", "Kıymalı Pide", 210, "r_104"), ("m_2008", "Lahmacun", 95, "r_104"),
    ("m_2009", "Adana Kebap", 380, "r_105"), ("m_2010", "Beyti", 420, "r_105"),
    ("m_2011", "Latte", 120, "r_106"), ("m_2012", "Filtre Kahve", 90, "r_106"),
    ("m_2013", "San Sebastian Cheesecake", 220, "r_106"),
    ("m_2014", "Cheeseburger", 310, "r_107"), ("m_2015", "Patates Kızartması", 110, "r_107"),
    ("m_2016", "Margherita", 290, "r_108"), ("m_2017", "Penne Arrabbiata", 260, "r_108"),
    ("m_2018", "Falafel Kase", 240, "r_109"), ("m_2019", "Mercimek Çorbası", 85, "r_112"),
    ("m_2020", "Fıstıklı Baklava", 260, "r_111"), ("m_2021", "Künefe", 230, "r_111"),
    ("m_2022", "Kuru Fasulye", 160, "r_112"), ("m_2023", "Somon Maki", 330, "r_113"),
    ("m_2024", "Tavuk Ramen", 320, "r_113"), ("m_2025", "Çiğ Köfte Dürüm", 80, "r_114"),
    ("m_2026", "Mantı", 230, "r_112"), ("m_2027", "Trileçe", 150, "r_111"),
    ("m_2028", "Mercimek Çorbası", 70, "r_103"), ("m_2029", "Türk Kahvesi", 75, "r_106"),
)
#: ``(venue id, what posts say about it)`` - what ``ask_about_posts`` reads.
OPINIONS = (
    ("r_106", "priz çok, laptopla saatlerce oturuluyor, sessiz"),
    ("r_101", "bahçesi geniş, çocuklu aileler rahat ediyor"),
    ("r_102", "deniz manzaralı, gün batımı çok güzel"),
    ("r_110", "canlı müzik var, kalabalık gruplar için iyi"),
    ("r_109", "kedi dostu, sakin, kitap okumalık"),
    ("r_108", "romantik, ışıklar loş, yıl dönümü için güzel"),
    ("r_105", "kalabalık ama servis hızlı, porsiyonlar büyük"),
)
POSTS = (
    ("ece.yer", "Buranın mantısı efsane, yoğurdu ev yapımı", "r_112"),
    ("gezgin_damak", "Lahmacun çıtır çıtır, 3 tane yedim", "r_104"),
    ("kahveci.baran", "Latte art çok iyi, kahve dengeli", "r_106"),
    ("deniz_k", "Künefe sıcak geldi, peyniri bol", "r_111"),
    ("yemekteyiz34", "Adana acılı ve sulu, lavaş taze", "r_105"),
    ("sushisever", "Somon maki taze ama porsiyon küçük", "r_113"),
    ("burak.yer", "Burger eti iyi pişmiş, patates soğuktu", "r_107"),
)
SAVED = (
    ("restaurant", "r_106"), ("restaurant", "r_102"), ("restaurant", "r_108"),
    ("menu_item", "m_2020"), ("menu_item", "m_2009"), ("menu_item", "m_2011"),
    ("post", 0), ("post", 3),
)
#: ``(direction, friend, item type, item id)``.
SHARED = (
    ("received", "Elif", "restaurant", "r_111"), ("received", "Elif", "menu_item", "m_2021"),
    ("received", "Kaan", "restaurant", "r_105"), ("sent", "Kaan", "restaurant", "r_102"),
    ("sent", "Zeynep", "menu_item", "m_2016"), ("received", "Mehmet", "restaurant", "r_108"),
)
FRIENDS = ("Ahmet", "Can", "Deniz", "Elif", "Kaan", "Mehmet", "Selin", "Zeynep")
SUGGESTED = (("Burak", 4), ("İrem", 3), ("Oğuz", 2), ("Dilara", 2))
MEMORY = (
    ("geçen hafta", "Fırın Napoli'yi önermiştim, Margherita'sı için"),
    ("3 gün önce", "vegan seçenek sormuştun, Yeşil Kase demiştim"),
    ("dün", "fıstığa alerjin olduğunu söylemiştin"),
)
_VENUE = {v[0]: v for v in VENUES}
_DISH = {d[0]: d for d in DISHES}


def _matches(query: object, *fields: str) -> bool:
    words = [w for w in fold(str(query or "")).split() if len(w) > 1]
    hay = fold(" ".join(fields))
    # "pideci" should find "pide", "kahveci" "kahve": match on the stem.
    return not words or any(w in hay or (len(w) > 4 and w[:-2] in hay) for w in words)


def _venue(row: tuple) -> dict:
    return venue_item(row[0], row[1], row[3], row[4])


def _dish(row: tuple) -> dict:
    return dish_item(row[0], row[1], row[2], _VENUE[row[3]][1])


def _flaky(name: str, args: dict) -> bool:
    """One call in forty fails, the same one every time: the app has to cope
    with a cache it cannot read, and so does the assistant."""
    digest = hashlib.sha256((name + repr(sorted(args.items()))).encode("utf-8")).digest()
    return digest[0] % 40 == 0


def run(name: str, args: dict, context: AgentContext | None = None) -> dict:
    """Execute a (validated) call against the invented neighbourhood."""
    if _flaky(name, args):
        return {"ok": False, "count": 0, "items": []}
    walk = args.get("max_walk_minutes")
    metres = args.get("max_distance_meters")

    def near(minutes: int) -> bool:
        return (not walk or minutes <= int(walk)) and (not metres or minutes * 80 <= int(metres))

    if name == "search_restaurants":
        rows = [v for v in VENUES if _matches(args.get("query"), v[1], v[2]) and near(v[3])]
        if args.get("tags"):
            rows = [v for v in rows if any(_matches(t, v[2]) for t in args["tags"])]
        if args.get("open_now"):
            rows = [v for v in rows if v[4]]
        key = {"popularity": lambda v: -v[5], "name": lambda v: v[1]}.get(
            args.get("sort_by"), lambda v: v[3])
        return result([_venue(v) for v in sorted(rows, key=key)])
    if name == "search_menu_items":
        rows = [d for d in DISHES if _matches(args.get("query"), d[1]) and near(_VENUE[d[3]][3])]
        if args.get("max_price") is not None:
            rows = [d for d in rows if d[2] <= float(args["max_price"])]
        if args.get("min_price") is not None:
            rows = [d for d in rows if d[2] >= float(args["min_price"])]
        if args.get("open_now"):
            rows = [d for d in rows if _VENUE[d[3]][4]]
        key = {"price_desc": lambda d: -d[2], "distance": lambda d: _VENUE[d[3]][3]}.get(
            args.get("sort_by"), lambda d: d[2])
        return result([_dish(d) for d in sorted(rows, key=key)])
    if name == "get_restaurant_details":
        venue = _VENUE.get(str(args.get("restaurant_id")))
        if venue is None:
            return result([])
        menu = [d[1] for d in DISHES if d[3] == venue[0]][:4]
        return result([{"id": venue[0], "name": venue[1], "open": venue[4],
                        "hours": "09:00-23:00" if venue[5] < 9 else "08:00-01:00",
                        "address": f"{venue[1]} Sk. No:{venue[3] * 3}", "menu": menu}])
    if name == "get_menu_item_details":
        dish = _DISH.get(str(args.get("menu_item_id")))
        if dish is None:
            return result([])
        item = _dish(dish)
        item["ingredients"] = "mevsim malzemeleri, ev yapımı sos"
        return result([item])
    if name == "get_directions":
        venue = _VENUE.get(str(args.get("restaurant_id")))
        return done(venue is not None, opened="maps", name=venue[1] if venue else "",
                    mode=args.get("mode") or "walk")
    if name == "set_active_location":
        if args.get("use_current_location"):
            return done(active_location=context.label if context else "GPS", is_gps=True)
        return done(active_location=str(args.get("location_query") or ""), is_gps=False)
    if name == "search_posts":
        rows = [p for p in POSTS if _matches(args.get("query"), p[1], _VENUE[p[2]][1])]
        return result([post_item(p[0], p[1], _VENUE[p[2]][1]) for p in rows])
    if name == "ask_about_posts":
        rows = [o for o in OPINIONS if _matches(args.get("question"), o[1])]
        return result([opinion_item(_VENUE[o[0]][1], o[1]) for o in rows])
    if name == "search_saved":
        kind = args.get("item_type") or "any"
        items = []
        for item_type, ref in SAVED:
            if kind not in ("any", item_type):
                continue
            if item_type == "restaurant":
                items.append(_venue(_VENUE[ref]))
            elif item_type == "menu_item":
                items.append(_dish(_DISH[ref]))
            else:
                post = POSTS[ref]
                items.append(post_item(post[0], post[1], _VENUE[post[2]][1]))
        items = [i for i in items if _matches(args.get("query"), *[str(v) for v in i.values()])]
        if args.get("sort_by") in ("price_asc", "price_desc"):
            items = sorted([i for i in items if "price" in i], key=lambda i: i["price"],
                           reverse=args["sort_by"] == "price_desc")
        elif args.get("sort_by") == "distance":
            items = sorted([i for i in items if "walk_min" in i], key=lambda i: i["walk_min"])
        return result(items)
    if name == "search_shared_items":
        direction, friend = args.get("direction") or "any", fold(str(args.get("friend_name") or ""))
        kind = args.get("item_type") or "any"
        items = []
        for way, who, item_type, ref in SHARED:
            if direction not in ("any", way) or kind not in ("any", item_type):
                continue
            if friend and friend not in fold(who):
                continue
            item = _venue(_VENUE[ref]) if item_type == "restaurant" else _dish(_DISH[ref])
            item["friend"] = who
            items.append(item)
        items = [i for i in items if _matches(args.get("query"), *[str(v) for v in i.values()])]
        return result(items)
    if name == "search_chat_memory":
        rows = [m for m in MEMORY if _matches(args.get("query"), m[1])]
        return result([{"when": m[0], "text": m[1]} for m in rows])
    if name == "list_friends":
        return result([{"name": f} for f in FRIENDS if _matches(args.get("query"), f)])
    if name == "get_suggested_users":
        return result([{"name": n, "mutual": m} for n, m in SUGGESTED])
    if name == "open_screen":
        return done(screen=str(args.get("screen") or ""))
    if name == "send_message_to_friends":
        return done(sent_to=list(args.get("friend_names") or []))
    return {"ok": False, "count": 0, "items": []}
