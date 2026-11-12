"""The hybrid suite: Turkish the way it is typed, with English mixed in.

feast AI's ``kanka`` suite has one "a bit of English" turn.  This suite is
that turn's whole population: 69 turns in 62 chats where the sentence is Turkish -
its grammar, its suffixes, its word order - and some of its words are not.
"kanka yarın kadıköye lets go" is Turkish; a model that answers it in
English, or reads "lets go" as a venue name, has been confused by it.

The mixing follows how it is actually done, not word substitution:

* an English stem with a Turkish light verb or suffix: "check etsene",
  "save'lediklerim", "location'ı değiştir", "cancel la";
* English discourse words around a Turkish clause: "btw", "pls", "lets go",
  "no way", "tbh", "asap";
* English nouns for app things and vibes: settings, notification, feed,
  spot, place, cozy, budget, nearby, open;
* typed fast: no diacritics, no apostrophes, lower case.

It is held out twice.  No sentence here is used for training, and none of
its dishes, districts or names is in the training lexicon either
(:data:`mustachio.lexicon.HELD_OUT` is built from this file and from feast
AI's two suites, and the generator refuses any slot value that matches).
A turn solved here was therefore solved for a dish, a place and a person
the fine-tune never saw.

The expectations use feast AI's scoring unchanged (``tool="a|b"`` accepts
either tool, strings match diacritic-insensitively as substrings).  Every
text turn is also checked for its reply language (see
:func:`mustachio.evaluate.reply_language`): a Turkish matrix sentence must
be answered in Turkish.
"""

from __future__ import annotations

from feast_ai.agent.suite import SEND, Case, Turn

_RESULTS = (
    ("user", "yakında iskender nerde yenir"),
    ("assistant",
     "Yakında üç yer buldum:\n"
     "1. Uludağ Kebapçısı (id: r_612) - 5 dk yürüme\n"
     "2. Bursa Garajı (id: r_138) - 8 dk yürüme\n"
     "3. Yeşil İskender (id: r_774) - 11 dk yürüme"),
)
_DISHES = (
    ("user", "buralarda cheesecake var mı"),
    ("assistant",
     "Yakındaki menülerde şunlar var:\n"
     "1. San Sebastian Cheesecake (id: m_4410) - 240 TL, Viyana Kahvesi\n"
     "2. Limonlu Cheesecake (id: m_2086) - 195 TL, Pasta Atölyesi"),
)


def _one(case_id: str, text: str, intent: str, tool: str = "", **args) -> Case:
    return Case(case_id, (Turn(text, intent, tool, args),))


HYBRID: tuple[Case, ...] = (
    # --- the search area ----------------------------------------------------
    _one("h-letsgo", "kanka yarın kadıköye lets go", "location_change",
         "set_active_location", location_query="Kadıköy"),
    _one("h-location-change", "location'ı Bebek yapsana pls", "location_change",
         "set_active_location", location_query="Bebek"),
    _one("h-heading", "bu aksam alsancak tarafindayim btw, oraya gore bakalim",
         "location_change", "set_active_location", location_query="Alsancak"),
    _one("h-switch", "search area'yı Taksim'e switch et", "location_change",
         "set_active_location", location_query="Taksim"),
    _one("h-gps", "ok current location'a geri dön", "location_reset",
         "set_active_location", use_current_location=True),
    _one("h-gps2", "nvm evdeyim, gps'e göre devam", "location_reset",
         "set_active_location", use_current_location=True),
    # --- venues -------------------------------------------------------------
    _one("h-sushi", "nearby iyi bi sushi place var mı ya", "restaurant_search",
         "search_restaurants", **{"query|tags": "sushi"}),
    _one("h-open", "şu an open olan bi tantuni yeri lazım asap", "restaurant_search",
         "search_restaurants|search_menu_items", query="tantuni", open_now=True),
    _one("h-walk", "walking distance 10 dk içinde waffle spot bul", "restaurant_search",
         "search_restaurants|search_menu_items", query="waffle", max_walk_minutes=10),
    _one("h-popular", "burda en popular kumpirci hangisi", "restaurant_search",
         "search_restaurants", **{"query|tags": "kumpir"}, sort_by="popularity"),
    _one("h-late", "gece 2de open olan bi yer var mı, after party açlığı 😅",
         "restaurant_search", "search_restaurants", open_at="02:00"),
    # --- dishes and prices --------------------------------------------------
    _one("h-budget", "max 200 TL budget'ım var, iskender nerde yenir", "menu_search",
         "search_menu_items", query="iskender", max_price=200),
    _one("h-cheapest", "en cheap çiğ köfte nerde abi", "menu_search",
         "search_menu_items", query="çiğ köfte", sort_by="price_asc"),
    _one("h-price", "midye dolma price ne kadar buralarda", "menu_search",
         "search_menu_items", query="midye"),
    _one("h-craving", "canım fena cheesecake çekti, nearby nerde var", "menu_search",
         "search_menu_items", query="cheesecake"),
    _one("h-pricey", "en pricey balık ekmek kaç lira merak ettim lol", "menu_search",
         "search_menu_items", query="balık ekmek", sort_by="price_desc"),
    # --- vibe ---------------------------------------------------------------
    _one("h-date", "first date için romantic bi mekan lazım, öneri?", "vibe_search",
         "ask_about_posts"),
    _one("h-rooftop", "rooftop'u olan chill bi yer biliyo musun", "vibe_search",
         "ask_about_posts"),
    _one("h-livemusic", "live music olan mekanlar var mı bu civarda", "vibe_search",
         "ask_about_posts"),
    _one("h-group", "10 kişilik crowded bi group için uygun yer bulsana", "vibe_search",
         "ask_about_posts"),
    # --- follow-ups on listed results ----------------------------------------
    Case("h-details", (
        Turn("ilkinin working hours'ı ne", "details", "get_restaurant_details",
             {"restaurant_id": "r_612"}),
        Turn("ok ikinciye navigate et, arabayla", "details", "get_directions",
             {"restaurant_id": "r_138", "mode": "drive"}),
    ), history=_RESULTS),
    Case("h-menu", (
        Turn("sonuncunun menu'süne bi bakayım", "details", "get_restaurant_details",
             {"restaurant_id": "r_774"}),
    ), history=_RESULTS),
    Case("h-ingredients", (
        Turn("ilk cheesecake'in içinde ne var, gluten free mi", "details",
             "get_menu_item_details", {"menu_item_id": "m_4410"}),
    ), history=_DISHES),
    Case("h-transit", (
        Turn("3. mekana public transport'la nasıl gidilir", "details", "get_directions",
             {"restaurant_id": "r_774", "mode": "transit"}),
    ), history=_RESULTS),
    # --- posts --------------------------------------------------------------
    _one("h-posts", "tantuni hakkında post atan var mı nearby", "posts",
         "search_posts", query="tantuni"),
    _one("h-posts2", "waffle postlarına bakalım bi", "posts", "search_posts", query="waffle"),
    # --- saved --------------------------------------------------------------
    _one("h-saved", "saved'lerimi açsana", "saved_open", "open_screen|search_saved",
         screen="saved"),
    _one("h-saved-list", "save'lediklerimi check edelim bi", "saved_open",
         "open_screen|search_saved", screen="saved"),
    _one("h-saved-dishes", "save ettiğim yemekleri cheap'ten pricey'e sırala", "saved_search",
         "search_saved", item_type="menu_item", sort_by="price_asc"),
    _one("h-saved-near", "saved mekanlardan en nearby olan hangisi", "saved_search",
         "search_saved", item_type="restaurant", sort_by="distance"),
    _one("h-saved-posts", "save'lediğim postlarda sushi geçen var mı", "saved_search",
         "search_saved", item_type="post", query="sushi"),
    # --- shared in chats ----------------------------------------------------
    _one("h-shared", "Burcu bana geçen hafta bi mekan share etmişti, neydi o", "shared",
         "search_shared_items", friend_name="Burcu", direction="received"),
    _one("h-shared-sent", "Tolga'ya forwardladığım yerler hangileriydi", "shared",
         "search_shared_items", friend_name="Tolga", direction="sent"),
    _one("h-shared-dm", "dm'den bana atılan restoranları göster", "shared",
         "search_shared_items", direction="received"),
    # --- memory -------------------------------------------------------------
    _one("h-memory", "btw geçen sana sorduğum kumpirci neydi ya", "memory",
         "search_chat_memory", query="kumpir"),
    _one("h-memory2", "bana recommend ettiğin sushi yerini hatırlıyo musun", "memory",
         "search_chat_memory", query="sushi"),
    # --- people -------------------------------------------------------------
    _one("h-friends", "friend listemde kimler var bi bakayım", "friends",
         "list_friends|open_screen"),
    _one("h-friend-check", "Dilara benim friend'im mi feast'te", "friends",
         "list_friends", query="Dilara"),
    _one("h-suggest", "kimleri add edebilirim, suggestion var mı", "suggested_users",
         "get_suggested_users"),
    # --- navigation ---------------------------------------------------------
    _one("h-settings", "settings'e girsene bi", "navigate", "open_screen", screen="settings"),
    _one("h-notif", "notification'lara bakıcam, aç pls", "navigate", "open_screen",
         screen="notifications"),
    _one("h-qr", "masadayım qr scan edicem", "navigate", "open_screen", screen="qr_scanner"),
    _one("h-feed", "feed'e dön", "navigate", "open_screen", screen="feed"),
    _one("h-chats", "chatlerimi aç bakalım", "navigate", "open_screen", screen="chats"),
    _one("h-map", "map view'a geçelim", "navigate", "open_screen", screen="map"),
    _one("h-profile", "profile git", "navigate", "open_screen", screen="profile"),
    # --- messaging: confirm first, in two languages at once -------------------
    Case("h-msg", (
        Turn("Oğuz'a text at, 5 dk late kalıyorum de", "message", "", forbid=(SEND,)),
        Turn("yes aynen gönder", "followup", SEND, {"friend_names": "Oğuz"}),
    )),
    Case("h-msg-cancel", (
        Turn("İrem'e yaz: plan cancel oldu sorry", "message", "", forbid=(SEND,)),
        Turn("nvm boşver gönderme", "followup", "", forbid=(SEND,)),
    )),
    Case("h-msg-okay", (
        Turn("kerem ve yagmura mesaj at, meeting 9a alindi fyi", "message", "",
             forbid=(SEND,)),
        Turn("ok send", "followup", SEND, {"friend_names": "Kerem"}),
    )),
    Case("h-msg-venue", (
        Turn("ikinci mekanı Dilara'ya share et, weekend'de gidelim mi diye", "message", "",
             forbid=(SEND,)),
        Turn("okey at gitsin", "followup", SEND,
             {"friend_names": "Dilara", "attach_restaurant_id": "r_138"}),
    ), history=_RESULTS),
    # --- no tool: context, small talk, refusals --------------------------------
    _one("h-time", "saat kaç oldu bro", "context_info"),
    _one("h-where", "şu an location'ım neresi görünüyo", "context_info"),
    _one("h-day", "bugün günlerden ne ya, haftayı karıştırdım", "context_info"),
    _one("h-hi", "hey naber, what's up", "chitchat"),
    _one("h-thanks", "thx kanka çok helpful oldun", "chitchat"),
    _one("h-what", "sen tam olarak ne yapabiliyosun, feature'ların ne", "chitchat"),
    _one("h-nice", "nice, perfect oldu bu 🙏", "chitchat"),
    _one("h-oos-flight", "bana yarına uçak bileti book etsene", "out_of_scope"),
    _one("h-oos-music", "spotify'dan bi playlist aç", "out_of_scope"),
    _one("h-oos-score", "dünkü maçın skoru kaç kaçtı", "out_of_scope"),
    # --- a follow-up that only adds a condition --------------------------------
    Case("h-followup", (
        Turn("buralarda kumpir kaça", "menu_search", "search_menu_items", {"query": "kumpir"}),
        Turn("150'yi geçmesin ama, budget kısıtlı", "followup|menu_search",
             "search_menu_items", {"query": "kumpir", "max_price": 150}),
    )),
    Case("h-followup-open", (
        Turn("nearby sushi place bul", "restaurant_search", "search_restaurants",
             {"query|tags": "sushi"}),
        Turn("sadece şu an open olanlar", "followup|restaurant_search", "search_restaurants",
             {"open_now": True}),
    )),
)

#: Dishes, places and names this suite tests - never used to build training data.
ENTITIES = (
    "kadıköy", "bebek", "alsancak", "taksim", "sushi", "tantuni", "waffle", "kumpir",
    "iskender", "çiğ köfte", "midye", "cheesecake", "balık ekmek", "burcu", "tolga",
    "dilara", "oğuz", "irem", "kerem", "yağmur", "uludağ", "bursa garajı",
)
#: Vibe wordings this suite tests - the training qualities avoid them, so a
#: vibe turn here is a concept the fine-tune was not shown.
CONCEPTS = ("romantic", "romantik", "rooftop", "live music", "canlı müzik", "first date",
            "crowded", "teras", "terrace")
