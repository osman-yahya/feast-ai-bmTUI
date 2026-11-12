"""Scenario cards for the blind suite: what the user wants, and the call that answers it.

The three other suites have a weakness each.  ``standard`` and ``kanka`` were
looked at while the v1 data was revised (clock times and "GPS" wording were
added after v0 missed them), and ``hybrid`` shares an author with the
training templates.  This suite is built so that neither can happen:

* a card states, in English, what the user wants and what the right call is.
  The dishes, districts, people and venue names on the cards are in none of
  the training pools and in none of the other suites;
* the *sentence* is written by a teacher model from the card's ``want`` line
  alone (:func:`mustachio.teacher.blind`) - it never sees a template, a tool
  name or a training sentence, so its wording is its own;
* every drafted sentence is checked by hand against its card before it goes
  into :mod:`mustachio.blind`, and the entities are added to the held-out
  list, so later training data cannot contain them either.

Nothing was tuned after this suite was first scored.
"""

from __future__ import annotations

SEND = "send_message_to_friends"
TO_ASSISTANT = "The user is talking to the assistant, not to the friend: they ask the assistant to "

VENUES = (
    ("user", "çevrede meze evi var mı"),
    ("assistant",
     "Yakında üç yer buldum:\n"
     "1. Tophane Meze Evi (id: r_914) - 6 dk yürüme\n"
     "2. Rıhtım Lokantası (id: r_937) - 9 dk yürüme\n"
     "3. Sarnıç Şarküteri (id: r_962) - 14 dk yürüme"),
)
DISHES = (
    ("user", "güllaç nerede var"),
    ("assistant",
     "Yakındaki menülerde şunlar var:\n"
     "1. Fıstıklı Güllaç (id: m_9521) - 210 TL, Tophane Meze Evi\n"
     "2. Narlı Güllaç (id: m_9544) - 185 TL, Rıhtım Lokantası"),
)


def _t(want: str, style: str, tool: str = "", intent: str = "", keep: tuple = (),
       forbid: tuple = (), **args) -> dict:
    return {"want": want, "style": style, "tool": tool, "intent": intent, "keep": list(keep),
            "forbid": forbid, "args": args}


def _c(card_id: str, *turns: dict, history: tuple = ()) -> dict:
    return {"id": card_id, "turns": list(turns), "history": history}


CARDS = [
    # --- venues ---------------------------------------------------------------
    _c("b-venue-1", _t("find a 'lokmacı' (a shop selling lokma) nearby", "casual",
                       "search_restaurants", "restaurant_search", ("lokmacı",),
                       **{"query|tags": "lokma"})),
    _c("b-venue-2", _t("find a Lebanese restaurant (Lübnan restoranı) that is open right now",
                       "hybrid", "search_restaurants", "restaurant_search", ("Lübnan",),
                       **{"query|tags": "lübnan", "open_now": True})),
    _c("b-venue-3", _t("find a 'meze evi' within 15 minutes on foot", "ascii",
                       "search_restaurants", "restaurant_search", ("meze evi", "15"),
                       **{"query|tags": "meze", "max_walk_minutes": 15})),
    _c("b-venue-4", _t("which Georgian restaurant (Gürcü restoranı) around here is the most "
                       "popular", "clean", "search_restaurants", "restaurant_search",
                       ("Gürcü",), **{"query|tags": "gürcü", "sort_by": "popularity"})),
    _c("b-venue-5", _t("find a Thai restaurant (Tayland restoranı) that is still open at 23:30",
                       "rambling", "search_restaurants", "restaurant_search",
                       ("Tayland", "23:30"), **{"query|tags": "tayland"})),
    _c("b-venue-6", _t("find the nearest 'şarküteri'", "terse", "search_restaurants",
                       "restaurant_search", ("şarküteri",),
                       **{"query|tags": "şarküteri", "sort_by": "distance"})),
    _c("b-venue-7", _t("find a tapas bar no more than 2 kilometres away", "hybrid",
                       "search_restaurants", "restaurant_search", ("tapas", "2 km"),
                       **{"query|tags": "tapas", "max_distance_meters": 2000})),
    _c("b-venue-8", _t("find any place that is open after 3 at night (03:00)", "casual",
                       "search_restaurants", "restaurant_search", ("3",))),
    # --- dishes and prices --------------------------------------------------------
    _c("b-dish-1", _t("where to find 'mumbar' (a dish) nearby", "casual", "search_menu_items",
                      "menu_search", ("mumbar",), query="mumbar")),
    _c("b-dish-2", _t("find 'ali nazik' (a dish) that costs at most 320 lira", "hybrid",
                      "search_menu_items", "menu_search", ("ali nazik", "320"),
                      query="ali nazik", max_price=320)),
    _c("b-dish-3", _t("where is the cheapest 'etli ekmek' (a dish)", "ascii",
                      "search_menu_items", "menu_search", ("etli ekmek",),
                      query="etli ekmek", sort_by="price_asc")),
    _c("b-dish-4", _t("how much does 'çılbır' (a dish) cost around here", "clean",
                      "search_menu_items", "menu_search", ("çılbır",), query="çılbır")),
    _c("b-dish-5", _t("craving 'höşmerim' (a dessert); where can it be found in a place that "
                      "is open now", "rambling", "search_menu_items", "menu_search",
                      ("höşmerim",), query="höşmerim", open_now=True)),
    _c("b-dish-6", _t("what is the most expensive 'tas kebabı' (a dish) around here", "terse",
                      "search_menu_items", "menu_search", ("tas kebabı",),
                      query="tas kebabı", sort_by="price_desc")),
    _c("b-dish-7", _t("find an 'affogato' (a coffee drink) within 10 minutes on foot", "hybrid",
                      "search_menu_items", "menu_search", ("affogato", "10"),
                      query="affogato", max_walk_minutes=10)),
    _c("b-dish-8", _t("is there 'kısır' (a dish) for less than 90 lira", "casual",
                      "search_menu_items", "menu_search", ("kısır", "90"),
                      query="kısır", max_price=90)),
    # --- vibe -------------------------------------------------------------------
    _c("b-vibe-1", _t("looking for a place where people do karaoke", "casual",
                      "ask_about_posts", "vibe_search")),
    _c("b-vibe-2", _t("looking for a place where one can watch the sunset", "hybrid",
                      "ask_about_posts", "vibe_search")),
    _c("b-vibe-3", _t("looking for an uncrowded, calm place with power sockets to charge a "
                      "phone", "rambling", "ask_about_posts", "vibe_search")),
    _c("b-vibe-4", _t("looking for a place that is comfortable to enter with a baby stroller",
                      "clean", "ask_about_posts", "vibe_search")),
    # --- follow-ups on listed results -----------------------------------------------
    _c("b-details-1",
       _t("ask until what time the second listed place is open", "casual",
          "get_restaurant_details", "details", restaurant_id="r_937"),
       _t("ask for directions to that same second place by car", "casual",
          "get_directions", "details", restaurant_id="r_937", mode="drive"),
       history=VENUES),
    _c("b-details-2",
       _t("ask to see the menu of Sarnıç Şarküteri (one of the listed places)", "hybrid",
          "get_restaurant_details", "details", ("Sarnıç Şarküteri",), restaurant_id="r_962"),
       history=VENUES),
    _c("b-details-3",
       _t("ask how to get to the first listed place by public transport", "ascii",
          "get_directions", "details", restaurant_id="r_914", mode="transit"),
       history=VENUES),
    _c("b-details-4",
       _t("ask what is inside the second listed dessert", "clean",
          "get_menu_item_details", "details", menu_item_id="m_9544"),
       history=DISHES),
    _c("b-details-5",
       _t("ask whether the first listed dessert contains nuts", "terse",
          "get_menu_item_details", "details", menu_item_id="m_9521"),
       history=DISHES),
    # --- the search area ----------------------------------------------------------
    _c("b-loc-1", _t("change the search area to Çukurcuma", "casual", "set_active_location",
                     "location_change", ("Çukurcuma",), location_query="Çukurcuma")),
    _c("b-loc-2", _t("going to Urla this weekend, so look at places there", "hybrid",
                     "set_active_location", "location_change", ("Urla",),
                     location_query="Urla")),
    _c("b-loc-3", _t("will be around Teşvikiye this evening, search according to there",
                     "ascii", "set_active_location", "location_change", ("Teşvikiye",),
                     location_query="Teşvikiye")),
    _c("b-loc-4", _t("go back to searching from the phone's real position", "casual",
                     "set_active_location", "location_reset", use_current_location=True)),
    _c("b-loc-5",
       _t("move the search area to Kaş", "terse", "set_active_location", "location_change",
          ("Kaş",), location_query="Kaş"),
       _t("now find a 'meze evi' there", "casual", "search_restaurants", "restaurant_search",
          ("meze evi",), **{"query|tags": "meze"})),
    # --- posts ------------------------------------------------------------------
    _c("b-posts-1", _t("see what people have posted about 'mumbar'", "casual",
                       "search_posts|ask_about_posts", "posts", ("mumbar",), query="mumbar")),
    _c("b-posts-2", _t("are there posts about 'frappe'", "hybrid", "search_posts", "posts",
                       ("frappe",), query="frappe")),
    # --- saved ------------------------------------------------------------------
    _c("b-saved-1", _t("open the list of saved items", "casual", "open_screen|search_saved",
                       "saved_open", screen="saved")),
    _c("b-saved-2", _t("among the saved dishes, which is the most expensive", "hybrid",
                       "search_saved", "saved_search", item_type="menu_item",
                       sort_by="price_desc")),
    _c("b-saved-3", _t("which venues did the user save in Fethiye", "clean", "search_saved",
                       "saved_search", ("Fethiye",), item_type="restaurant",
                       location_query="Fethiye")),
    _c("b-saved-4", _t("among the saved posts, find the ones that mention 'aşure'", "ascii",
                       "search_saved", "saved_search", ("aşure",), query="aşure")),
    _c("b-saved-5", _t("show the venues that were saved most recently", "terse",
                       "search_saved", "saved_search", sort_by="recency")),
    # --- shared in chats ----------------------------------------------------------
    _c("b-shared-1", _t("a while ago the user's friend Hatice shared a restaurant with "
                        "the user in a chat; the user wants the assistant to find which "
                        "restaurant it was", "casual", "search_shared_items", "shared", ("Hatice",),
                        friend_name="Hatice", direction="received")),
    _c("b-shared-2", _t("the user shared some dishes with their friend Koray in a chat "
                        "earlier and wants the assistant to list which dishes those were",
                        "hybrid",
                        "search_shared_items", "shared", ("Koray",), friend_name="Koray",
                        direction="sent")),
    _c("b-shared-3", _t("list all venues that friends have sent to the user in chats", "clean",
                        "search_shared_items", "shared", direction="received")),
    # --- memory -----------------------------------------------------------------
    _c("b-memory-1", _t("earlier the user asked the assistant about a 'şarküteri'; which one "
                        "was it", "casual", "search_chat_memory", "memory", ("şarküteri",),
                        query="şarküteri")),
    _c("b-memory-2", _t("what was the place the assistant recommended for 'kruvasan' last "
                        "time", "hybrid", "search_chat_memory", "memory", ("kruvasan",),
                        query="kruvasan")),
    # --- people -----------------------------------------------------------------
    _c("b-friends-1", _t("show who the user's friends are in the app", "casual",
                         "list_friends|open_screen", "friends")),
    _c("b-friends-2", _t("check whether a person named Yunus is among the user's friends",
                         "ascii", "list_friends", "friends", ("Yunus",), query="Yunus")),
    _c("b-suggest-1", _t("who could the user add as a new friend; any suggestions", "hybrid",
                         "get_suggested_users", "suggested_users")),
    # --- navigation ---------------------------------------------------------------
    _c("b-nav-1", _t("open the settings of the app", "casual", "open_screen", "navigate",
                     screen="settings")),
    _c("b-nav-2", _t("see the notifications", "hybrid", "open_screen", "navigate",
                     screen="notifications")),
    _c("b-nav-3", _t("scan the QR code on the restaurant table", "ascii", "open_screen",
                     "navigate", screen="qr_scanner")),
    _c("b-nav-4", _t("go to the user's own profile page", "terse", "open_screen", "navigate",
                     screen="profile")),
    _c("b-nav-5", _t("look at the map", "casual", "open_screen", "navigate", screen="map")),
    _c("b-nav-6", _t("open the chats / messages screen", "hybrid", "open_screen", "navigate",
                     screen="chats")),
    _c("b-nav-7", _t("open the screen for adding a new friend", "clean", "open_screen",
                     "navigate", screen="add_friends")),
    _c("b-nav-8", _t("go back to the home feed of posts", "casual", "open_screen", "navigate",
                     screen="feed")),
    # --- messaging: confirm first ---------------------------------------------------
    _c("b-msg-1",
       _t(TO_ASSISTANT + "send their friend Zehra a message saying the user will be "
          "there in 20 minutes", "casual", "", "message", ("Zehra", "20"), forbid=(SEND,)),
       _t("confirm: yes, send it", "casual", SEND, "followup", friend_names="Zehra")),
    _c("b-msg-2",
       _t(TO_ASSISTANT + "send their friend Hüseyin a message saying the dinner is "
          "cancelled", "hybrid", "", "message", ("Hüseyin",), forbid=(SEND,)),
       _t("change of mind: do not send it after all", "hybrid", "", "followup",
          forbid=(SEND,))),
    _c("b-msg-3",
       _t("tell the assistant to ask the friends Emine and Tarık whether they are free on "
          "Saturday", "ascii", "", "message", ("Emine", "Tarık"), forbid=(SEND,)),
       _t("confirm that it should be sent", "ascii", SEND, "followup",
          friend_names="Emine")),
    _c("b-msg-4",
       _t(TO_ASSISTANT + "send the first place in the list to their friend Bilge, with "
          "the note 'shall we go here tomorrow'", "casual", "", "message", ("Bilge",),
          forbid=(SEND,)),
       _t("confirm: ok, send", "casual", SEND, "followup", friend_names="Bilge",
          attach_restaurant_id="r_914"),
       history=VENUES),
    _c("b-msg-5",
       _t(TO_ASSISTANT + "send their friend Gamze a message wishing her a happy "
          "birthday", "clean", "", "message", ("Gamze",), forbid=(SEND,)),
       _t("say the wording should be different: write 'iyi ki doğdun canım' instead", "clean",
          "", "followup", ("iyi ki doğdun canım",), forbid=(SEND,)),
       _t("confirm that this new version should be sent", "clean", SEND, "followup",
          friend_names="Gamze")),
    # --- no tool: context, small talk, things feast cannot do ---------------------------
    _c("b-ctx-1", _t("ask what time it is", "hybrid", "", "context_info")),
    _c("b-ctx-2", _t("ask what the date is today", "casual", "", "context_info")),
    _c("b-ctx-3", _t("ask which area the assistant is currently searching in", "ascii", "",
                     "context_info")),
    _c("b-talk-1", _t("greet the assistant", "casual", "", "chitchat")),
    _c("b-talk-2", _t("thank the assistant, it was very helpful", "hybrid", "", "chitchat")),
    _c("b-talk-3", _t("ask the assistant what it can do", "ascii", "", "chitchat")),
    _c("b-talk-4", _t("say goodbye", "terse", "", "chitchat")),
    _c("b-oos-1", _t("ask the assistant to book a hotel room for the weekend", "casual", "",
                     "out_of_scope")),
    _c("b-oos-2", _t("ask which pharmacy is on night duty nearby", "clean", "",
                     "out_of_scope")),
    _c("b-oos-3", _t("ask how much petrol costs today", "hybrid", "", "out_of_scope")),
    _c("b-oos-4", _t("ask the assistant to pay the user's credit card bill", "ascii", "",
                     "out_of_scope")),
    # --- a follow-up that only adds a condition -----------------------------------------
    _c("b-follow-1",
       _t("where to find 'perde pilavı' (a dish) nearby", "casual", "search_menu_items",
          "menu_search", ("perde pilavı",), query="perde pilavı"),
       _t("add a condition: it should not cost more than 400 lira", "casual",
          "search_menu_items", "followup|menu_search", ("400",), query="perde pilavı",
          max_price=400)),
    _c("b-follow-2",
       _t("find a 'bowl restoranı' nearby", "hybrid", "search_restaurants",
          "restaurant_search", ("bowl",), **{"query|tags": "bowl"}),
       _t("add a condition: only the ones that are open right now", "hybrid",
          "search_restaurants", "followup|restaurant_search", open_now=True)),
    _c("b-follow-3",
       _t("how much is 'mocha' (a coffee drink) around here", "ascii", "search_menu_items",
          "menu_search", ("mocha",), query="mocha"),
       _t("ask for the cheapest one", "ascii", "search_menu_items", "followup|menu_search",
          query="mocha", sort_by="price_asc")),
    # --- a chat about more than one thing -----------------------------------------------
    _c("b-mixed-1",
       _t("change the search area to Yeldeğirmeni", "casual", "set_active_location",
          "location_change", ("Yeldeğirmeni",), location_query="Yeldeğirmeni"),
       _t("find a sandwich shop (sandviççi) there", "casual", "search_restaurants",
          "restaurant_search", ("sandviççi",), **{"query|tags": "sandviç"}),
       _t(TO_ASSISTANT + "send their friend Utku a message saying let's meet at 7 in "
          "the evening", "casual", "", "message", ("Utku", "7"), forbid=(SEND,)),
       _t("confirm: yes send it", "casual", SEND, "followup", friend_names="Utku"),
       _t("ask what time it is now", "casual", "", "context_info")),
    _c("b-mixed-2",
       _t("is there 'irmik helvası' (a dessert) nearby", "hybrid",
          "search_menu_items|search_restaurants", "menu_search", ("irmik helvası",),
          query="irmik helvası"),
       _t("open the notifications", "hybrid", "open_screen", "navigate",
          screen="notifications"),
       _t("which venues did the friend Meryem send to the user", "hybrid",
          "search_shared_items", "shared", ("Meryem",), friend_name="Meryem",
          direction="received"),
       _t("thank the assistant", "hybrid", "", "chitchat")),
    _c("b-mixed-3",
       _t("greet the assistant", "ascii", "", "chitchat"),
       _t("find the cheapest 'kefir' (a drink)", "ascii", "search_menu_items", "menu_search",
          ("kefir",), query="kefir", sort_by="price_asc"),
       _t(TO_ASSISTANT + "send their friend Damla a message saying the user is running "
          "late", "ascii", "", "message", ("Damla",), forbid=(SEND,)),
       _t("cancel it, do not send", "ascii", "", "followup", forbid=(SEND,)),
       _t("open the map", "ascii", "open_screen", "navigate", screen="map")),
]

#: Dishes, places and people on the cards - never used to build training data.
ENTITIES = (
    "lokmacı", "lübnan", "meze evi", "gürcü", "tayland", "şarküteri", "tapas", "mumbar",
    "ali nazik", "etli ekmek", "çılbır", "höşmerim", "tas kebabı", "affogato", "kısır",
    "karaoke", "gün batımı", "bebek arabası", "çukurcuma", "urla", "teşvikiye", "kaş",
    "frappe", "fethiye", "aşure", "hatice", "koray", "kruvasan", "yunus", "zehra", "hüseyin",
    "emine", "tarık", "bilge", "gamze", "perde pilavı", "bowl", "mocha", "yeldeğirmeni",
    "sandviççi", "utku", "irmik helvası", "meryem", "kefir", "damla", "tophane", "rıhtım",
    "sarnıç", "güllaç",
)
