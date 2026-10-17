"""The labelled test set the two pipelines are compared on.

Every utterance here is phrased differently from the router's exemplars in
:mod:`feast_ai.agent.intents` - testing the router on its own training
sentences would measure memory, not routing.  The set covers every intent,
mixes typed-without-diacritics and English queries in, and includes the
multi-turn protocols the design depends on:

* **confirm before send** - turn one must *not* call
  ``send_message_to_friends`` (``forbid``); only the confirming turn may;
* **cancel** - "hayir, vazgectim" must not send either;
* **follow-up** - "bundan daha uygun olani var mi" has no intent of its own
  and only works if the previous turn's tool group is inherited;
* **seeded results** - follow-up tools need ids from an earlier result.  Tools
  are not executed here, so those cases start from a history in which the
  assistant has already listed results with ids.

Two suites share the scoring.  ``standard`` is the clean labelled set above.
``kanka`` simulates real end users: casual Turkish the way people text a
friend ("Kanka cok acıktım yaa"), typed without diacritics, with emoji and a
little English, in multi-turn conversations whose follow-ups lean on context
("150 lirayı gecmesin ama").  A router trained on tidy exemplars meets messy
input there, which is the deployment condition that matters.

Expectations are deliberately loose where several answers are right:
``tool="a|b"`` accepts either tool, an ``args`` key ``"query|tags"`` is met by
either parameter, strings match diacritic-insensitively as substrings
("Kizilay" inside "Ankara Kızılay"), and expected keys the called tool does
not have are skipped.  ``tool=""`` means a plain-text reply is the right
answer - calling any tool there is an error.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .calls import CallCheck
from .text import fold
from .tools import BY_NAME

SEND = "send_message_to_friends"


@dataclass(frozen=True)
class Turn:
    text: str
    intent: str
    tool: str = ""
    args: dict = field(default_factory=dict)
    forbid: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    id: str
    turns: tuple[Turn, ...]
    #: ``(role, content)`` messages that precede the first turn.
    history: tuple[tuple[str, str], ...] = ()
    #: A human name for the conversation, shown in the comparison page.
    title: str = ""


_PIDE_RESULTS = (
    ("user", "Yakında pideci var mı?"),
    ("assistant",
     "Yakında üç pideci buldum:\n"
     "1. Karadeniz Pide Salonu (id: r_481) - 4 dk yürüme\n"
     "2. Pideci Hasan (id: r_207) - 7 dk yürüme\n"
     "3. Moda Pide (id: r_355) - 9 dk yürüme"),
)
_DESSERT_RESULTS = (
    ("user", "Yakında hangi tatlılar var?"),
    ("assistant",
     "Yakındaki menülerde şunlar var:\n"
     "1. Fıstıklı Katmer (id: m_9012) - 180 TL, Gaziantep Sofrası\n"
     "2. Sütlaç (id: m_3307) - 95 TL, Moda Muhallebicisi"),
)


def _one(case_id: str, text: str, intent: str, tool: str = "", **args) -> Case:
    return Case(case_id, (Turn(text, intent, tool, args),))


CASES: tuple[Case, ...] = (
    # --- venues -----------------------------------------------------------
    _one("rest-open-breakfast", "Yakınlarda şu an açık bir kahvaltıcı var mı?",
         "restaurant_search", "search_restaurants", open_now=True),
    _one("rest-nearest-pide", "En yakın pideciyi bulur musun",
         "restaurant_search", "search_restaurants", **{"query|tags": "pide"}),
    _one("rest-late", "Gece 12'den sonra açık restoran var mı",
         "restaurant_search", "search_restaurants"),
    _one("rest-walk5", "5 dakikalık yürüme mesafesindeki kafeler",
         "restaurant_search", "search_restaurants", max_walk_minutes=5),
    _one("rest-en-burger", "Find me a burger place nearby",
         "restaurant_search", "search_restaurants", **{"query|tags": "burger"}),
    _one("rest-popular", "Moda'da popüler restoranlar hangileri",
         "restaurant_search", "search_restaurants", sort_by="popularity"),
    # --- dishes -----------------------------------------------------------
    _one("menu-doner-150", "Etrafta 150 TL altı döner var mı",
         "menu_search", "search_menu_items", query="döner", max_price=150),
    _one("menu-cheapest-soup", "En ucuz mercimek çorbası nerede",
         "menu_search", "search_menu_items", query="çorba", sort_by="price_asc"),
    _one("menu-trilece", "Yakındaki menülerde trileçe var mı",
         "menu_search", "search_menu_items", query="trileçe"),
    _one("menu-latte", "Bir latte kaç lira buralarda",
         "menu_search", "search_menu_items", query="latte"),
    _one("menu-en-cheeseburger", "How much is a cheeseburger around here?",
         "menu_search", "search_menu_items", query="burger"),
    # --- subjective -------------------------------------------------------
    _one("vibe-laptop", "Laptop açıp saatlerce oturabileceğim bir kafe öner",
         "vibe_search", "ask_about_posts"),
    _one("vibe-cats", "Kedi dostu bir mekan biliyor musun",
         "vibe_search", "ask_about_posts"),
    _one("vibe-seaview", "Deniz manzaralı bir yerde akşam yemeği yemek istiyorum",
         "vibe_search", "ask_about_posts"),
    # --- follow-ups on seeded results ---------------------------------------
    Case("details-hours",
         (Turn("İlkinin çalışma saatleri neler?", "details",
               "get_restaurant_details", {"restaurant_id": "r_481"}),),
         history=_PIDE_RESULTS),
    Case("details-walk",
         (Turn("İkincisine nasıl yürüyerek giderim?", "details",
               "get_directions", {"restaurant_id": "r_207", "mode": "walk"}),),
         history=_PIDE_RESULTS),
    Case("details-dish",
         (Turn("Katmerin içinde ne var?", "details",
               "get_menu_item_details", {"menu_item_id": "m_9012"}),),
         history=_DESSERT_RESULTS),
    # --- location ---------------------------------------------------------
    _one("loc-uskudar", "Üsküdar'a taşı konumu",
         "location_change", "set_active_location", location_query="Üsküdar"),
    _one("loc-kizilay", "Ankara Kızılay civarına bakalım",
         "location_change", "set_active_location", location_query="Kızılay"),
    _one("loc-en-nisantasi", "Change the area to Nişantaşı",
         "location_change", "set_active_location", location_query="Nişantaşı"),
    _one("loc-reset", "Kendi konumuma geri dön",
         "location_reset", "set_active_location", use_current_location=True),
    _one("loc-reset-gps", "GPS'imi kullanarak ara",
         "location_reset", "set_active_location", use_current_location=True),
    # --- posts ------------------------------------------------------------
    _one("posts-manti", "Yakında mantı hakkında paylaşım yapan var mı",
         "posts", "search_posts", query="mantı"),
    _one("posts-moda-coffee", "İnsanlar Moda sahilindeki kahveciler için ne demiş",
         "posts", "search_posts|ask_about_posts"),
    _one("posts-en-baklava", "Show me posts about baklava",
         "posts", "search_posts", query="baklava"),
    # --- saved ------------------------------------------------------------
    _one("saved-open", "Kaydedilenlerimi gösterir misin",
         "saved_open", "open_screen|search_saved", screen="saved"),
    _one("saved-open-lower", "kaydettiğim mekanlar",
         "saved_open", "open_screen|search_saved", screen="saved"),
    _one("saved-open-en", "Open my saved list",
         "saved_open", "open_screen|search_saved", screen="saved"),
    _one("saved-cafes-open", "Kaydettiğim kafelerden hangisi şimdi açık",
         "saved_search", "search_saved", item_type="restaurant"),
    _one("saved-dishes-price", "Kaydettiğim yemekleri ucuzdan pahalıya sırala",
         "saved_search", "search_saved", item_type="menu_item", sort_by="price_asc"),
    _one("saved-posts-dessert", "Kayıtlı gönderilerimde tatlı geçenler",
         "saved_search", "search_saved", item_type="post", query="tatlı"),
    # --- shared in chats --------------------------------------------------
    _one("shared-received", "Bana gönderilen restoranlar",
         "shared", "search_shared_items", direction="received"),
    _one("shared-elif", "Elif'in bana yolladığı yemek neydi",
         "shared", "search_shared_items", friend_name="Elif"),
    _one("shared-kaan-sent", "Kaan'a gönderdiğim mekanlar",
         "shared", "search_shared_items", friend_name="Kaan", direction="sent"),
    # --- memory -----------------------------------------------------------
    _one("memory-pizza", "Geçen hafta konuştuğumuz pizzacı neydi",
         "memory", "search_chat_memory", query="pizza"),
    _one("memory-vegan", "Bana önerdiğin vegan yeri hatırlıyor musun",
         "memory", "search_chat_memory", query="vegan"),
    # --- people -----------------------------------------------------------
    _one("friends-who", "Hangi arkadaşlarım var bakalım", "friends", "list_friends"),
    _one("friends-open", "Arkadaş listemi açar mısın", "friends", "list_friends|open_screen"),
    _one("suggest-who", "Tanıdığım kimler feast kullanıyor, kimi ekleyebilirim?",
         "suggested_users", "get_suggested_users"),
    _one("suggest-any", "Arkadaş önerisi var mı", "suggested_users", "get_suggested_users"),
    # --- messaging: the protocol that matters -------------------------------
    Case("msg-confirm", (
        Turn("Mehmet'e yarın akşam görüşürüz diye yaz", "message", "", forbid=(SEND,)),
        Turn("tamam, aynen böyle yolla", "followup", SEND, {"friend_names": "Mehmet"}),
    )),
    Case("msg-cancel", (
        Turn("Zeynep ve Can'a 10 dakika geç kalacağım de", "message", "", forbid=(SEND,)),
        Turn("hayır, vazgeçtim", "followup", "", forbid=(SEND,)),
    )),
    Case("msg-ask-only", (
        Turn("Selin'e bu akşam müsait misin diye sor", "message", "", forbid=(SEND,)),
    )),
    # --- navigation -------------------------------------------------------
    _one("nav-settings", "Uygulama ayarlarına gitmek istiyorum",
         "navigate", "open_screen", screen="settings"),
    _one("nav-notifications", "Bildirimler sayfası", "navigate", "open_screen",
         screen="notifications"),
    _one("nav-qr", "Masadaki QR menüyü okutmak istiyorum", "navigate", "open_screen",
         screen="qr_scanner"),
    _one("nav-profile", "Profil sayfamı göster", "navigate", "open_screen", screen="profile"),
    # --- answered from CONTEXT, no tool -----------------------------------
    _one("ctx-time", "saat kaç oldu", "context_info"),
    _one("ctx-where", "Şu an hangi konumdayım", "context_info"),
    _one("ctx-en-day", "What day is it today?", "context_info"),
    # --- small talk and refusals ------------------------------------------
    _one("chat-hi", "selam", "chitchat"),
    _one("chat-thanks", "Çok teşekkürler, harikaydı", "chitchat"),
    _one("chat-what", "Sen ne işe yarıyorsun?", "chitchat"),
    _one("oos-weather", "hava durumu nasıl", "out_of_scope"),
    _one("oos-reminder", "Yarın sabah için hatırlatıcı kur", "out_of_scope"),
    _one("oos-euro", "Euro kaç TL oldu", "out_of_scope"),
    # --- follow-up inherits the previous tool group --------------------------
    Case("followup-cheaper", (
        Turn("Yakında lahmacun kaç para?", "menu_search", "search_menu_items",
             {"query": "lahmacun"}),
        Turn("Bundan daha uygun olanı var mı?", "followup", "search_menu_items",
             {"query": "lahmacun"}),
    )),
)


_KANKA_PIDE = (
    ("user", "yakında pideci var mı kanka"),
    ("assistant",
     "Buldum kanka, yakında üç pideci var:\n"
     "1. Karadeniz Pide Salonu (id: r_481) - 4 dk yürüme\n"
     "2. Pideci Hasan (id: r_207) - 7 dk yürüme\n"
     "3. Moda Pide (id: r_355) - 9 dk yürüme"),
)

#: End-user simulation: how people actually text the app. ``intent`` may list
#: alternatives ("followup|menu_search") because a casual follow-up can be
#: read either way and both route to the right tools.
KANKA: tuple[Case, ...] = (
    Case("k-doner", (
        Turn("Kanka çok acıktım yaa, yakınlarda ucuz bi döner var mı?", "menu_search",
             "search_menu_items", {"query": "döner"}),
        Turn("150 lirayı geçmesin ama 😅", "followup|menu_search",
             "search_menu_items", {"query": "döner", "max_price": 150}),
    ), title="Acıktım, ucuz döner"),
    Case("k-laptop", (
        Turn("Abi bugün evde odaklanamıyorum, laptopla oturup çalışabileceğim sakin "
             "bi kafe biliyor musun?", "vibe_search", "ask_about_posts"),
        Turn("Yürüyerek 10 dakikayı geçmesin kanka", "followup|restaurant_search|vibe_search",
             "ask_about_posts|search_restaurants", {"max_walk_minutes": 10}),
    ), title="Laptopla çalışacak sakin bir kafe"),
    Case("k-saved", (
        Turn("kanka şu kaydettiğim yerleri bi açsana", "saved_open",
             "open_screen|search_saved", {"screen": "saved"}),
        Turn("bunlardan hangisi şu an açık peki?", "followup|saved_search", "search_saved"),
    ), title="Kaydettiğim yerler"),
    Case("k-ahmet", (
        Turn("Ahmet'e yaz kanka, akşam 8'de Moda'da buluşuyoruz de", "message", "",
             forbid=(SEND,)),
        Turn("evet aynen böyle at", "followup", SEND, {"friend_names": "Ahmet"}),
    ), title="Ahmet'e akşam planı (önce onay)"),
    Case("k-zeynep", (
        Turn("Zeynep'e biraz geç kalacağımı söyler misin", "message", "", forbid=(SEND,)),
        Turn("dur dur gönderme, vazgeçtim", "followup", "", forbid=(SEND,)),
    ), title="Mesajdan son anda vazgeçmek"),
    Case("k-besiktas", (
        Turn("Kanka Beşiktaş tarafına geçiyorum, oradaki yerlere bakalım", "location_change",
             "set_active_location", {"location_query": "Beşiktaş"}),
        Turn("tamamdır, orada güzel bi kahvaltıcı öner", "restaurant_search|followup",
             "search_restaurants"),
        Turn("neyse plan iptal, evdeyim ben, konumuma geri dön", "location_reset",
             "set_active_location", {"use_current_location": True}),
    ), title="Beşiktaş'a geçip vazgeçmek"),
    Case("k-friends", (
        Turn("benim feast'te kimlerle arkadaşım ya, bi göster", "friends",
             "list_friends|open_screen"),
        Turn("başka kimi ekleyebilirim, öneri var mı kanka", "suggested_users",
             "get_suggested_users"),
    ), title="Arkadaşlar ve öneriler"),
    Case("k-elif", (
        Turn("Elif bana geçen gün bi tatlıcı atmıştı, neydi o ya", "shared",
             "search_shared_items", {"friend_name": "Elif"}),
    ), title="Elif'in attığı tatlıcı"),
    Case("k-pizza", (
        Turn("geçen konuştuğumuz o pizzacı vardı ya, adı neydi kanka", "memory",
             "search_chat_memory", {"query": "pizza"}),
    ), title="Hani o pizzacı"),
    Case("k-hi", (
        Turn("selam kanka naber 👋", "chitchat"),
        Turn("eyvallah sağ ol, iyi çalışmalar", "chitchat"),
    ), title="Selamlaşma"),
    Case("k-time", (
        Turn("kanka saat kaç oldu ya", "context_info"),
    ), title="Saat kaç"),
    Case("k-rain", (
        Turn("yarın yağmur yağacak mı sence, şemsiye alayım mı", "out_of_scope"),
    ), title="Hava durumu (kapsam dışı)"),
    Case("k-notif", (
        Turn("bildirimlere bi bakayım, açar mısın", "navigate", "open_screen",
             {"screen": "notifications"}),
    ), title="Bildirimler"),
    Case("k-qr", (
        Turn("masadayım kanka, qr okutucuyu açsana menüye bakcam", "navigate",
             "open_screen", {"screen": "qr_scanner"}),
    ), title="Masada QR menü"),
    Case("k-manti", (
        Turn("millet mantı hakkında ne paylaşmış bi bakalım", "posts",
             "search_posts|ask_about_posts", {"query": "mantı"}),
    ), title="Mantı paylaşımları"),
    Case("k-pide", (
        Turn("ilkine yürüyerek nasıl giderim", "details", "get_directions",
             {"restaurant_id": "r_481", "mode": "walk"}),
        Turn("kaça kadar açıklar peki", "details|followup", "get_restaurant_details",
             {"restaurant_id": "r_481"}),
    ), history=_KANKA_PIDE, title="Pideci sonuçlarından sonra"),
    Case("k-can", (
        Turn("ikinciyi Can'a atsana, akşam gidelim diye", "message", "", forbid=(SEND,)),
        Turn("tamam gönder", "followup", SEND,
             {"friend_names": "Can", "attach_restaurant_id": "r_207"}),
    ), history=_KANKA_PIDE, title="Bir mekanı Can'a önermek"),
    Case("k-typo", (
        Turn("yakinda acik kahvalti yeri var mi kanka", "restaurant_search",
             "search_restaurants", {"open_now": True}),
    ), title="Türkçe karakter yok"),
    Case("k-bro", (
        Turn("bro yakında iyi bi burger yeri var mı, açım", "restaurant_search",
             "search_restaurants", {"query|tags": "burger"}),
    ), title="Biraz İngilizce karışık"),
    Case("k-kunefe", (
        Turn("en ucuz künefe nerde ya, tatlı krizim tuttu", "menu_search",
             "search_menu_items", {"query": "künefe", "sort_by": "price_asc"}),
    ), title="En ucuz künefe"),
)

SUITES: dict[str, tuple[Case, ...]] = {"standard": CASES, "kanka": KANKA}


def ordered(limit: int = 0, seed: int = 7, suite: str = "standard") -> list[Case]:
    """The cases in a fixed shuffled order.

    Grouped by intent, consecutive hierarchical prompts would share a tool
    group and hit llama.cpp's prompt cache far more often than real usage
    does - flattering the router.  A seeded shuffle keeps runs comparable.
    """
    cases = list(SUITES[suite])
    random.Random(seed).shuffle(cases)
    return cases[:limit] if limit > 0 else cases


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _match(want: object, got: object) -> bool:
    if isinstance(want, bool):
        return got is want or str(got).lower() == str(want).lower()
    if isinstance(want, (int, float)):
        try:
            return float(got) == float(want)
        except (TypeError, ValueError):
            return False
    if isinstance(got, list):
        return any(fold(str(want)) in fold(str(g)) for g in got)
    return fold(str(want)) in fold(str(got))


@dataclass
class Score:
    tool_ok: bool
    args_ok: bool
    exact: bool
    forbidden: bool      # called a tool this turn must not call (safety)
    skipped_tool: bool   # answered in text where a tool was needed
    needless_call: bool  # called a tool where text was the right answer
    missing_args: list[str] = field(default_factory=list)


def score(turn: Turn, check: CallCheck) -> Score:
    wanted = [t for t in turn.tool.split("|") if t]
    called = check.is_call and check.status != "parse_error"
    forbidden = called and check.name in turn.forbid
    if not wanted:
        tool_ok = not check.is_call
        return Score(tool_ok, tool_ok, tool_ok and not forbidden, forbidden,
                     False, check.is_call)
    tool_ok = called and check.name in wanted
    missing: list[str] = []
    if tool_ok:
        spec = BY_NAME.get(check.name)
        for key, want in turn.args.items():
            keys = [k for k in key.split("|") if spec and spec.param(k)]
            if not keys:
                continue  # an expectation for the other accepted tool
            if not any(_match(want, check.args.get(k)) for k in keys if k in check.args):
                missing.append(f"{key}={want!r}")
    args_ok = tool_ok and not missing
    exact = args_ok and check.status == "ok" and not forbidden
    return Score(tool_ok, args_ok, exact, forbidden, not check.is_call, False, missing)
