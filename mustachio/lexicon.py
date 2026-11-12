"""Slot values for the training generator, and the entities it may not use.

**Held out twice.**  feast AI's rule is that nothing tuned may paraphrase a
suite sentence.  A fine-tune can break that rule in a second way - by
memorising the suites' *entities*: if "döner", "Üsküdar" and "Elif" are in
the training data, a correct ``query="döner"`` on the suite says little
about dishes the model has not seen.  So every dish, venue kind, district,
person and vibe that the three suites test is listed in :data:`HELD_OUT`,
and :func:`check_pools` fails the build if a pool below contains one.  A
suite turn the fine-tuned model gets right was got right for an entity it
never trained on.  (For the final production fit, when measuring is over,
they can simply be added to the pools.)

**Turkish suffixes.**  Templates need "Karaköy'e", "Balat'ta", "Onur'un":
the case endings follow vowel harmony and consonant voicing, so they are
computed (:func:`dative` and friends), not listed.  Place names that are
possessive compounds (Beyoğlu'na, Eminönü'nde) take an extra buffer and are
left out of the pools rather than special-cased.
"""

from __future__ import annotations

import re

from feast_ai.agent.text import fold

from .blind_cards import ENTITIES as BLIND_ENTITIES
from .hybrid import CONCEPTS as HYBRID_CONCEPTS
from .hybrid import ENTITIES as HYBRID_ENTITIES

# --------------------------------------------------------------------------
# Held-out entities
# --------------------------------------------------------------------------
#: What feast AI's ``standard`` and ``kanka`` suites test, folded.
SUITE_ENTITIES = (
    "kahvaltı", "pide", "burger", "döner", "çorba", "mercimek", "trileçe", "latte",
    "katmer", "mantı", "baklava", "pizza", "vegan", "lahmacun", "künefe", "sütlaç",
    "tatlı", "kafe", "moda", "üsküdar", "kızılay", "ankara", "nişantaşı", "beşiktaş",
    "elif", "kaan", "mehmet", "zeynep", "can", "selin", "ahmet", "hasan", "laptop",
    "kedi", "deniz manzara", "sakin", "gaziantep", "karadeniz", "hava durumu",
    "hatırlatıcı", "euro", "yağmur",
)
HELD_OUT = tuple(sorted({fold(w) for w in SUITE_ENTITIES + HYBRID_ENTITIES
                         + HYBRID_CONCEPTS + BLIND_ENTITIES}))
_WORD = re.compile(r"[a-z0-9]+")


def held_out_hit(text: str) -> str:
    """The held-out entity ``text`` mentions, or ``""``.

    Single words match a whole word or its stem ("pideci" hits "pide",
    "cancel" does not hit "can"); phrases match as substrings.
    """
    folded = fold(text)
    words = _WORD.findall(folded.replace("'", " "))
    for entity in HELD_OUT:
        if " " in entity:
            if entity in folded:
                return entity
        elif any(w == entity or (len(entity) >= 4 and w.startswith(entity)) for w in words):
            return entity
    return ""


# --------------------------------------------------------------------------
# Suffixes
# --------------------------------------------------------------------------
_VOWELS = "aeıioöuü"
_BACK = "aıou"
_VOICELESS = "fstkçşhp"


def _last_vowel(word: str) -> str:
    for char in reversed(word.replace("İ", "i").replace("I", "ı").lower()):
        if char in _VOWELS:
            return char
    return "e"


def _ends_vowel(word: str) -> bool:
    return word[-1].lower() in _VOWELS


#: Last words that are already possessive ("Köşe Evi"): they take -n-, not -y-.
POSSESSIVE = ("Meyhanesi", "Lokantası", "Sofrası", "Evi", "Köftecisi", "Fırını", "Ocakbaşı")


def dative(word: str) -> str:
    """Karaköy'e, Galata'ya, Onur'a, Ece'ye - and Köşe Evi'ne."""
    vowel = "a" if _last_vowel(word) in _BACK else "e"
    if word.split()[-1] in POSSESSIVE:
        return f"{word}'n{vowel}"
    return f"{word}'{'y' if _ends_vowel(word) else ''}{vowel}"


def locative(word: str) -> str:
    """Karaköy'de, Balat'ta, Levent'te, Galata'da."""
    vowel = "a" if _last_vowel(word) in _BACK else "e"
    return f"{word}'{'t' if word[-1].lower() in _VOICELESS else 'd'}{vowel}"


def ablative(word: str) -> str:
    """Karaköy'den, Balat'tan."""
    return locative(word) + "n"


def genitive(word: str) -> str:
    """Onur'un, Ece'nin, Fatma'nın, Gül'ün."""
    vowel = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u",
             "ö": "ü", "ü": "ü"}[_last_vowel(word)]
    return f"{word}'{'n' if _ends_vowel(word) else ''}{vowel}n"


CASES = {"dat": dative, "loc": locative, "abl": ablative, "gen": genitive}

# --------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------
#: ``(as the user says it, the query the app should search for)``.
KINDS = (
    ("kebapçı", "kebapçı"), ("balıkçı", "balıkçı"), ("meyhane", "meyhane"),
    ("kahveci", "kahveci"), ("köfteci", "köfteci"), ("ocakbaşı", "ocakbaşı"),
    ("esnaf lokantası", "esnaf lokantası"), ("börekçi", "börekçi"), ("tostçu", "tostçu"),
    ("dondurmacı", "dondurmacı"), ("pastane", "pastane"), ("kokoreççi", "kokoreççi"),
    ("çay bahçesi", "çay bahçesi"), ("steakhouse", "steakhouse"), ("ramenci", "ramenci"),
    ("makarnacı", "makarnacı"), ("ciğerci", "ciğerci"), ("pilavcı", "pilavcı"),
    ("dürümcü", "dürümcü"), ("falafelci", "falafelci"), ("simitçi", "simitçi"),
    ("gözlemeci", "gözlemeci"), ("fırın", "fırın"), ("bar", "bar"), ("pub", "pub"),
    ("şarap evi", "şarap evi"), ("kokteyl barı", "kokteyl barı"),
    ("İtalyan restoranı", "İtalyan"), ("Meksika restoranı", "Meksika"),
    ("Kore restoranı", "Kore"), ("Hint restoranı", "Hint"), ("Çin restoranı", "Çin"),
    ("Japon restoranı", "Japon"), ("Uzak Doğu restoranı", "Uzak Doğu"),
    ("tavukçu", "tavukçu"), ("nargileci", "nargileci"), ("büfe", "büfe"),
)
#: English venue words -> the Turkish query (the data they search is Turkish).
KINDS_EN = (
    ("kebab place", "kebap"), ("fish restaurant", "balık"), ("coffee shop", "kahve"),
    ("steakhouse", "steakhouse"), ("ramen place", "ramen"), ("bar", "bar"),
    ("bakery", "fırın"), ("ice cream shop", "dondurma"), ("Italian restaurant", "İtalyan"),
    ("meatball place", "köfte"), ("pub", "pub"),
)
DISHES = (
    "kokoreç", "adana kebap", "urfa kebap", "beyti", "cağ kebabı", "hünkar beğendi",
    "karnıyarık", "kuru fasulye", "menemen", "gözleme", "su böreği", "poğaça", "simit",
    "tost", "ayvalık tostu", "sandviç", "wrap", "sezar salata", "makarna", "ramen",
    "noodle", "taco", "falafel", "humus", "kumru", "boyoz", "çiğ börek", "içli köfte",
    "köfte", "tavuk şiş", "kanat", "ciğer", "levrek", "hamsi tava", "kalamar", "karides",
    "nohut pilav", "mücver", "sarma", "börek", "lokma", "tiramisu", "profiterol",
    "magnolia", "dondurma", "kazandibi", "revani", "şekerpare", "tulumba", "brownie",
    "türk kahvesi", "filtre kahve", "americano", "cappuccino", "flat white", "espresso",
    "çay", "ayran", "limonata", "salep", "boza", "şalgam", "smoothie", "milkshake",
    "matcha", "bubble tea", "cold brew", "sıcak çikolata", "omlet", "pankek", "krep",
    "nachos", "quesadilla", "pad thai", "biftek", "antrikot", "bonfile", "şnitzel",
    "ızgara köfte", "tavuk kanat", "çoban salata", "piyaz", "cacık", "haydari",
)
#: English dish words -> the Turkish query.
DISHES_EN = (
    ("meatballs", "köfte"), ("grilled chicken", "tavuk"), ("sea bass", "levrek"),
    ("tea", "çay"), ("Turkish coffee", "türk kahvesi"), ("lemonade", "limonata"),
    ("ice cream", "dondurma"), ("ramen", "ramen"), ("steak", "biftek"), ("tacos", "taco"),
    ("pasta", "makarna"), ("omelette", "omlet"), ("pancakes", "pankek"),
)
PLACES = (
    "Karaköy", "Galata", "Cihangir", "Arnavutköy", "Ortaköy", "Şişli", "Mecidiyeköy",
    "Levent", "Etiler", "Bakırköy", "Yeşilköy", "Ataşehir", "Maltepe", "Kartal", "Pendik",
    "Suadiye", "Caddebostan", "Göztepe", "Acıbadem", "Kuzguncuk", "Çengelköy", "Balat",
    "Sultanahmet", "Sarıyer", "Tarabya", "Emirgan", "Bostancı", "Erenköy", "Fenerbahçe",
    "Kalamış", "Bornova", "Karşıyaka", "Çeşme", "Alaçatı", "Bodrum", "Antalya", "Bursa",
    "Eskişehir", "İzmir", "Adana", "Trabzon", "Kayseri", "Mersin", "Konya", "Samsun",
    "Çankaya", "Kordon", "Maslak", "Florya", "Beykoz", "Kandilli", "Ayvalık", "Datça",
)
NAMES = (
    "Deniz", "Emre", "Merve", "Ayşe", "Fatma", "Mustafa", "Ali", "Hakan", "Gökhan", "Onur",
    "Barış", "Cem", "Umut", "Yiğit", "Efe", "Arda", "Berk", "Mert", "Serkan", "Tuğba",
    "Büşra", "Esra", "Ceren", "Gizem", "Melis", "Naz", "Defne", "Duru", "Ebru", "Seda",
    "Sinem", "Pınar", "Özge", "Aslı", "Nil", "Batu", "Kutay", "Alp", "Doruk", "Ozan",
    "Sarp", "Tuna", "Yağız", "Bora", "Ece", "Burak", "Eda", "Sude", "Hande", "Kübra",
    "Furkan", "Enes", "Yusuf", "Ömer", "Murat", "Volkan", "Sibel", "Gül", "Nur", "Okan",
)
#: ``(clean, casual, hybrid, the plain question the tool should be asked)``.
QUALITIES = (
    ("bahçesi olan", "bahçeli", "garden'ı olan", "Bahçesi olan bir mekan var mı?"),
    ("şöminesi olan", "şömineli", "fireplace'i olan", "Şöminesi olan bir mekan var mı?"),
    ("tek başına rahatça oturulabilecek", "tek başıma takılmalık", "solo takılmak için rahat",
     "Tek başına rahatça oturulabilecek bir mekan var mı?"),
    ("sigara içilebilen açık alanı olan", "sigara içilen açık alanlı",
     "smoking area'sı olan", "Sigara içilebilen açık alanı olan bir mekan var mı?"),
    ("ailemle gidebileceğim nezih", "aileyle gidilecek nezih", "family friendly",
     "Aileyle gidilebilecek nezih bir mekan var mı?"),
    ("dekoru güzel olan", "dekoru güzel", "decor'u nice olan",
     "Dekoru güzel olan bir mekan var mı?"),
    ("öğrenci dostu", "öğrenciye uygun", "student friendly",
     "Öğrenciler için uygun bir mekan var mı?"),
    ("çocuklarla gidilebilecek", "çocukla rahat gidilen", "kids friendly",
     "Çocuklarla gidilebilecek bir mekan var mı?"),
    ("doğum günü kutlamak için", "doğum günü kutlamalık", "birthday kutlamak için",
     "Doğum günü kutlamak için uygun bir mekan var mı?"),
    ("iş yemeği için uygun", "iş yemeğine uygun", "business lunch için uygun",
     "İş yemeği için uygun bir mekan var mı?"),
    ("kitap okuyup saatlerce oturulabilecek", "kitap okumalık rahat",
     "kitap okumak için cozy", "Kitap okumak için rahat bir mekan var mı?"),
    ("maç izlenebilecek", "maç izlemelik", "maç izlemek için screen'i olan",
     "Maç izlenebilecek bir mekan var mı?"),
    ("kutu oyunu oynanabilen", "kutu oyunu oynamalık", "board game oynanan",
     "Kutu oyunu oynanabilen bir mekan var mı?"),
    ("geç saate kadar oturulabilecek", "gece geç saate kadar takılmalık",
     "late night takılmalık", "Geç saate kadar oturulabilecek bir mekan var mı?"),
    ("nostaljik bir havası olan", "nostaljik havalı", "vintage vibe'ı olan",
     "Nostaljik bir havası olan bir mekan var mı?"),
    ("fotoğraf çekmelik şık", "instagramlık", "instagrammable",
     "Fotoğraf çekmek için güzel bir mekan var mı?"),
    ("ortamı samimi ve sıcak", "samimi ortamlı", "cozy ve samimi vibe'lı",
     "Samimi ve sıcak ortamı olan bir mekan var mı?"),
    ("servisi hızlı olan", "servisi hızlı", "service'i fast olan",
     "Servisi hızlı olan bir mekan var mı?"),
    ("porsiyonları doyurucu olan", "porsiyonu bol", "porsiyonları big olan",
     "Porsiyonları doyurucu olan bir mekan var mı?"),
    ("arkadaşlarla uzun uzun sohbet edilebilecek", "arkadaşlarla sohbet etmelik",
     "friends'le hangout'luk", "Arkadaşlarla uzun uzun sohbet edilebilecek bir mekan var mı?"),
    ("turistik olmayan, yerel halkın gittiği", "turistik olmayan yerel",
     "local'ların gittiği, touristy olmayan",
     "Turistik olmayan, yerel halkın gittiği bir mekan var mı?"),
    ("yıl dönümü kutlamak için şık", "yıl dönümü için şık", "anniversary için fancy",
     "Yıl dönümü kutlamak için şık bir mekan var mı?"),
    ("ferah ve geniş", "ferah geniş", "spacious",
     "Ferah ve geniş bir mekan var mı?"),
    ("çalışanları güler yüzlü olan", "garsonları iyi", "staff'i friendly olan",
     "Çalışanları güler yüzlü olan bir mekan var mı?"),
    ("temiz ve hijyenik", "tertemiz", "hygiene'i iyi olan",
     "Temiz ve hijyenik bir mekan var mı?"),
    ("açık havada oturulabilecek", "dışarıda oturmalık", "outdoor seating'i olan",
     "Açık havada oturulabilecek bir mekan var mı?"),
    ("otoparkı olan", "park yeri olan", "parking'i olan", "Otoparkı olan bir mekan var mı?"),
    ("engelli erişimine uygun", "tekerlekli sandalyeyle girilebilen",
     "wheelchair accessible", "Engelli erişimine uygun bir mekan var mı?"),
    ("otantik bir havası olan", "otantik", "authentic vibe'lı",
     "Otantik bir havası olan bir mekan var mı?"),
    ("hafta sonu keyfi yapılacak", "hafta sonu takılmalık", "weekend keyfi için nice",
     "Hafta sonu keyfi yapılacak bir mekan var mı?"),
)

# --------------------------------------------------------------------------
# Clock times
# --------------------------------------------------------------------------
#: Turkish case endings follow the *spoken* number: 3'te, 6'da, 9'dan, 2'ye.
_LOC = {1: "de", 2: "de", 3: "te", 4: "te", 5: "te", 6: "da", 7: "de", 8: "de", 9: "da",
        10: "da", 11: "de", 12: "de", 13: "te", 14: "te", 15: "te", 16: "da", 17: "de",
        18: "de", 19: "da", 20: "de", 21: "de", 22: "de", 23: "te", 30: "da"}
_DAT = {1: "e", 2: "ye", 3: "e", 4: "e", 5: "e", 6: "ya", 7: "ye", 8: "e", 9: "a", 10: "a",
        11: "e", 12: "ye", 13: "e", 14: "e", 15: "e", 16: "ya", 17: "ye", 18: "e", 19: "a",
        20: "ye", 21: "e", 22: "ye", 23: "e", 30: "a"}
#: ``(word, hours it is said with as (spoken, 24h))``.
_DAYPARTS = (
    ("sabah", tuple((h, h) for h in range(5, 12))),
    ("öğlen", ((12, 12), (1, 13))),
    ("öğleden sonra", tuple((h, h + 12) for h in range(1, 6))),
    ("akşam", tuple((h, h + 12) for h in range(5, 12))),
    ("gece", tuple((h, h + 12) for h in range(9, 12)) + ((12, 0), (1, 1), (2, 2), (3, 3),
                                                            (4, 4))),
)
#: How a time bounds "open": at it, after it, until it. All three ask for
#: ``open_at`` - the tool has one clock argument.
_RELATIONS = ("at", "at", "at", "after", "after", "until")


def _clock_suffix(number: int, relation: str) -> str:
    loc, dat = _LOC.get(number, "de"), _DAT.get(number, "e")
    return {"at": f"'{loc}", "after": f"'{loc}n sonra", "until": f"'{dat} kadar"}[relation]


def clock_phrase(rng) -> tuple[str, str]:
    """``("gece 2'den sonra {open}", "02:00")`` - a spoken time around "open".

    Covers what a fixed list of phrases could not: every hour, half hours,
    the 24-hour clock, and the three ways a time is tied to opening hours.
    The word for "open" is left as ``{open}`` so a hybrid sentence can say
    "open" where a Turkish one says "açık".
    """
    relation = rng.choice(_RELATIONS)
    if rng.random() < 0.08:
        return {"at": "gece yarısı {open}", "after": "gece yarısından sonra {open}",
                "until": "gece yarısına kadar {open}"}[relation], "00:00"
    half = rng.random() < 0.22
    if rng.random() < 0.7:  # a part of the day and a 12-hour number
        word, hours = rng.choice(_DAYPARTS)
        spoken, hour = rng.choice(hours)
        if half:
            phrase = {"at": f"{word} {spoken} buçukta",
                      "after": f"{word} {spoken} buçuktan sonra",
                      "until": f"{word} {spoken} buçuğa kadar"}[relation]
        else:
            phrase = f"{word} {spoken}" + _clock_suffix(spoken, relation)
    else:  # the 24-hour clock
        hour = rng.choice(range(7, 24))
        if half:
            phrase = f"saat {hour}:30" + _clock_suffix(30, relation)
        else:
            phrase = f"saat {hour}" + _clock_suffix(hour, relation)
    return phrase + " {open}", f"{hour:02d}:{30 if half else 0:02d}"


#: Something said before the request that changes nothing about it.
PREAMBLES = {
    "clean": ("Bugün hiçbir şey yemedim.", "İşten yeni çıktım.", "Misafirim geldi.",
              "Evde yemek yok.", "Hava çok güzel bugün.", "Bugün canım dışarıda yemek istiyor."),
    "casual": ("karnım zil çalıyor,", "bugün hiçbir şey yemedim ya,", "işten yeni çıktım,",
               "misafir geldi,", "evde yemek yok,", "hava süper bugün,", "canım sıkıldı,",
               "dışarı çıkasım geldi,", "bütçe sıkışık bu ay,"),
    "hybrid": ("çok hungry'yim,", "işten yeni çıktım, so tired,", "guest'lerim var,",
               "evde food yok,", "weather süper bugün,", "mood'um düşük,",
               "bu ay budget sıkışık,", "long day oldu,"),
}
#: Something added after it.
AFTERTHOUGHTS = {
    "clean": ("Acelem var.", "Öneriye ihtiyacım var.", "Çok uzak olmasın lütfen."),
    "casual": ("acelem var", "öneri lazım", "bi el at", "çok bekletme"),
    "hybrid": ("acelem var asap", "any suggestion?", "help lazım", "quick olsun"),
}
DISTANCES = (
    ("300 metre", 300), ("500 metre", 500), ("750 metre", 750), ("1 km", 1000),
    ("1 kilometre", 1000), ("1,5 km", 1500), ("2 km", 2000), ("3 kilometre", 3000),
)
WALKS = (3, 5, 7, 8, 10, 12, 15, 20, 25, 30)
PRICES = (40, 50, 60, 75, 80, 90, 100, 120, 125, 180, 200, 220, 250, 300, 350, 400, 500, 750)

SCREENS = {
    "settings": (("Ayarları aç", "Ayarlar sayfasına git", "Uygulama ayarlarını göster"),
                 ("ayarları açsana", "ayarlara gircem", "ayarlar nerde ya, aç"),
                 ("settings'i aç", "settings sayfasına git", "app settings'e bakıcam")),
    "notifications": (("Bildirimlerimi göster", "Bildirimleri aç", "Bildirimlerime bakmak istiyorum"),
                      ("bildirimlere bakayım", "bildirimleri açsana", "bildirim gelmiş mi bi bak"),
                      ("notification'ları aç", "notifications sayfasına git",
                       "notif'lerime bakayım")),
    "qr_scanner": (("QR kod okuyucuyu aç", "Karekod okutmak istiyorum", "Menünün QR kodunu tarayacağım"),
                   ("qr okutucam aç", "karekodu okutcam", "masadaki qr'ı okutayım"),
                   ("qr scanner'ı aç", "qr code scan edicem", "menüyü scan edeyim")),
    "profile": (("Profilimi aç", "Profil sayfama git", "Hesabımı göster"),
                ("profilime gir", "profilimi açsana", "hesabıma bakayım"),
                ("profile sayfamı aç", "profile page'e gir", "account'uma bakayım")),
    "chats": (("Mesajlarımı aç", "Sohbetlerimi göster", "Sohbetler sayfasına git"),
              ("mesajlara bakayım", "sohbetleri aç", "mesaj kutuma gir"),
              ("chat'leri aç", "dm'lerime bakayım", "messages'a gir")),
    "feed": (("Akışı aç", "Ana sayfaya dön", "Gönderi akışını göster"),
             ("ana sayfaya dön", "akışa bakayım", "anasayfayı aç"),
             ("feed'i aç", "home'a dön", "feed'e bakayım")),
    "map": (("Haritayı aç", "Harita görünümüne geç", "Mekanları haritada göster"),
            ("haritayı açsana", "haritaya bakayım", "haritadan bakalım"),
            ("map'i aç", "map'ten bakalım", "map üzerinden gösterir misin")),
    "add_friends": (("Arkadaş ekleme sayfasını aç", "Arkadaş eklemek istiyorum",
                     "Yeni arkadaş ekleme ekranına git"),
                    ("arkadaş eklicem, sayfayı aç", "arkadaş ekleme yerini aç",
                     "birini eklicem nerden yapıyoduk"),
                    ("add friend sayfasını aç", "friend eklemek istiyorum, aç",
                     "add friends ekranına git")),
    "friends": (("Arkadaşlar sayfasını aç", "Arkadaşlar ekranına git"),
                ("arkadaşlar sayfasına gir", "arkadaşlar ekranını aç"),
                ("friends sayfasını aç", "friends ekranına git")),
    "saved": (("Kaydettiklerimi aç", "Kaydedilenler sayfasına git", "Kaydedilenlerimi göster"),
              ("kaydettiklerime bakayım", "kaydedilenleri aç", "kaydettiğim şeyleri göster"),
              ("saved listemi aç", "saved'e git", "save ettiklerimi aç")),
}
SCREENS_EN = {
    "settings": ("Open settings", "Take me to the app settings"),
    "notifications": ("Show my notifications", "Open notifications"),
    "qr_scanner": ("Open the QR scanner", "I want to scan the menu QR code"),
    "profile": ("Go to my profile", "Open my profile page"),
    "chats": ("Open my chats", "Show my messages"),
    "feed": ("Go to the feed", "Back to the home feed"),
    "map": ("Open the map", "Show the map view"),
    "saved": ("Show my saved items", "Open my saved places"),
}

#: ``(what the user dictates, as written in the message)`` for "X'e ... yaz".
MESSAGES = (
    "10 dakikaya oradayım", "yoldayım geliyorum", "biraz gecikeceğim", "akşam 8'de buluşalım",
    "yarın öğlen müsait misin", "rezervasyonu ben yaptım", "kapının önündeyim",
    "bu akşam dışarı çıkıyor muyuz", "masayı tuttum, acele et", "bugün gelemiyorum",
    "hesabı ben ödedim", "çıkınca ara beni", "nerede buluşuyoruz", "yarım saate çıkıyorum",
    "bu mekanı denemeliyiz", "hafta sonu planın var mı", "toplantı uzadı, geç kalacağım",
    "seni bekliyoruz", "bir şey lazım mı gelirken", "otoparkta buluşalım",
    "doğum günün kutlu olsun", "çok güzeldi, teşekkürler", "yarın ben ısmarlıyorum",
    "iş çıktı, başka zaman yapalım", "metrodan indim, 5 dakikaya oradayım",
)
#: ``(reported form, the message it stands for)`` for "X'e ... söyle".
REPORTED = (
    ("geç kalacağımı", "Geç kalacağım"), ("yolda olduğumu", "Yoldayım"),
    ("bu akşam gelemeyeceğimi", "Bu akşam gelemeyeceğim"),
    ("toplantının iptal olduğunu", "Toplantı iptal oldu"),
    ("masayı ayırttığımı", "Masayı ayırttım"), ("onu beklediğimizi", "Seni bekliyoruz"),
    ("yarın müsait olmadığımı", "Yarın müsait değilim"),
    ("on dakikaya orada olacağımı", "On dakikaya oradayım"),
    ("hesabı ödediğimi", "Hesabı ödedim"), ("eve vardığımı", "Eve vardım"),
)
MESSAGES_HYBRID = (
    "trafik bad, 15 dk gecikiyorum", "iş çıktı, bu akşam skip", "ok ben yoldayım",
    "location atar mısın", "dinner bende bu akşam", "5 dk'ya oradayım wait et",
    "reservation tamam, see you", "bugün busy'im, yarın yapalım", "nerdesin, call me",
    "weekend için plan var mı", "check ettim, yer var", "happy birthday kanka",
)
VENUE_FIRST = ("Liman", "Köşe", "Sahil", "Çınaraltı", "Tarihi", "Meşhur", "Usta", "Dede",
               "Nazar", "Lale", "Zeytin", "Fener", "Yalı", "Bahçe", "Konak", "Mavi", "Asma",
               "Dut", "Nar", "Çardak", "İskele", "Kule", "Avlu", "Han", "Değirmen")
VENUE_LAST = ("Balık", "Kebap", "Ocakbaşı", "Meyhanesi", "Kahve", "Lokantası", "Mutfak",
              "Sofrası", "Evi", "Restoran", "Köftecisi", "Fırını", "Büfe", "Bistro")

OOS = (
    ("Bana bir taksi çağır.", "taksi çağırsana bi", "bana bi taxi call et"),
    ("Bugünkü burç yorumum ne?", "burcumda ne diyo bugün", "horoscope'um ne diyo"),
    ("Bu cümleyi İngilizceye çevir.", "şunu ingilizceye çevirsene", "şunu translate et"),
    ("Bana bir fıkra anlat.", "bi fıkra patlat", "bi joke anlat"),
    ("Bu mekanda rezervasyon yap.", "bana masa ayırt", "reservation yap bana"),
    ("Yemek siparişi ver.", "bana yemek söyle", "order ver bana"),
    ("Bugün trafik nasıl?", "trafik var mı şu an", "traffic nasıl şu an"),
    ("E-posta gönder.", "mail atsana birine", "bi mail send et"),
    ("Fotoğraf çek.", "foto çeksene", "bi photo çek"),
    ("Son haberler ne?", "gündemde ne var", "news'te ne var bugün"),
    ("12 çarpı 34 kaç eder?", "12 çarpı 34 kaç ya", "12 kere 34 calculate et"),
    ("Bana bir film öner.", "film önersene", "bi movie recommend et"),
    ("Telefonumun şarjı ne kadar?", "şarjım kaç ya", "battery kaç percent"),
    ("Not al: süt alınacak.", "şunu not et, süt alıcam", "bi note al, süt alınacak"),
    ("Annemi ara.", "annemi arasana", "annemi call et"),
    ("Kargom nerede?", "kargom nerde kaldı", "kargomu track et"),
)

POOLS = {
    "kinds": [k for pair in KINDS for k in pair], "dishes": DISHES, "places": PLACES,
    "names": NAMES, "venue names": VENUE_FIRST + VENUE_LAST, "messages": MESSAGES,
    "hybrid messages": MESSAGES_HYBRID,
    "qualities": [text for row in QUALITIES for text in row],
}


def check_pools() -> list[str]:
    """Pool entries that mention a held-out entity. Must be empty."""
    return [f"{pool}: {value!r} hits {held_out_hit(value)!r}"
            for pool, values in POOLS.items() for value in values if held_out_hit(value)]
