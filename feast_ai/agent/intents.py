"""feast intents: what the router can tell apart, and what each one unlocks.

An intent is three things: example utterances the router matches against, a
*route* (how the message is served) and a *tool group* (what the SLM is
shown when it does get involved).

Routes
------
``slm``      the SLM answers with only this intent's tools in its prompt;
``direct``   a deterministic handler answers with no SLM at all - a tool
             call or a templated sentence - when the router is confident;
             otherwise it degrades to ``slm`` with the same tool group;
``inherit``  a follow-up ("evet", "daha ucuzu var mi") carries no intent of
             its own, so it reuses the previous turn's tool group.  Without
             this, confirming a message - the one protocol that matters
             most - would be routed as small talk.

Tool groups overlap on purpose: intents the router can confuse share tools,
so a near-miss (restaurant vs dish search) still lands on a prompt that can
serve it.  Route to *groups*, not to single tools.  Three overlaps were added
after the first router benchmark exposed the pairs it confuses - message vs
shared items (a name plus "gonder"), posts vs venue search ("populer"),
friends vs suggestions - so the suite is no longer blind for those pairs.

The exemplars are proper Turkish with diacritics - an exception to the ASCII
convention elsewhere, because the encoder's cased WordPiece vocabulary is
Turkish and "göster" is one token where "goster" is three.  A few are typed
without diacritics, as users do, and a few are English.  They are written to
be different from the benchmark suite in :mod:`feast_ai.agent.suite`, so
routing accuracy is never measured on its own training data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    id: str
    route: str  # "slm" | "direct" | "inherit"
    tools: tuple[str, ...]
    summary: str
    exemplars: tuple[str, ...]


INTENTS: tuple[Intent, ...] = (
    Intent(
        "restaurant_search", "slm",
        ("search_restaurants", "search_menu_items", "set_active_location"),
        "find venues near the active location",
        (
            "Yakınlarda güzel bir kahvaltı yeri var mı?",
            "Şu an açık olan restoranları göster",
            "Bu akşam yemek yiyebileceğim bir yer öner",
            "Yürüyerek 10 dakikalık mesafede kafe bul",
            "Etrafta popüler kebapçılar hangileri",
            "En yakın pizzacı nerede",
            "Gece yarısı açık bir yer lazım",
            "Saat 9'da açık olan kahvaltıcılar",
            "yakinda vegan restoran var mi",
            "Kadıköy'de iyi bir meyhane öner",
            "Find a good breakfast place nearby",
            "Which cafés are open right now?",
        ),
    ),
    Intent(
        "menu_search", "slm",
        ("search_menu_items", "search_restaurants", "set_active_location"),
        "find dishes, drinks and prices",
        (
            "En ucuz lahmacun hangi restoranda?",
            "200 liranın altında burger bulabilir miyim",
            "Yakınlarda künefe kaç lira?",
            "Menüsünde mantı olan yerler",
            "Filtre kahve fiyatları ne kadar",
            "Uygun fiyatlı bir tatlı arıyorum",
            "50 TL altı çorba var mı",
            "Hangi restoranda glutensiz makarna var",
            "en pahalı biftek nerede",
            "Bir porsiyon iskender buralarda kaç para",
            "Where can I get the cheapest coffee?",
            "Show me desserts under 100 lira",
        ),
    ),
    Intent(
        "vibe_search", "slm",
        ("ask_about_posts", "search_restaurants", "search_posts"),
        "venues with a subjective quality no tag captures",
        (
            "Laptopla rahat çalışabileceğim bir kafe var mı?",
            "Sessiz, kitap okunabilecek bir yer öner",
            "Manzaralı bir yerde kahvaltı yapmak istiyorum",
            "Köpeğimle gidebileceğim mekanlar",
            "Çocuklu aileler için uygun restoran",
            "Romantik bir akşam yemeği için mekan",
            "Prizi bol olan kafeler hangileri",
            "Kalabalık olmayan, sakin bir yer arıyorum",
            "Arkadaşlarla uzun oturulacak samimi bir mekan",
            "Bahçesi olan bir kafe var mı",
            "A quiet café where I can work on my laptop",
            "Pet-friendly places around here",
        ),
    ),
    Intent(
        "details", "slm",
        ("get_restaurant_details", "get_menu_item_details", "get_directions"),
        "follow up on a venue or dish from an earlier result",
        (
            "İlk restoranın çalışma saatleri ne?",
            "Bu mekanın menüsünü göster",
            "Oranın adresi neydi?",
            "İkinci sıradaki yere nasıl giderim?",
            "Oraya yürüyerek yol tarifi ver",
            "Bu yemeğin içinde ne var?",
            "Az önceki tatlının fiyatı neydi",
            "Bu kafe kaçta kapanıyor",
            "Seçtiğim restoranın detaylarını aç",
            "Oraya arabayla gitmek istiyorum",
            "Show me the menu of the first one",
            "How do I get there?",
        ),
    ),
    Intent(
        "location_change", "slm",
        ("set_active_location",),
        "move the search area somewhere else",
        (
            "Konumu Beşiktaş olarak değiştir",
            "Ankara'daki yerlere bakmak istiyorum",
            "Arama bölgesini Nişantaşı yap",
            "İzmir Alsancak'a geç",
            "Cihangir civarına taşı",
            "Şehri Eskişehir yap",
            "Karaköy'e göre ara",
            "konumu bakirkoy yap",
            "Başka bir semte bakalım, Bebek olsun",
            "Switch the location to Taksim",
            "Search around Bebek instead",
        ),
    ),
    Intent(
        "location_reset", "direct",
        ("set_active_location",),
        "go back to the GPS location",
        (
            "Beni bulunduğum yere göre ara",
            "Bulunduğum yeri kullan",
            "GPS konumuma göre ara",
            "Şu an olduğum yere dön",
            "Gerçek konumumu kullan",
            "Konumu sıfırla",
            "Mevcut konumuma göre göster",
            "konumumu kullan",
            "Use my current location",
            "Go back to my GPS location",
        ),
    ),
    Intent(
        "posts", "slm",
        ("search_posts", "ask_about_posts", "search_restaurants"),
        "what people posted nearby",
        (
            "Yakındaki gönderilerde pizza geçenler",
            "Burger hakkında paylaşımlar göster",
            "İnsanlar Moda'daki kafeler hakkında ne paylaşmış",
            "Etrafta kahvaltı fotoğrafı paylaşan var mı",
            "Son paylaşılan tatlı gönderileri",
            "Bu restoran hakkında yapılan postlar",
            "Gönderilerde en çok övülen kahveci",
            "Kullanıcılar sushi hakkında ne yazmış",
            "yakindaki postlarda baklava var mi",
            "Show posts about ramen nearby",
            "What are people posting about brunch?",
        ),
    ),
    Intent(
        "saved_open", "direct",
        ("open_screen", "search_saved"),
        "open the saved items screen",
        (
            "Kaydettiklerimi aç",
            "Kayıtlı yerlerimi göster",
            "Favorilerim",
            "Kaydedilenler sayfasına git",
            "Kaydettiğim şeyler nerede",
            "Kaydedilen listemi aç",
            "Favori mekanlarımı göster",
            "kaydettiklerim",
            "Show my saved items",
            "Open my favorites",
        ),
    ),
    Intent(
        "saved_search", "slm",
        ("search_saved",),
        "filter or search the user's saved items",
        (
            "Kayıtlı mekanlarımdan gece açık olanlar",
            "Kayıtlı tatlıcılardan en yakını hangisi",
            "Kaydettiğim yemeklerden 300 liranın altındakiler",
            "Kaydettiğim gönderilerde kahve geçenler",
            "Favorilerimdeki kahvaltıcılar",
            "Kaydettiğim yerlerden Kadıköy'de olanlar",
            "En son kaydettiğim mekan hangisiydi",
            "Kayıtlı yemeklerimi fiyata göre sırala",
            "kaydettigim kafelerden yuruyerek 5 dakika olanlar",
            "Which of my saved places are open now?",
            "My saved desserts sorted by price",
        ),
    ),
    Intent(
        "shared", "slm",
        ("search_shared_items", "list_friends"),
        "venues and dishes sent in chats",
        (
            "Bana gönderilen restoranları göster",
            "Ali'nin bana attığı mekan neydi?",
            "Benim arkadaşlarıma gönderdiğim yemekler",
            "Sohbetlerde paylaşılan kafeler",
            "Zeynep'in geçen hafta yolladığı tatlıcı",
            "Mesajlarda bana önerilen yerler",
            "Kime hangi restoranı göndermiştim",
            "Arkadaşlarımın bana yolladığı gönderiler",
            "bana atilan restoran linkleri",
            "Restaurants my friends sent me",
            "What did I share with Can?",
        ),
    ),
    Intent(
        "memory", "slm",
        ("search_chat_memory",),
        "earlier conversations with the assistant",
        (
            "Daha önce konuştuğumuz balıkçı neydi?",
            "Geçen sefer önerdiğin kafeyi hatırlıyor musun",
            "Seninle dün ne konuşmuştuk",
            "Bana daha önce önerdiğin tatlıcının adı neydi",
            "Geçen hafta sorduğum soruyu hatırla",
            "Önceki sohbetimizde bahsettiğin restoran",
            "Daha önce sana söylediğim alerjimi hatırlıyor musun",
            "Konuşmalarımızda geçen vegan mekan",
            "gecen konustugumuz meyhane hangisiydi",
            "What was the place you recommended last time?",
            "Do you remember what we talked about yesterday?",
        ),
    ),
    Intent(
        "friends", "direct",
        ("list_friends", "get_suggested_users"),
        "list the user's friends",
        (
            "Arkadaşlarım kimler",
            "Arkadaş listemi göster",
            "Arkadaşlarımı listele",
            "feast'teki arkadaşlarım",
            "Arkadaşlarımı görmek istiyorum",
            "Ekli arkadaşlarım kimler",
            "arkadas listem",
            "Show my friends",
            "Who are my friends?",
        ),
    ),
    Intent(
        "suggested_users", "direct",
        ("get_suggested_users", "open_screen"),
        "people the user may know",
        (
            "Kimi arkadaş olarak ekleyebilirim",
            "Tanıyor olabileceğim kişiler",
            "Bana arkadaş öner",
            "Yeni arkadaş önerileri göster",
            "Ortak arkadaşlarım olan kişiler",
            "Kimleri ekleyebilirim",
            "Önerilen kullanıcılar",
            "arkadas onerileri",
            "People I may know",
            "Suggest some friends to add",
        ),
    ),
    Intent(
        "message", "slm",
        ("send_message_to_friends", "list_friends", "search_shared_items"),
        "send a message to friends (needs confirmation)",
        (
            "Ece'ye yarın öğlen buluşalım diye mesaj at",
            "Ayşe'ye bu restoranı gönder",
            "Arkadaşlarıma yarın kahvaltı yapalım mı diye yaz",
            "Mehmet ve Elif'e geliyorum de",
            "Can'a bu mekanı öner",
            "Selin'e 10 dakika gecikeceğimi söyle",
            "Burak'a mesaj gönder: rezervasyonu yaptım",
            "zeynepe mesaj at",
            "Text Ali that I'm on my way",
            "Send this place to Deniz",
        ),
    ),
    Intent(
        "navigate", "direct",
        ("open_screen",),
        "open a screen of the app",
        (
            "Ayarları aç",
            "Profilime git",
            "Bildirimlerimi göster",
            "Mesajlar ekranını aç",
            "Haritayı aç",
            "Ana akışa dön",
            "QR kod okuyucuyu aç",
            "Sohbetlerimi aç",
            "Arkadaş ekleme sayfasına git",
            "Hesap ayarlarım",
            "Open settings",
            "Go to my profile",
        ),
    ),
    Intent(
        "context_info", "direct",
        (),
        "time, date or active location - already in CONTEXT",
        (
            "Saat kaç?",
            "Bugün günlerden ne?",
            "Bugünün tarihi ne",
            "Şu an neredeyim?",
            "Aktif konum neresi",
            "Hangi bölgede arama yapıyorsun",
            "Konumum ne görünüyor",
            "Saati söyler misin",
            "saat kac",
            "What time is it?",
            "Where am I right now?",
        ),
    ),
    Intent(
        "chitchat", "slm",
        (),
        "greetings, thanks, questions about the assistant",
        (
            "Merhaba",
            "Selam, nasılsın?",
            "Teşekkür ederim",
            "Sen kimsin?",
            "Neler yapabilirsin?",
            "Günaydın",
            "Harikasın, sağ ol",
            "İyi geceler",
            "Bana yardım eder misin",
            "Hi there",
            "Thanks a lot",
            "What can you do?",
        ),
    ),
    Intent(
        "out_of_scope", "direct",
        (),
        "outside what the feast assistant does",
        (
            "Yarın hava nasıl olacak?",
            "Sabah 7'ye alarm kur",
            "Bir şarkı çal",
            "Dolar kaç lira?",
            "Bana bir fıkra anlat",
            "Taksi çağır",
            "Maç kaç kaç bitti",
            "Işıkları kapat",
            "Bana bir şiir yaz",
            "Will it rain tomorrow?",
            "Set a timer for 10 minutes",
        ),
    ),
    Intent(
        "followup", "inherit",
        (),
        "confirmation or continuation of the previous turn",
        (
            "Evet",
            "Evet, gönder",
            "Tamam, onaylıyorum",
            "Hayır, vazgeç",
            "Olur",
            "Aynen öyle",
            "İptal et",
            "Daha fazla göster",
            "Başka var mı?",
            "Daha ucuzu var mı",
            "Bir sonrakine geç",
            "evet yolla",
            "Yes, send it",
            "Show more",
        ),
    ),
)

BY_ID: dict[str, Intent] = {i.id: i for i in INTENTS}


def exemplar_bank() -> list[tuple[str, str]]:
    """``(intent id, utterance)`` pairs, the router's whole knowledge."""
    return [(intent.id, text) for intent in INTENTS for text in intent.exemplars]
