"""The blind suite: 97 turns nobody tuned on, in words that are not the generator's.

Cards first, sentences second.  :mod:`mustachio.blind_cards` fixes what each
user wants and which call answers it, with dishes, districts and people that
are in no training pool and no other suite.  The sentences below were then
written by a teacher model (Gemma 4 12B) from each card's English
description alone - it saw no template, no tool name and no training
sentence - and each one was read against its card:

* 89 of the 97 are the teacher's first draft, kept as written (typos
  included);
* 3 are its second draft, after the first said something else than the card;
* 5 were written by hand: asked twice for "tell the assistant to message a
  friend", the teacher wrote the message to the friend itself;
* six cards had their expectation loosened to what the kept sentence
  actually says (a draft that drops "at 23:30" cannot be held to it).

It was scored once the pipeline was final; nothing was changed afterwards.
The teacher has habits of its own ("kanka ... ya", "check edebilir misin"),
so this is one more author, not the population of feast's users - real
messages remain the test that matters.
"""

from __future__ import annotations

from feast_ai.agent.suite import Case, Turn

from .blind_cards import CARDS

#: ``(card id, turn index) -> (the sentence, who wrote it)``.
TEXTS: dict[tuple[str, int], tuple[str, str]] = {
    ('b-venue-1', 0): ("kanka yakında bi lokmacı var mı bi bakar mısın ya", 'teacher'),
    ('b-venue-2', 0): ("Şu an açık olan ve yakındaki güzel bir Lübnan restoranı check edebilir miyiz, mood tamına uygun bir yer arıyorum.", 'teacher'),
    ('b-venue-3', 0): ("yakinlarda 15 dakikalik mesafede meze evi var mi?", 'teacher'),
    ('b-venue-4', 0): ("Buralarda en popüler Gürcü restoranı hangisidir?", 'teacher'),
    ('b-venue-5', 0): ("Gece acıktım ve canım çok güzel bir şeyler yemek istiyor, şu an açık olan bir Tayland restoranı bulabilir misin acaba? Saat 23:30 olduğu için seçenekler kısıtlı olabilir.", 'teacher'),
    ('b-venue-6', 0): ("en yakın şarküteri", 'teacher'),
    ('b-venue-7', 0): ("Yakınlarda tapas yapabileceğimiz, max 2 km mesafede güzel bir mekan check edebilir misin?", 'teacher'),
    ('b-venue-8', 0): ("abi 3'ten sonra açık olan bi yer var mı ya bi bakayım", 'teacher'),
    ('b-dish-1', 0): ("kanka yakınlarda mumbar nerede var ya bi bakabilir misin", 'teacher'),
    ('b-dish-2', 0): ("Ali nazik yiyebilirim ama budget olarak 320 TL'yi geçmesin, uygun bir location var mı check edebilir misin?", 'teacher'),
    ('b-dish-3', 0): ("en ucuz etli ekmek nerede?", 'teacher'),
    ('b-dish-4', 0): ("Yakındaki restoranlarda çılbır fiyatları ne kadar?", 'teacher'),
    ('b-dish-5', 0): ("Canım çok tatlı bir şeyler çekiyor, özellikle de höşmerim yemek istiyorum; şu an açık olan ve bu tatlıyı bulabileceğim bir yer önerebilir misin?", 'teacher'),
    ('b-dish-6', 0): ("En pahalı tas kebabı nerede?", 'teacher'),
    ('b-dish-7', 0): ("Şu an çok acil bir affogato ihtiyacım var, nearby bir yer var mı 10 dakika yürüyüş mesafesinde check edebilir misin?", 'teacher'),
    ('b-dish-8', 0): ("kanka 90 liraya kısır olan bi yer var mı ya?", 'teacher'),
    ('b-vibe-1', 0): ("abi karaoke yapabileceğim güzel bir yer var mı bi baksana ya", 'teacher'),
    ('b-vibe-2', 0): ("Sunset izleyebileceğim güzel bir location var mı nearby, tam bir mood arıyorum.", 'teacher'),
    ('b-vibe-3', 0): ("Bugün biraz kafa dinlemeye ve sakin bir ortamda vakit geçirmeye ihtiyacım var, telefonumun şarjı da bitmek üzere; acaba çevrede kalabalık olmayan, huzurlu ve priz bulabileceğim bir yer önerebilir misiniz?", 'teacher'),
    ('b-vibe-4', 0): ("Bebek arabasıyla rahatça giriş yapabileceğim, geniş ve konforlu bir restoran önerisi verebilir misiniz?", 'teacher'),
    ('b-details-1', 0): ("ya kanka ikinci sıradaki yer kaça kadar açık ya?", 'teacher'),
    ('b-details-1', 1): ("kanka ya bi de arabayla nasıl gidilir onu söyler misin ya bi bakayım", 'teacher'),
    ('b-details-2', 0): ("Sarnıç Şarküteri'nin menüsünü check edebilir misin, bir bakmak istiyorum.", 'teacher'),
    ('b-details-3', 0): ("ilk listedeki yere toplu tasimayla nasil giderim?", 'teacher'),
    ('b-details-4', 0): ("İkinci sırada yer alan tatlının içeriğinde neler var?", 'teacher'),
    ('b-details-5', 0): ("İlk tatlıda kuruyemiş var mı?", 'teacher'),
    ('b-loc-1', 0): ("kanka arama alanını çukurcuma yapalım ya", 'teacher'),
    ('b-loc-2', 0): ("Bu hafta sonu Urla tarafına gideceğiz, orada güzel mekanları bir check edebilir misin?", 'teacher'),
    ('b-loc-3', 0): ("bu aksam tesvikiye civarindayim ona gore bakabilir misin", 'teacher'),
    ('b-loc-4', 0): ("ya kanka bi konumumdaki yerlere dönelim ya", 'teacher'),
    ('b-loc-5', 0): ("Kaş ara", 'teacher'),
    ('b-loc-5', 1): ("kanka şimdi bi meze evi bulur musun orada ya", 'teacher'),
    ('b-posts-1', 0): ("kanka bi mumbar yiyelim diyorum millet ne demiş nerde iyi yer var ya", 'teacher'),
    ('b-posts-2', 0): ("Frappe üzerine paylaşılan herhangi bir post var mı, bir check edebilir misin?", 'teacher'),
    ('b-saved-1', 0): ("kanka bi saved listimi açar mısın ya", 'teacher'),
    ('b-saved-2', 0): ("Kaydettiğim dishler arasında en expensive olanı hangisi, bir check edebilir misin?", 'teacher'),
    ('b-saved-3', 0): ("Fethiye'de kaydettiğim mekanları görebilir miyim?", 'teacher'),
    ('b-saved-4', 0): ("kaydettiklerim arasinda asure gecerenleri bulur musun", 'teacher'),
    ('b-saved-5', 0): ("Son kaydedilenler", 'teacher'),
    ('b-shared-1', 0): ("kanka hatice bi ara bi restoran atmıştı ya onu bulabilir misin bi bakar mısın neydi o", 'teacher, second draft'),
    ('b-shared-2', 0): ("Koray ile paylaştığım o dish'leri bir listeleyebilir misin?", 'teacher, second draft'),
    ('b-shared-3', 0): ("Arkadaşlarımın sohbetlerde gönderdiği tüm mekanları listeleyebilir misiniz?", 'teacher'),
    ('b-memory-1', 0): ("abi az önce bi şarküteri sormuştum ya o hangisiydi ya bi hatırlatsana", 'teacher'),
    ('b-memory-2', 0): ("Selam, geçen sefer kruvasan için hangi location'ı check etmiştin, tekrar bakmak istiyorum.", 'teacher'),
    ('b-friends-1', 0): ("kanka benim arkadaşları nereden görebiliyorum ya bi bakar mısın", 'teacher'),
    ('b-friends-2', 0): ("yunus arkadaşlarim arasında mi", 'teacher'),
    ('b-suggest-1', 0): ("Yeni arkadaş eklemek için herhangi bir suggestion var mı, acaba kimleri add edebilirim?", 'teacher'),
    ('b-nav-1', 0): ("ay kanka şu ayarlar nereye ya bi bakıp söyler misin", 'teacher'),
    ('b-nav-2', 0): ("Bildirimleri check edebilir miyim, son gelenler neler?", 'teacher'),
    ('b-nav-3', 0): ("qr kodunu okutabilir misiniz?", 'teacher'),
    ('b-nav-4', 0): ("Profilim", 'teacher'),
    ('b-nav-5', 0): ("kanka haritaya bi baksak ya nerde bunlar", 'teacher'),
    ('b-nav-6', 0): ("Mesajları bir check edebilir miyim?", 'teacher'),
    ('b-nav-7', 0): ("Yeni bir arkadaş eklemek için ilgili ekranı açabilir misiniz?", 'teacher'),
    ('b-nav-8', 0): ("ya kanka ana sayfaya dönelim bi ya", 'teacher'),
    ('b-msg-1', 0): ("zehraya haber versene 20 dkya ordayım diye", 'hand'),
    ('b-msg-1', 1): ("olur kanka yolla ya", 'teacher'),
    ('b-msg-2', 0): ("Hüseyin'e mesaj atar mısın, dinner iptal oldu diye", 'hand'),
    ('b-msg-2', 1): ("Aslında vazgeçtim, şu an başka bir plan yapalım, iptal edebiliriz.", 'teacher'),
    ('b-msg-3', 0): ("emine ve tarik cumartesi uygun mu diye sorar misin", 'teacher'),
    ('b-msg-3', 1): ("tamam gonderebilirsin", 'teacher'),
    ('b-msg-4', 0): ("bilge'ye ilk sıradakini \"yarın buraya gidelim mi\" notuyla atar mısın kanka", 'teacher, second draft'),
    ('b-msg-4', 1): ("tamam kanka gönder gitsin ya", 'teacher'),
    ('b-msg-5', 0): ("Gamze'ye bir mesaj gönderir misin: doğum günün kutlu olsun", 'hand'),
    ('b-msg-5', 1): ("Lütfen metni 'iyi ki doğdun canım' şeklinde değiştirir misiniz?", 'teacher'),
    ('b-msg-5', 2): ("Evet, bu yeni versiyonun gönderilmesini onaylıyorum.", 'teacher'),
    ('b-ctx-1', 0): ("Şu an saat kaç, bir check edebilir misin?", 'teacher'),
    ('b-ctx-2', 0): ("bugün günlerden ne kanka ya", 'teacher'),
    ('b-ctx-3', 0): ("hangi bolgede arama yapyorsun?", 'teacher'),
    ('b-talk-1', 0): ("selam kanka naber ya", 'teacher'),
    ('b-talk-2', 0): ("Çok teşekkürler, gerçekten çok helpful oldu, tam aradığım mood'u yakaladık.", 'teacher'),
    ('b-talk-3', 0): ("neler yapabiliyorsun?", 'teacher'),
    ('b-talk-4', 0): ("Güle güle", 'teacher'),
    ('b-oos-1', 0): ("abi hafta sonu için bi otel rezervasyonu yapar mısın ya", 'teacher'),
    ('b-oos-2', 0): ("Yakınlarda gece nöbeti olan bir eczane var mı öğrenebilir miyim?", 'teacher'),
    ('b-oos-3', 0): ("Selam, bugünkü akaryakıt fiyatlarını bir check edebilir misin?", 'teacher'),
    ('b-oos-4', 0): ("kredi karti borcumu odeyebilir misin?", 'teacher'),
    ('b-follow-1', 0): ("kanka bi perde pilavı gömücem ama yakında güzel bi yer var mı ya", 'teacher'),
    ('b-follow-1', 1): ("kanka ya kişi başı max 400 geçmesin bi baksan olur mu?", 'teacher'),
    ('b-follow-2', 0): ("Yakınlarda güzel bir bowl restoranı var mı, check edebilir misin?", 'teacher'),
    ('b-follow-2', 1): ("Şu an açık olan yerleri check eder misin, sadece açık olanları görmek istiyorum.", 'teacher'),
    ('b-follow-3', 0): ("burada mocha ne kadar?", 'teacher'),
    ('b-follow-3', 1): ("en ucuz hangisi?", 'teacher'),
    ('b-mixed-1', 0): ("ya kanka aramayı yeldeğirmeni olarak değiştirir misin bi", 'teacher'),
    ('b-mixed-1', 1): ("kanka buralarda bi sandviççi var mı ya nerede güzel yer var bi bak bi", 'teacher'),
    ('b-mixed-1', 2): ("utkuya da yaz bi, akşam 7de buluşalım diye", 'hand'),
    ('b-mixed-1', 3): ("olur kanka gönder gitsin ya", 'teacher'),
    ('b-mixed-1', 4): ("abi saat kaç şu an ya", 'teacher'),
    ('b-mixed-2', 0): ("Nearby lokasyonlarda irmik helvası yapan güzel bir yer var mı check edebilir misin?", 'teacher'),
    ('b-mixed-2', 1): ("Bildirimleri check edebilir miyim, favori mekanlarımı kaçırmak istemiyorum.", 'teacher'),
    ('b-mixed-2', 2): ("Meryem bana hangi locationları göndermişti, bir check edebilir misin?", 'teacher'),
    ('b-mixed-2', 3): ("Harika önerilerin için çok teşekkürler, tam aradığım mood'u yakaladın.", 'teacher'),
    ('b-mixed-3', 0): ("selam, naber?", 'teacher'),
    ('b-mixed-3', 1): ("en ucuz kefir nerede var", 'teacher'),
    ('b-mixed-3', 2): ("damlaya gecikecegimi haber ver", 'hand'),
    ('b-mixed-3', 3): ("iptal et, gonderme", 'teacher'),
    ('b-mixed-3', 4): ("haritayi ac", 'teacher'),
}


def _build() -> tuple[Case, ...]:
    cases = []
    for card in CARDS:
        turns = tuple(
            Turn(TEXTS[(card["id"], index)][0], turn["intent"], turn["tool"], dict(turn["args"]),
                 tuple(turn["forbid"]))
            for index, turn in enumerate(card["turns"]))
        cases.append(Case(card["id"], turns, history=tuple(card["history"])))
    return tuple(cases)


BLIND: tuple[Case, ...] = _build()
