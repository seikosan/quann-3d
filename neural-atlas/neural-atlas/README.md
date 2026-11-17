# Neural Atlas

Kendi sinir ağı mimarin için yerelde çalışan 3B çizim arayüzü. Döndürme, yakınlaşma, katman seçimi, JSON içe aktarma/düzenleme, PNG dışa aktarma ve şematik akış oynatma içerir. Referans: https://bbycroft.net/llm — uygulama kodu bağımsız yazılmıştır.

## Çalıştır

Mac üzerinde `Baslat.command` dosyasını çalıştır. Python 3 kuruluysa hazır sürüm tarayıcıda açılır; paket kurulumu gerekmez.

Bu klasörde:

```sh
npm install
npm run dev
```

Terminaldeki yerel adresi aç. Hazır derlenmiş sürümü çalıştırmak için `python3 -m http.server 8080 --bind 127.0.0.1 --directory dist` komutunu kullan; ardından http://127.0.0.1:8080 adresini aç. Hazır sürüm internet bağlantısı gerektirmez. HTML dosyasını doğrudan çift tıklamak yerine yerel sunucu kullan.

## Kendi kodunu bağla

Kendi kodun veya yerel LLM'in şu biçimde JSON üretsin. Arayüzde **Mimari yükle** veya **JSON düzenle** üzerinden ver:

```json
{
  "name": "Benim ağım",
  "nodes": [
    {"id": "girdi", "label": "Girdi", "type": "input", "shape": [8, 64]},
    {"id": "ozel", "label": "Özel katmanım", "type": "custom", "shape": [8, 128], "params": 8320, "description": "Kendi işlemim"},
    {"id": "cikti", "label": "Çıktı", "type": "output", "shape": [8, 4]}
  ],
  "edges": [
    {"source": "girdi", "target": "ozel"},
    {"source": "ozel", "target": "cikti"}
  ]
}
```

- `id`: benzersiz katman kimliği. `shape`: taşınan verinin boyutları; ağırlık matrisinin boyutu olmak zorunda değildir.
- `type`: input, embedding, linear, attention, norm, add, output veya istediğin özel bir ad. Tür yalnızca rengi belirler; matematiksel işlem çalıştırmaz.
- `params`: isteğe bağlı parametre sayısı. Arayüz verilen sayıları toplar; kodundan kendisi hesaplamaz. Paylaşılan ağırlıkları iki kez saymamak senin sorumluluğunda.
- `description`: isteğe bağlı basit açıklama.
- `position`: isteğe bağlı `[x,y,z]` konumu. Verilmezse bağlantılara göre otomatik yerleşir.
- `values`: isteğe bağlı, düzleştirilmiş gerçek sayılar dizisi. Sağlanırsa hücre renkleri bu değerlerden örneklenir. Her katmanın renk ölçeği kendi en büyük mutlak değerine göredir. Sağlanmazsa katman türü rengi kullanılır.

Limits: 160 katman, 600 bağlantı, 8 MB dosya; her shape 1–6 boyut, her boyut 1–1 milyar; katman başına en fazla 100.000 değer. Hücre çizimleri en fazla 22×10×3 hücreye indirgenir. Shape ile values uzunluğu eşit olmak zorunda değildir; kısmi değer örnekleri kabul edilir. Otomatik yerleşim yönlü döngüsüz ağlara göre tasarlanmıştır; döngülü bölümler giriş sırasıyla yerleştirilir. Karmaşık ağlarda `position` kullanabilirsin.

### JavaScript bağlantısı

Sayfa içindeki kendi kodundan:

```js
window.neuralAtlas.load(mimariNesnesi);
window.neuralAtlas.select('ozel');
window.neuralAtlas.fit();
```

Kendi yerel servisini sen bağlayabilirsin:

```js
const mimari = await fetch('/kendi-mimari-endpointin').then(r => r.json());
window.neuralAtlas.load(mimari);
```

Bu örnek çağrı otomatik yapılmaz. Farklı bir porttaki servisten okuma için o servisin tarayıcı erişimine (CORS) izin vermesi gerekir.

### Çizim motorunu ayrı kullan

`src/renderer.js` içindeki `NeuralRenderer` arayüzden ayrıdır:

```js
import { NeuralRenderer } from './src/renderer.js';
const viewer = new NeuralRenderer(container, { onSelect: node => console.log(node) });
viewer.load(mimariNesnesi);
viewer.fit();
// Bileşeni kaldırırken:
viewer.dispose();
```

Kapsayıcı elemanın genişliği ve yüksekliği olmalı. Three.js bağımlılığı bir paketleyiciyle derlenmeli. `src/model.js` JSON doğrulaması ve otomatik yerleşimi içerir.

## Sınırlar

Model çalıştırılmaz, Python kodu analiz edilmez, yerel LLM bağlantısı kurulmaz. Akış oynatma katmanları JSON sırasıyla vurgulayan şematik bir anlatımdır; gerçek hesaplama veya zamanlama değildir. Örnek Transformer öğretici bir blok şemasıdır, eksiksiz bir çalıştırılabilir model değildir. PNG, 3B çizimi içerir; HTML etiketleri ve paneller dahil edilmez. JSON yalnızca açık sayfada tutulur; saklamak için örnek JSON indirme düğmesi güncel mimariyi kaydeder.

## Doğrulama

`npm test` veri doğrulaması, paralel kollar, atlama bağlantıları, döngüler ve elle konumlandırmayı kontrol eder. `npm run build` dağıtılabilir dosyaları `dist/` altında üretir.
