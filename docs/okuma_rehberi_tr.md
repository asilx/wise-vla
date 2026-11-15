# WISE-VLA kod okuma rehberi

WISE-VLA, Grounded Forge için hazır bir VLM üzerine kurulan, graph ve action katmanları bu repoda sıfırdan yazılmış bir PyTorch VLA implementasyonu. Grounded-VLA projesindeki dünya bilgisiyle koşullandırma fikrini yeni bir eylem politikasının içine taşıyor. Önceki repodaki bir VLA checkpoint'ini kullanmıyor.

## İlk çalıştırma

```bash
pip install -e '.[dev]'
wise-vla-demo --output runs/ilk-deneme --train-steps 20
```

Bu komut sentetik görüntü, robot durumu, graph ve eylem verisi üretir; küçük test modeliyle eğitim yapar; checkpoint'i yeniden yükler; bir eylem parçası üretir. Bu demo robot becerisi veya hazır VLM kalitesini ölçmez. Gerçek VLM için `configs/qwen3_vl_2b.yaml` kullanılır.

## Okuma sırası

| Sıra | Dosya | Anlamaya odaklan |
|---|---|---|
| 1 | `src/wise_vla/config.py` | State/action boyutu, horizon, graph sözlüğü ve VLM ayarları |
| 2 | `src/wise_vla/demo.py` | Bir eğitim örneği nasıl oluşturuluyor? |
| 3 | `src/wise_vla/graph.py` | Nesneler, yönlü ilişkiler, güven değerleri ve zaman bilgisi nasıl tensöre dönüşüyor? |
| 4 | `src/wise_vla/backbones.py` | Hazır Qwen görüntü/dil token'larını ve nesne etiketi embedding'lerini nasıl sağlıyor? |
| 5 | `src/wise_vla/nn/graph_encoder.py` | İlişki türü ve yönü attention'ı ve node representation'ını nasıl etkiliyor? |
| 6 | `src/wise_vla/nn/action_expert.py` | Action token'ları graph, VLM, robot durumu ve zamanla nasıl koşullandırılıyor? |
| 7 | `src/wise_vla/model.py` | Flow-matching loss ve örnekleme birbirine nasıl karşılık geliyor? |
| 8 | `src/wise_vla/data/` | Bölüm sınırları, eksik action maskeleri ve normalizasyon |
| 9 | `src/wise_vla/train.py`, `checkpoint.py` | Optimizer, AMP, DDP, EMA ve kaldığı yerden devam |
| 10 | `src/wise_vla/policy.py`, `evaluate.py` | Robot arayüzü ve graph etkisinin ölçülmesi |

## Graph nerede gerçekten kullanılıyor?

`GraphTensorizer`, ilişkileri `[batch, source, target, relation]` biçiminde bir tensöre dönüştürüyor. `RelationBlock` bu tensörü hem attention bias hem de ilişki mesajları için kullanıyor. Ardından her `ActionBlock` içindeki `graph_attn`, eylem token'larını graph token'larıyla doğrudan koşullandırıyor. `flow_loss.backward()` bu yoldan graph encoder'a da gradyan gönderiyor.

Nesne etiketleri VLM'nin pretrained token embedding'lerinden, görüntü/dil bağlamı ise VLM'nin son hidden state'lerinden geliyor. Bbox ve 3B pozisyonlar graph girdisi. Bu sürüm görüntüden otomatik olarak doğru nesne/ilişki grafı çıkaran bir perception sistemi içermiyor; bu veriyi Isaac ground truth, tracker veya KnowRob tarafındaki bir dönüştürücü sağlamalı.

## Flow matching için temel denklem

`x_t = (1-t) * noise + t * action`.

Eğitimde model `action-noise` hızını tahmin ediyor. Çıkarımda `t=0` gürültüden başlayıp `t=1` eyleme doğru Euler veya Heun adımlarıyla ilerliyor. Aynı observation için birden fazla gelecek eylem birlikte üretiliyor. Controller, bu parçanın ilk birkaç adımını uygulayıp yeni observation ile yeniden planlayabilir.

## Öğrenilmesi faydalı konular

- PyTorch attention maskeleri, autograd, AMP ve DDP.
- Conditional flow matching; noise-to-data yönü ve ODE örnekleme.
- Graph transformer, yönlü relation bias ve node permutation equivariance.
- Action chunking, episode boundary maskeleri ve train/validation veri sızıntısı.
- Robot koordinat dönüşümleri, kamera kalibrasyonu ve state/action zaman hizalaması.

Önce `tests/test_model.py` ve `tests/test_graph.py` içindeki davranış testlerini okuyup ilgili modüle dönmek de iyi bir yol. Testler yalnızca shape kontrolü yapmıyor; graph değişince eylemin değiştiğini, masked padding'in sonucu etkilemediğini ve eğitim sinyalinin doğru katmanlara ulaştığını kontrol ediyor.

## Gerçek robot için sıradaki somut iş

Bir görev ve tek bir robot action şeması seçip küçük, doğru hizalanmış bir demonstrasyon seti hazırlamak gerekiyor. İlk gerçek eğitimde Qwen'i dondurup graph ve action katmanlarını eğitmek; ardından graph'lı/graph'sız kontrollü değerlendirme yapmak uygun. Şu an repo eğitim ve çıkarım yazılımını sağlıyor; eğitilmiş robot policy ağırlıkları, Isaac görev entegrasyonu ve gerçek robot başarı ölçümü henüz yok.
