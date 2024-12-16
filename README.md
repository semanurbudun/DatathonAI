Proje Ağırlıkları runs klasöründe bulunmaktadır.
# DatathonAI
 Bu proje, DatathonAI Ön Eleme kapsamında Google Street View görüntülerinden Türkiye şehirlerini tanımlayabilen bir yapay zeka modelini geliştirmek amacıyla oluşturulmuştur. Model, sokak görüntülerinin özelliklerini analiz ederek, her şehri benzersiz kılan mimari stiller ve coğrafi özellikler üzerinden doğru tahminler yapmaktadır.
 
Kod, bir YOLO tabanlı görüntü sınıflandırma modeli eğitmek ve test etmek için kullanılan birkaç aşamalı bir süreçtir. Her aşama belirli bir işlevi yerine getirir. İşte her aşamanın ayrıntıları:

1. Veri Hazırlığı (Data Preparation)
İlk aşama, eğitim, doğrulama ve test verilerini hazırlamaya yöneliktir. Bu işlem şu adımları içerir:

CSV Dosyasını Okuma: Eğitim ve test verileri içeren CSV dosyaları okunur. Bu CSV dosyaları, her bir görüntü için dosya adı ve etiket bilgisini içerir.
Etiketleri Sayısallaştırma: 'city' adlı sütundaki şehir adları sayılara dönüştürülür, böylece model sayısal etiketlerle çalışabilir.
Görüntü Yollarını Güncelleme: Görüntü dosya adlarıyla birlikte tam yollar oluşturulur.
Eğitim ve Doğrulama Verilerini Ayırma: train_test_split kullanılarak eğitim verisi ve doğrulama verisi ayrılır (eğitim verisinin %80'i, doğrulama verisinin %20'si).
Dosya Çıktıları: İşlenen veriler CSV formatında kaydedilir ve etiketler bir metin dosyasına yazılır.

2. Görüntülerin Organize Edilmesi (Organizing Images)
Bu aşama, görüntüleri etiketlerine göre uygun klasörlere kopyalamayı içerir:

Eğitim, Doğrulama ve Test Klasörlerini Oluşturma: Veriler etiketlerine göre train, val ve test klasörlerine ayrılır.
Dosyaların Kopyalanması: Her bir etiket için ilgili görüntü dosyaları belirtilen klasörlere kopyalanır.
Hata Yönetimi: Dosya bulunamadığında hata mesajları yazdırılır.

3. Model Eğitimi (Model Training)
Bu aşama, YOLO modelinin eğitilmesini içerir:
Model Yükleme: Öntanımlı bir YOLO modelini yükler (bu örnekte yolo11s-cls.pt modeli kullanılır).
Modeli Eğitme: Eğitim verisi üzerinde modelin eğitilmesi yapılır. Model, belirtilen parametrelerle eğitilir (epoch sayısı, batch boyutu, görüntü boyutu, vb.).
Sonuçların Kaydedilmesi: Eğitim sonuçları belirtilen dizine kaydedilir.

5. Test (Model Testing)
Eğitim tamamlandıktan sonra model test edilir:

Model Yükleme: Eğitilen modelin en iyi ağırlıkları yüklenir (best.pt).
Test Görüntüleri ile Sınıflandırma: Test verisi üzerinde tahminler yapılır. Her görüntü için model tarafından tahmin edilen sınıf alınır.
Sonuçları Kaydetme: Tahmin edilen sınıflar bir dosyaya kaydedilir (bu adım tamamlanmamış).
