###############################################################
# #1.AŞAMA -> ÖNCE BU KISMI ÇALIŞTIRIN!
###############################################################

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(train_csv_path, test_csv_path, train_dir, test_dir):
    """
    Eğitim ve test verilerini model eğitimine uygun hale getirir.
    Bu fonksiyon, eğitim verilerini içeren CSV dosyasını okuyarak etiketleri sayısallaştırır ve
    görüntü dosyalarına tam yollar ekler. Ayrıca, eğitim verilerini eğitim ve doğrulama verisine ayırır.
    """
    # Eğitim CSV'sini yükle
    # Eğitim verisi, her satırda bir görüntü dosyasının adı ve ona karşılık gelen etiket bilgisini içerir
    train_df = pd.read_csv(train_csv_path)
    
    # Etiketleri sayısallaştır (ör: Istanbul -> 0, Ankara -> 1, Izmir -> 2)
    # 'city' sütunundaki her şehir değeri, bir sayıya dönüştürülür
    label_mapping = {label: idx for idx, label in enumerate(train_df['city'].unique())}
    train_df['label_idx'] = train_df['city'].map(label_mapping)

    # Görüntü yollarını oluştur
    # 'filename' sütunundaki dosya adı ile 'train_dir' dizini birleştirilerek tam dosya yolu oluşturulur
    train_df['image_path'] = train_df['filename'].apply(lambda x: os.path.join(train_dir, x))

    # Test CSV'sini yükle ve yollarını oluştur
    # Test verisi de aynı şekilde işlenir, ancak doğrulama yapılmaz
    test_df = pd.read_csv(test_csv_path)
    test_df['image_path'] = test_df['filename'].apply(lambda x: os.path.join(test_dir, x))

    # Eğitim ve doğrulama verilerini ayır
    # 'train_test_split' fonksiyonu ile eğitim verisi %80 ve doğrulama verisi %20 oranında ayrılır
    # 'stratify' parametresi, etiketlerin (label_idx) dağılımının orantılı olmasını sağlar
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label_idx'])

    # Hazırlanan verileri geri döndür
    return train_data, val_data, test_df, label_mapping

# Burada, eğitim ve test CSV dosyalarının yolları ve ilgili görüntülerin bulunduğu dizinler belirtilir
train_csv_path = r'C:\Users\Sema\Desktop\DatathonAI\Dataset\train_data.csv'
test_csv_path = r'C:\Users\Sema\Desktop\DatathonAI\Dataset\test.csv'
train_dir = r'C:\Users\Sema\Desktop\DatathonAI\Dataset\train\train'
test_dir = r'C:\Users\Sema\Desktop\DatathonAI\Dataset\test\test'

# Veriyi hazırlama fonksiyonunu çağır
train_data, val_data, test_data, label_mapping = prepare_data(train_csv_path, test_csv_path, train_dir, test_dir)

# Çıktıları CSV'ye kaydet
# Eğitim, doğrulama ve test verileri ayrı ayrı CSV dosyalarına kaydedilir
train_data.to_csv('train_data_output.csv', index=False)
val_data.to_csv('val_data_output.csv', index=False)
test_data.to_csv('test_data_output.csv', index=False)

# Etiket haritasını bir metin dosyasına kaydet
# Etiket haritası, her şehir ismi ve sayısal karşılıklarıyla birlikte 'label_mapping.txt' dosyasına yazılır
with open('label_mapping.txt', 'w') as f:
    for city, idx in label_mapping.items():
        f.write(f'{city}: {idx}\n')

# Kontrol mesajları
# İşlemlerin başarıyla tamamlandığını belirten çıktılar
print("Eğitim verisi CSV'ye kaydedildi.")
print("Doğrulama verisi CSV'ye kaydedildi.")
print("Test verisi CSV'ye kaydedildi.")
print("Etiket haritası metin dosyasına kaydedildi.")

##############################################################
# 2.AŞAMA -> DOSYALAR KAYDEDİLDİKTEN SONRA BU KISMI ÇALIŞTIRIN!
##############################################################

import os
import shutil
import pandas as pd

def organize_images(train_csv_path, val_csv_path, train_dir, val_dir, test_csv_path, test_dir, output_dir):
    """
    Verileri etiketlerine göre uygun klasörlere ayırır.
    Bu fonksiyon, eğitim, doğrulama ve test verilerindeki görüntüleri
    belirtilen dizinlere kopyalar ve her etiket için ayrı klasörler oluşturur.
    """
    # Eğitim, doğrulama ve test verilerini CSV dosyalarından yükle
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Çıktı dizini oluştur
    # Eğitim, doğrulama ve test verileri için ana klasörler oluşturuluyor
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # Eğitim ve doğrulama etiketlerini al
    train_labels = train_df['label_idx'].unique()
    val_labels = val_df['label_idx'].unique()

    # Eğitim verilerini etiketlere göre klasörlere yerleştir
    for label in train_labels:
        label_name = str(label)  # Etiket ismi
        os.makedirs(os.path.join(output_dir, 'train', label_name), exist_ok=True)  # Etiket klasörünü oluştur
        label_images = train_df[train_df['label_idx'] == label]['image_path'].values  # Etiketlere göre resim yollarını al
        for image_path in label_images:
            if os.path.exists(image_path):  # Dosya mevcutsa
                # Dosyayı ilgili klasöre kopyala
                shutil.copy(image_path, os.path.join(output_dir, 'train', label_name, os.path.basename(image_path)))
            else:
                # Dosya bulunamazsa hata mesajı yazdır
                print(f"Dosya bulunamadı: {image_path}")

    # Doğrulama verilerini etiketlere göre klasörlere yerleştir
    for label in val_labels:
        label_name = str(label)  # Etiket ismi
        os.makedirs(os.path.join(output_dir, 'val', label_name), exist_ok=True)  # Etiket klasörünü oluştur
        label_images = val_df[val_df['label_idx'] == label]['image_path'].values  # Etiketlere göre resim yollarını al
        for image_path in label_images:
            if os.path.exists(image_path):  # Dosya mevcutsa
                # Dosyayı ilgili klasöre kopyala
                shutil.copy(image_path, os.path.join(output_dir, 'val', label_name, os.path.basename(image_path)))
            else:
                # Dosya bulunamazsa hata mesajı yazdır
                print(f"Dosya bulunamadı: {image_path}")

    # Test verilerini kopyala (Test verisinde etiketler yok, sadece resim yolları var)
    for image_path in test_df['image_path'].values:
        if os.path.exists(image_path):  # Dosya mevcutsa
            # Dosyayı 'test' klasörüne kopyala
            shutil.copy(image_path, os.path.join(output_dir, 'test', os.path.basename(image_path)))
        else:
            # Dosya bulunamazsa hata mesajı yazdır
            print(f"Dosya bulunamadı: {image_path}")
    
    # İşlemlerin tamamlandığını belirten mesaj
    print("Görüntüler başarıyla organize edildi.")

# Dosya yolları (verdiğiniz yollara göre)
train_csv_path = r'C:\Users\Sema\Desktop\DatathonAI\train_data_output.csv'
val_csv_path = r'C:\Users\Sema\Desktop\DatathonAI\val_data_output.csv'  # Eğer doğrulama veriniz yoksa, val_data.csv'yi atlayabilirsiniz
test_csv_path = r'C:\Users\Sema\Desktop\DatathonAI\test_data_output.csv'
train_dir = r'C:\Users\Sema\Desktop\DatathonAI\Dataset\train\train'
test_dir = r'C:\Users\Sema\Desktop\DatathonAI\Dataset\test\test'
output_dir = r'C:\Users\Sema\Desktop\DatathonAI\Dataset\organized_data'  # Yeni çıkış dizini

# Görüntüleri organize et
organize_images(train_csv_path, val_csv_path, train_dir, test_dir, test_csv_path, test_dir, output_dir)

#######################################################################################
# 3.AŞAMA -> DOSYALAR KAYDEDİLDİKTEN SONRA EĞİTİME BAŞLAMAK İÇİN BU KISMI ÇALIŞTIRIN!
#######################################################################################

from ultralytics import YOLO

if __name__ == '__main__':
    """
    YOLO modelini yükler, eğitir ve sınıflandırma işlemi gerçekleştirir.
    Bu script, YOLO modelini kullanarak görüntü sınıflandırma için eğitim yapar.
    """

    # Modeli yükle (önceden eğitilmiş bir model kullanılıyor)
    # Burada "yolo11s-cls.pt" modelini yükleyerek başlangıç için küçük bir model kullanıyoruz.
    model = YOLO("yolo11s-cls.pt")  # Küçük bir model (başlangıç için uygun)

    # Modeli eğit
    model.train(
        data=r"C:\Users\Sema\Desktop\DatathonAI\Dataset\organized_data",  # Veri seti yolu (yapılandırma dosyası burada bulunuyor)
        epochs=50,  # Eğitim sayısı (kaç epoch boyunca eğitim yapılacağı)
        imgsz=640,  # Görüntü boyutu (modelin giriş görüntülerinin boyutu)
        batch=16,  # Mini-batch boyutu (her adımda kaç görüntü işlenecek)
        name='yolo_classification',  # Eğitim deneyimi adı (sonuçlar burada kaydedilecek)
        device='cuda'  # GPU kullanımı (eğer GPU yoksa 'cuda' yerine 'cpu' kullanılabilir)
    )

    # Eğitim tamamlandıktan sonra model kaydedilecek ve sonuçlar belirtilen isimle çıkacak.
    # Model, verilen veri seti üzerinde belirtilen parametrelerle eğitilecektir.

######################################################################################
# #4.AŞAMA -> MODEL EĞİTİMİNDEN SONRA TEST İÇİN BU KISMI ÇALIŞTIRIN!
######################################################################################

import os
import csv
from ultralytics import YOLO


#    """
#    ‘YOLO’ sınıfı kullanılarak önceden eğitilmiş bir model (best.pt) yüklenir.
#     Bu model, test görüntüleri üzerinde sınıflandırma yapacaktır.
#     """

model = YOLO(r"C:\Users\Sema\Desktop\DatathonAI\runs\classify\yolo_classification\weights\best.pt")

#    """
#    Test Görüntüleri Dizini
#    Test için kullanılacak görüntülerin bulunduğu dizin belirtilir.
#    """
test_dir = r"C:\Users\Sema\Desktop\DatathonAI\Dataset\test\test"

#     """
#     Test Sonuçları Listesi
#     Her görüntünün tahmin edilen sınıfını saklamak için boş bir liste oluşturulur.
#     """
test_results = []

#     """
#     Test Görüntülerini İşleme
#     Belirtilen dizindeki her görüntü için sınıflandırma tahmini yapılır.
#     """
for image_file in os.listdir(test_dir):
    # Görüntünün tam dosya yolu oluşturulur.
    image_path = os.path.join(test_dir, image_file)
    
    # Model üzerinde görüntü işlenir ve sonuçlar alınır.
    results = model(image_path)
    
    # Tahmin edilen sınıf bilgisi alınır.
    # top1 özelliği, en olası sınıfın etiketini döndürür.
    predicted_class = results[0].probs.top1

    # Görüntü adı ve tahmin edilen sınıf, sonuç listesine eklenir.
    test_results.append({"image": image_file, "predicted_class": predicted_class})

#     """
#     Test Sonuçlarını CSV Dosyasına Kaydetme
#     Tahmin edilen sonuçlar bir CSV dosyasına yazılır.
#     """
csv_file_path = r"C:\Users\Sema\Desktop\DatathonAI\Dataset\organized_data\test_predictions.csv"
with open(csv_file_path, mode="w", newline="") as file:
    # CSV yazıcı nesnesi, sütun adlarını tanımlar.
    writer = csv.DictWriter(file, fieldnames=["image", "predicted_class"])
    
    # CSV dosyasına başlık satırı yazılır.
    writer.writeheader()
    
    # Tüm sonuçlar CSV dosyasına eklenir.
    writer.writerows(test_results)

#     """
#     İşlem Tamamlandığında Bilgilendirme
#     CSV dosyasının başarıyla oluşturulduğunu ekrana yazdırır.
#     """
print(f"Test sonuçları '{csv_file_path}' dosyasına kaydedildi.")

#     """
#     Şehir Haritası
#     Predicted label (tahmin edilen sınıf) değerlerinin şehir adlarına dönüştürülmesi için harita oluşturulur.
#     """
city_map = {
    0: 'Istanbul',
    1: 'Ankara',
    2: 'Izmir'
}

#     """
#     CSV Dosyasını Okuma ve Dönüştürme
#     Bu adımda, ilk oluşturduğumuz CSV dosyasındaki 'predicted_class' değerini şehir isimleriyle değiştiriyoruz
#     ve yeni bir CSV dosyasına kaydediyoruz.
#     """
input_csv = r"C:\Users\Sema\Desktop\DatathonAI\Dataset\organized_data\test_predictions.csv"  # Giriş CSV dosyasının yolu
output_csv = r"C:\Users\Sema\Desktop\DatathonAI\Dataset\organized_data\test_predictions_output.csv"  # Çıkış CSV dosyasının yolu

with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['image_path', 'predicted_label']  # Yeni CSV dosyasına yazılacak sütunlar
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        # Görüntü adı, yolu üzerinden çıkarılır
        image_name = row['image'].split('/')[-1]
        
        # Tahmin edilen sınıfın (predicted_label) şehre karşılık gelen adı ile değiştirilmesi
        predicted_city = city_map[int(row['predicted_class'])]
        
        # Yeni satır yazılır
        writer.writerow({'image_path': image_name, 'predicted_label': predicted_city})

#     """
#     Yeni CSV Dosyasının Kaydedilmesi
#     Sonuç olarak, modelin tahmin ettiği sınıf etiketleri şehir adlarına dönüştürülüp
#     yeni bir CSV dosyasına kaydedilir.
#     """
print(f"Yeni CSV dosyası '{output_csv}' olarak kaydedildi.")
