# Fraud Detection Modelling dengan Keras Preprocessing dan Class Weighting

Fraud detection sudah menjadi studi kasus populer untuk klasifikasi dimana data nya tidak seimbang (*imbalance*). Disini data yang [digunakan](https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv) memiliki persentase kelas positif (sampel dimana terjadi penipuan) sebesar 0.58% dari total 1296675 sampel. Buka notebook di [Google Colab](https://colab.research.google.com/github/zeerafle/fraud-detection/blob/main/Fraud_Detection.ipynb)

```python
neg, pos = np.bincount(raw_df['is_fraud'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
```
```console
Examples:
	Total: 1296675
	Positive: 7506 (0.58% of total)
```

## Feature Engineering

Sebelum mulai melatih model tentukan fitur fitur yang akan digunakan. Kolom kolom yang tersedia pada dataset antara lain: `trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `first`, `last`, `gender`, `street`, `city`, `state`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`, `trans_num`, `unix_time`, `merch_lat`, `merch_long`, dan `is_fraud`

### Umur

Umur seseorang pada saat transaksi bisa didapatkan dari selisih antara tahun transaksi pada kolom `trans_date_trans_time` dan tahun kelahiran pada kolom `dob`

```python
trans_date = pd.to_datetime(raw_df.trans_date_trans_time, format="%Y-%m-%d %H:%M:%S")
dob = pd.to_datetime(raw_df.dob, format="%Y-%m-%d")

raw_df['age_when_trans'] = trans_date.map(lambda x: x.year) - dob.map(lambda x: x.year)
```

Dari histogram dapat dilihat transaksi banyak dilakukan oleh orang berumur 30-50 dibanding yang lain.

![](https://raw.githubusercontent.com/zeerafle/fraud-detection/master/images/Pasted%20image%2020220909090528.png)
### Jam terjadi transaksi

Jam transaksi mulai dari 00 hingga 23 bisa didapatkan dengan mengakses atribut `hour` pada setiap elemen `trans_date`.

```python
raw_df['time_hour'] = trans_date.map(lambda x: x.hour)
```

![](https://raw.githubusercontent.com/zeerafle/fraud-detection/master/images/Pasted%20image%2020220909091312.png)

### Hari terjadi transaksi

Hari dalam minggu disini bernilai 0-6 dimana 0 = senin dan 6 = minggu. Bisa didapatkan dengan memanggil method `weekday` pada tiap elemen `trans_date`.

```python
raw_df['weekday'] = trans_date.map(lambda x: x.weekday())
```

![](https://raw.githubusercontent.com/zeerafle/fraud-detection/master/images/Pasted%20image%2020220909091700.png)

### Jarak tempat tinggal terhadap merchant

Dalam data terdapat kolom `lat` dan `long` yang menunjukkan lokasi pemilik kartu kredit serta kolom `merch_lat` dan `merch_long` yang menunjukkan lokasi merchant dimana transaksi terjadi. Jaraknya bisa didapat dengan menghitung [euclidean distance]([Euclidean distance - Wikipedia](https://en.wikipedia.org/wiki/Euclidean_distance)) pada nilai latitude dan longitude.

```python
def euclidean(params):
    lon1, lat1, lon2, lat2 = params
    londiff = lon2 - lon1
    latdiff = lat2 - lat1
    return np.sqrt(londiff*londiff + latdiff*latdiff)

raw_df['distance'] = raw_df[['lat', 'long', 'merch_lat', 'merch_long']].apply(euclidean, axis=1)
```

### Split data

Pisahkan data menjadi training dan validation. Data test sudah disiapkan oleh penyedia data.

## Keras Preprocessing Layer

### Buat tensorflow dataset

Tensorflow dataset api menyediakan metode yang efisien untuk membuat input pipeline yang optimal.

```python
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(LABEL_COLUMN)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

BATCH_SIZE = 2048
train_ds = df_to_dataset(train_df, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val_df, shuffle=False, batch_size=BATCH_SIZE)
```

### Buat input pipeline

Input pipeline akan membuat input layer, normalization layer, dan category encoding layer pada arsitektur model nanti. Sehingga preprocessing data dapat berjalan sekaligus saat model dilatih.

```python
def get_normalization_layer(name, dataset):
  normalizer = layers.Normalization(axis=None)
  feature_ds = dataset.map(lambda x, y: x[name])
  normalizer.adapt(feature_ds)
  return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  feature_ds = dataset.map(lambda x, y: x[name])
  index.adapt(feature_ds)
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  return lambda feature: encoder(index(feature))
```

Buat variabel penampung input dan fitur yang sudah di preprocess. Selain itu juga kelompokkan fitur fitur kategorikal, diskrit, dan numerik (kontinu).

```python
all_inputs = dict()
encoded_features = []

CATEGORICAL_COLS = ['category', 'gender']
DISCRETE_COLS = ['time_hour', 'weekday']
NUMERIC_COLS = ['amt', 'city_pop', 'lat',
                'long', 'merch_lat', 'merch_long',
                'age_when_trans', 'distance']
```

Kemudian input pipeline pada setiap jenis kolom

```python
# Numerical features.
for header in NUMERIC_COLS:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs[header] = numeric_col
    encoded_features.append(encoded_numeric_col)

# categorical features
for header in CATEGORICAL_COLS:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(name=header,
                                                dataset=train_ds,
                                                dtype='string',
                                                max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs[header] = categorical_col
    encoded_features.append(encoded_categorical_col)

# discrete features
for header in DISCRETE_COLS:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
    encoding_layer = get_category_encoding_layer(name=header,
                                                dataset=train_ds,
                                                dtype='int64')
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs[header] = categorical_col
    encoded_features.append(encoded_categorical_col)
```

Ubah input layer menjadi list agar dapat di satukan saat membuat arsitektur model.

```python
all_inputs_list = list(all_inputs.values())
```

## Buat Model dan Metrik

### Metrik

Beberapa metrik yang akan digunakan antara lain:

- False negative dan false positive yaitu sampel yang diklasifikasikan salah 
- True negative dan true positive yaitu sampel yang diklasifikasikan benar 
- Accuracy, yaitu persentase jumlah sampel yang diklasifikasikan benar

	![](https://blog.paperspace.com/content/images/2020/09/Fig06.jpg)

- Precision, yaitu persentase **prediksi** positif yang diklasifikasikan dengan benar

	![](https://blog.paperspace.com/content/images/2020/09/Fig07.jpg)

- Recall, yaitu presentase positif aktual yang diklasifikasikan benar

	![](https://blog.paperspace.com/content/images/2020/09/Fig11.jpg)

- AUC, yaitu Area Under the Curve pada kurva Receiver Operating Characteristic (ROC-AUC). Metrik ini sama dengan probabilitas bahwa pengklasifikasi akan memberi peringkat sampel positif random lebih tinggi daripada sampel negatif random
- AURPC, yaitu Area Under the Curve dari kurva Precision-Recall. Metrik ini menghitung pasangan precision-recall pada ambang (threshold) probabilitas yang berbeda.

```python
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
```

Buat callback early stopping yang memonitor precision-recall curve. Akurasi tidak digunakan karena masalah pada data yang tidak seimbang sehingga akurasi bisa bernilai tinggi dengan selalu memprediksi kelas False (bukan penipuan).

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
```

### Model

Buat dugaan awal pada nilai bias. Ini akan membantu proses konvergen. Nilai bias didapat dari rumus dibawah. Spesifikasikan nilai awal bias ini pada layer output model.

![](https://raw.githubusercontent.com/zeerafle/fraud-detection/master/images/Pasted%20image%2020220909103826.png)

```python
initial_bias = np.log([pos/neg])
print(initial_bias) # -5.1460504
```

Selanjutnya merge seluruh input fitur --`encoded_features`--ke dalam satu vektor menggunakan `tf.keras.layers.concatenate`. Arsitektur model ini memiliki 3 hidden layer dengan layer pertama dan kedua berjumlah 32 neuron, layer ke tiga berjumlah 16 neuron.

Compile model dengan optimizer adam dan loss binary crossentropy.

```python
def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    all_features = tf.keras.layers.concatenate(encoded_features)
     x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dense(32, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.L1(0.001),
                              activity_regularizer=tf.keras.regularizers.L2(0.001))(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid',
                                   bias_initializer=output_bias)(x)

    model = tf.keras.Model(all_inputs, output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    return model
```

Graf modelnya akan terlihat seperti ini

```python
model = make_model(output_bias=initial_bias)
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
```

![](https://raw.githubusercontent.com/zeerafle/fraud-detection/master/images/plot_model.png)

## Class Weight

Tujuan disini adalah memberi sampel positif (sampel dimana terjadi penipuan) bobot yang lebih besar karena jumlah sampel nya yang sedikit. Sehingga model akan "lebih melirik" kelas dengan sampel yang sedikit.

```python
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0)) # 0.50
print('Weight for class 1: {:.2f}'.format(weight_for_1)) # 86.38
```

### Latih model dengan bobot kelas

Nilai `class_weight` dapat dimasukkan dalam parameter `class_weight` saat memanggil method `fit` pada model.

```python
model.load_weights(initial_weights)
weighted_history = model.fit(
    train_ds,
    epochs=100,
    callbacks=[early_stopping],
    validation_data=val_ds,
    # class weights
    class_weight=class_weight)
```

## Evaluasi Model

Berikut grafik loss, precision-recall curve, precision, dan recall pada data training dan validation selama proses training yang berhenti pada epoch 82. Namun bobot terbaik yang digunakan adalah bobot pada saat epoch 72.

![](https://raw.githubusercontent.com/zeerafle/fraud-detection/master/images/plot_metrics.png)

- Transaksi sah yang terdeteksi (true negative) sebesar 533594
- Transaksi sah yang dianggap penipuan (false positive) sebesar 19980
- Transaksi penipuan yang tidak terdeteksi (false negative) sebesar 191
- Transaksi penipuan yang terdeteksi (true positive) sebesar 1954
- Total transaksi penipuan sebesar 2145

```
loss :  0.0969950333237648
tp :  1954.0
fp :  19980.0
tn :  533594.0
fn :  191.0
accuracy :  0.9637028574943542
precision :  0.08908543735742569
recall :  0.9109557271003723
auc :  0.9836622476577759
prc :  0.6992051601409912
```

![](https://raw.githubusercontent.com/zeerafle/fraud-detection/master/images/confusion_matrix.png)


