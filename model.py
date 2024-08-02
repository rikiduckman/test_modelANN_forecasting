import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# อ่านไฟล์ ARFF
def load_arff(file_path):
    with open(file_path, 'r') as file:
        data = arff.load(file)
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
    return df

# โหลดข้อมูล
file_path = r'C:\Users\rikid\Desktop\data\File_N\data.arff'  # ใช้ path ที่ถูกต้อง
df = load_arff(file_path)

# เตรียมข้อมูลสำหรับการสร้างโมเดล
X = df.drop(columns=['Label'])
y = df['Label']

# แปลงข้อมูล y เป็น one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# แปลงข้อมูล X ที่เป็นหมวดหมู่เป็น one-hot encoding
categorical_columns = X.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
    remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_categorical, test_size=0.3, random_state=100)

# สเกลข้อมูล
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล Neural Network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# ฟังก์ชันสำหรับการพยากรณ์
def predict(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_processed = preprocessor.transform(input_df)
    input_scaled = scaler.transform(input_processed)
    prediction = model.predict(input_scaled)
    predicted_class = label_encoder.inverse_transform([prediction.argmax()])
    return predicted_class[0]