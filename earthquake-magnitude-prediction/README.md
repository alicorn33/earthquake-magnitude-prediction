# 🌏 Earthquake Magnitude Level Prediction (Thailand Region)

โปรเจกต์นี้สร้างโมเดล Machine Learning เพื่อ**ทำนายระดับความรุนแรงของแผ่นดินไหว
(Magnitude Level Classification)** จาก**ตำแหน่งพื้นที่ (ละติจูด/ลองจิจูด)** และ
**ความลึก (depth)** โดยใช้ข้อมูลแผ่นดินไหวในภูมิภาคประเทศไทยและใกล้เคียง
พร้อมเว็บแอปพลิเคชันสำหรับทำนายผลแบบ interactive ด้วย **Streamlit**

## 📂 โครงสร้างโปรเจกต์

```
earthquake-magnitude-prediction/
├── data/
│   └── thailand_earthquakes.csv   # ข้อมูลดิบ (จาก USGS)
├── model/                         # ไฟล์โมเดลที่เทรนแล้ว (สร้างโดย train_model.py)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_cols.pkl
│   ├── eda_overview.png
│   └── confusion_matrix.png
├── train_model.py                 # สคริปต์เทรนโมเดล (EDA -> Preprocess -> Train -> Evaluate -> Save)
├── app.py                         # เว็บแอป Streamlit สำหรับทำนายผล
├── requirements.txt
└── README.md
```

## ⚙️ ขั้นตอนการทำงานของโมเดล (`train_model.py`)

1. **Import ข้อมูล** — โหลดข้อมูลจาก `data/thailand_earthquakes.csv`
2. **EDA** — ตรวจสอบค่าว่าง, สถิติเชิงพรรณนา, กราฟการกระจายตัวของ magnitude และตำแหน่ง
3. **Preprocessing** — ลบค่าว่าง/ค่าผิดปกติ และแบ่งระดับความรุนแรงจากค่า `mag` ตามเกณฑ์ USGS:
   | ระดับ (Class) | ช่วง Magnitude |
   |---|---|
   | Minor | < 4.0 |
   | Light | 4.0 - 4.9 |
   | Moderate | 5.0 - 5.9 |
   | Strong | 6.0 - 6.9 |
   | Major | ≥ 7.0 |
4. **Transform Data** — Label Encoding (target), Train/Test split (80/20, stratified),
   Feature Scaling ด้วย `StandardScaler` (features: `latitude`, `longitude`, `depth`)
5. **Train & Compare Models** — เทรนและเปรียบเทียบ RandomForest, Logistic Regression, SVM
   แล้วเลือกโมเดลที่ดีที่สุดจากค่า **weighted F1-score** (เหมาะกับข้อมูลที่ไม่สมดุลระหว่าง class)
6. **Evaluate** — แสดง Accuracy, Classification Report, Confusion Matrix, Feature Importance
7. **Save Model** — บันทึกโมเดล/scaler/label encoder ด้วย `joblib` ลงในโฟลเดอร์ `model/`

## 🚀 วิธีใช้งาน

### 1. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2. เทรนโมเดล

```bash
python train_model.py
```

คำสั่งนี้จะสร้างไฟล์โมเดลในโฟลเดอร์ `model/` (ต้องรันก่อนเปิดเว็บแอปครั้งแรก)

### 3. รันเว็บแอป Streamlit

```bash
streamlit run app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่ `http://localhost:8501`

## 📤 การอัปโหลดขึ้น GitHub

```bash
cd earthquake-magnitude-prediction
git init
git add .
git commit -m "Initial commit: earthquake magnitude prediction model + Streamlit app"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
git push -u origin main
```

> แก้ `<YOUR-USERNAME>` และ `<YOUR-REPO-NAME>` เป็นบัญชี GitHub และชื่อ repository ของคุณ
> (ต้องสร้าง repository เปล่าบน GitHub ก่อน และมี Personal Access Token/SSH key สำหรับ push)

## ☁️ Deploy บน Streamlit Community Cloud (ฟรี)

1. Push โค้ดขึ้น GitHub ตามขั้นตอนด้านบน
2. เข้า [https://share.streamlit.io](https://share.streamlit.io) แล้ว Sign in ด้วย GitHub
3. กด **New app** → เลือก repository และ branch (`main`) → ตั้งค่า Main file path เป็น `app.py`
4. กด **Deploy** — ระบบจะติดตั้ง dependencies จาก `requirements.txt` และรันแอปให้อัตโนมัติ

## ⚠️ ข้อควรทราบ

- ข้อมูลในชุดนี้มีความไม่สมดุลระหว่าง class ค่อนข้างมาก (เหตุการณ์ระดับ `Light` มีจำนวนมากที่สุด
  ส่วน `Major` มีน้อยมาก) จึงใช้ `class_weight="balanced"` และ weighted F1-score ในการประเมิน/เลือกโมเดล
- โมเดลนี้จัดทำเพื่อการศึกษา ไม่ควรใช้แทนระบบเตือนภัยแผ่นดินไหวที่เป็นทางการ
- แหล่งข้อมูล: United States Geological Survey (USGS) Earthquake Catalog
