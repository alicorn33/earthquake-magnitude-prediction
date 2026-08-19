# -*- coding: utf-8 -*-
"""
app.py
=======
เว็บแอปพลิเคชัน Streamlit สำหรับทำนายระดับความรุนแรงของแผ่นดินไหว
(Earthquake Magnitude Level Prediction)
จากตำแหน่งพื้นที่ (ละติจูด/ลองจิจูด) และความลึก (depth)
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ทำนายระดับความรุนแรงแผ่นดินไหว",
    page_icon="🌏",
    layout="centered",
)

MODEL_DIR = "model"

CLASS_INFO = {
    "Minor":    {"emoji": "🟢", "desc": "เบา (Magnitude < 4.0) — มักไม่รู้สึกหรือรู้สึกได้เล็กน้อย"},
    "Light":    {"emoji": "🟡", "desc": "ค่อนข้างเบา (4.0 - 4.9) — รู้สึกได้ชัดเจน ความเสียหายน้อยมาก"},
    "Moderate": {"emoji": "🟠", "desc": "ปานกลาง (5.0 - 5.9) — อาจสร้างความเสียหายต่อสิ่งก่อสร้างที่ไม่แข็งแรง"},
    "Strong":   {"emoji": "🔴", "desc": "รุนแรง (6.0 - 6.9) — สร้างความเสียหายในพื้นที่ที่มีประชากรหนาแน่น"},
    "Major":    {"emoji": "🟣", "desc": "รุนแรงมาก (≥ 7.0) — ก่อความเสียหายรุนแรงในวงกว้าง"},
}


# ---------------------------------------------------------------------------
# Load model artifacts (cache เพื่อไม่ต้องโหลดซ้ำทุกครั้งที่ interact)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    return model, scaler, label_encoder, feature_cols


try:
    model, scaler, label_encoder, feature_cols = load_artifacts()
except FileNotFoundError:
    st.error(
        "ไม่พบไฟล์โมเดล กรุณารัน `python train_model.py` ก่อน "
        "เพื่อสร้างไฟล์ในโฟลเดอร์ `model/`"
    )
    st.stop()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌏 ทำนายระดับความรุนแรงของแผ่นดินไหว")
st.markdown(
    "กรอกตำแหน่งพื้นที่ (ละติจูด, ลองจิจูด) และความลึกของจุดศูนย์กลางแผ่นดินไหว "
    "เพื่อทำนายระดับความรุนแรง (Magnitude Level) ด้วยโมเดล Machine Learning "
    "ที่เทรนจากข้อมูลแผ่นดินไหวในภูมิภาคใกล้ประเทศไทย"
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input(
        "ละติจูด (Latitude)", min_value=-90.0, max_value=90.0,
        value=13.0, step=0.01, format="%.4f",
        help="ตัวอย่าง: ประเทศไทยอยู่ราว 5 - 22 องศาเหนือ"
    )
with col2:
    longitude = st.number_input(
        "ลองจิจูด (Longitude)", min_value=-180.0, max_value=180.0,
        value=100.0, step=0.01, format="%.4f",
        help="ตัวอย่าง: ประเทศไทยอยู่ราว 92 - 106 องศาตะวันออก"
    )

depth = st.slider(
    "ความลึก (Depth, กิโลเมตร)",
    min_value=0.0, max_value=300.0, value=10.0, step=0.5,
    help="ความลึกของจุดศูนย์กลางแผ่นดินไหวใต้ผิวโลก"
)

st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=3)

predict_clicked = st.button("🔮 ทำนายระดับความรุนแรง", type="primary", use_container_width=True)

st.divider()

if predict_clicked:
    X_input = np.array([[latitude, longitude, depth]])
    X_scaled = scaler.transform(X_input)

    pred_encoded = model.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    info = CLASS_INFO.get(pred_label, {"emoji": "❓", "desc": ""})

    st.subheader("ผลการทำนาย")
    st.markdown(f"## {info['emoji']} {pred_label}")
    st.write(info["desc"])

    # แสดงความน่าจะเป็นของแต่ละ class ถ้าโมเดลรองรับ predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        proba_df = pd.DataFrame({
            "ระดับความรุนแรง": label_encoder.inverse_transform(np.arange(len(proba))),
            "ความน่าจะเป็น": proba,
        }).sort_values("ความน่าจะเป็น", ascending=False)

        st.markdown("**ความน่าจะเป็นของแต่ละระดับ:**")
        st.bar_chart(proba_df.set_index("ระดับความรุนแรง"))
        st.dataframe(
            proba_df.style.format({"ความน่าจะเป็น": "{:.2%}"}),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "⚠️ ผลการทำนายนี้มาจากโมเดล Machine Learning ที่เทรนจากข้อมูลในอดีต "
        "ใช้เพื่อการศึกษาเท่านั้น ไม่สามารถใช้แทนการเตือนภัยแผ่นดินไหวอย่างเป็นทางการได้"
    )

with st.expander("ℹ️ เกี่ยวกับโมเดลนี้"):
    st.markdown(
        f"""
        - **Feature ที่ใช้ทำนาย:** {', '.join(feature_cols)}
        - **ประเภทโมเดล:** {type(model).__name__}
        - **ระดับความรุนแรง (class):** {', '.join(label_encoder.classes_)}
        - ข้อมูลที่ใช้เทรน: เหตุการณ์แผ่นดินไหวในภูมิภาคประเทศไทยและใกล้เคียง
        """
    )
