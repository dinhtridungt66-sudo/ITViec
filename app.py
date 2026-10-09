import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="Trung Tâm Tin Học – Đồ án TN Data Science",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #003366;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .recommend {
        background-color: #e6f4ea;
        color: #137333;
        border: 2px solid #34a853;
    }
    .not-recommend {
        background-color: #fce8e6;
        color: #c5221f;
        border: 2px solid #ea4335;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ĐƯỜNG DẪN MODEL ====================
_candidates = [Path("models"), Path("."), Path("/home/workdir/attachments")]
MODEL_DIR = next(
    (p for p in _candidates
     if (p / "svm_tuned_model.pkl").exists() or (p / "company_recommender.pkl").exists()),
    Path(".")
)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_classification_models():
    svm_model, threshold, hybrid_model, hybrid_error = None, 0.7, None, None
    try:
        with open(MODEL_DIR / "svm_tuned_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        with open(MODEL_DIR / "svm_threshold.pkl", "rb") as f:
            threshold = float(pickle.load(f))
    except Exception as e:
        st.sidebar.error(f"Không load được SVM: {e}")

    try:
        with open(MODEL_DIR / "hybrid_model.pkl", "rb") as f:
            hybrid_model = pickle.load(f)
    except Exception as e:
        hybrid_error = str(e)

    return svm_model, threshold, hybrid_model, hybrid_error


@st.cache_resource
def load_recommender():
    path = MODEL_DIR / "company_recommender.pkl"
    if not path.exists():
        return None, "Không tìm thấy company_recommender.pkl"
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data, None
    except Exception as e:
        return None, str(e)


svm_model, THRESHOLD, hybrid_model, hybrid_error = load_classification_models()
recommender, recommender_error = load_recommender()

# ==================== SIDEBAR ====================
st.sidebar.title("🎓 Trung Tâm Tin Học")
st.sidebar.markdown("---")

menu = [
    "Home",
    "Capstone Project",
    "Gợi ý công ty tương tự",
    "Recommend or Not"
]
choice = st.sidebar.selectbox("Menu", menu)

st.sidebar.markdown("---")
st.sidebar.caption("Đồ án tốt nghiệp Data Science\nPhòng LT & Mạng – CSC")

# ==================== 1. HOME ====================
if choice == "Home":
    st.markdown('<p class="main-header">Trung Tâm Tin Học</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Trường Đại học Khoa học Tự nhiên – ĐHQG TP.HCM</p>',
        unsafe_allow_html=True
    )

    st.subheader("[Trang chủ CSC](https://csc.edu.vn)")
    st.write("""
    Chào mừng bạn đến với hệ thống demo **Đồ án tốt nghiệp Data Science**.

    Hệ thống hỗ trợ **Topic 2**:
    - **Gợi ý công ty tương tự** (Content-Based Filtering)
    - **Recommend / Not Recommend** công ty từ review ứng viên
    """)

    st.markdown("---")
    st.markdown("### 👥 Thông tin nhóm")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Thành viên 1**  
        Họ và tên: *Phạm Trung Tín*  
        """)
    with col2:
        st.info("""
        **Thành viên 2**  
        Họ và tên: *Đinh Trí Dũng*  
        """)

    st.info("👈 Sử dụng menu bên trái để chuyển chức năng.")

# ==================== 2. CAPSTONE PROJECT ====================
elif choice == "Capstone Project":
    st.markdown('<p class="main-header">Đồ án TN Data Science</p>', unsafe_allow_html=True)
    st.subheader(
        "[Thông tin khóa học]"
        "(https://csc.edu.vn/data-science-machine-learning/do-an-tot-nghiep-data-science_310)"
    )

    st.write("""
    ### Chủ đề thực hiện
    **Content-Based Company Similarity Recommendation and 'Recommend or Not' Classification for Candidates**

    #### Yêu cầu 1 – Content-Based Filtering
    Gợi ý các công ty tương tự dựa trên mô tả công ty (TF-IDF + Cosine Similarity / Gensim).

    #### Yêu cầu 2 – Classification “Recommend or Not”
    Dự đoán khả năng Recommend công ty từ điểm đánh giá review (Linear SVM tuned + Hybrid).
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.success("**Recommendation**\n\nTF-IDF + Cosine / Gensim\n~468 công ty ITViec")
    with c2:
        st.success("**Classification**\n\nLinear SVM (threshold 0.7)\n+ Hybrid (optional)")

    st.caption("Dữ liệu: ITViec.com | Chỉ dùng cho mục đích học tập")

# ==================== 3. GỢI Ý CÔNG TY TƯƠNG TỰ (YÊU CẦU 1) ====================
elif choice == "Gợi ý công ty tương tự":
    st.markdown('<p class="main-header">1. Tìm kiếm công ty tương tự</p>', unsafe_allow_html=True)
    st.write("Content-Based Filtering – Gợi ý công ty dựa trên nội dung mô tả")

    if recommender is None:
        st.error(f"Không load được model Recommendation: {recommender_error}")
        st.info(
            "Hãy đặt file `company_recommender.pkl` cùng thư mục với app.py "
            "hoặc trong thư mục `models/`."
        )
        st.stop()

    companies = recommender["companies"].copy()
    final_model_name = recommender.get("final_model", "Cosine (sklearn)")
    sk = recommender["sklearn"]
    ge = recommender.get("gensim")

    st.write(f"##### Dữ liệu công ty ({len(companies)} công ty)")
    st.dataframe(
        companies[["Company Name", "Company industry", "Company Type", "Overall rating"]].head(10),
        use_container_width=True
    )

    engine_options = ["Cosine (sklearn)"]
    if ge is not None:
        engine_options.append("Gensim")

    engine = st.radio(
        "Chọn mô hình",
        engine_options,
        horizontal=True,
        help=f"Model được chọn khi train: **{final_model_name}**"
    )

    sim_matrix = sk["similarity"] if engine.startswith("Cosine") else ge["similarity"]
    vectorizer = sk["vectorizer"]
    tfidf_matrix = sk["tfidf_matrix"]

    tab1, tab2 = st.tabs(["Chọn công ty có sẵn", "Nhập mô tả công ty mới"])

    # ----- Tab 1 -----
    with tab1:
        company_names = companies["Company Name"].tolist()
        selected = st.selectbox("Chọn công ty", company_names)
        top_k = st.slider("Số lượng gợi ý (top-k)", 3, 10, 5)

        if st.button("Tìm công ty tương tự", type="primary", key="btn_exist"):
            idx = companies[companies["Company Name"] == selected].index[0]
            pos = (
                companies.index.get_loc(idx)
                if idx in companies.index
                else list(companies.index).index(idx)
            )

            scores = sim_matrix[pos].copy()
            scores[pos] = -1
            top_idx = np.argsort(scores)[::-1][:top_k]

            rows = []
            for i in top_idx:
                row = companies.iloc[i]
                rows.append({
                    "Công ty": row["Company Name"],
                    "Độ tương đồng": round(float(scores[i]), 4),
                    "Lĩnh vực": row.get("Company industry", "N/A"),
                    "Loại hình": row.get("Company Type", "N/A"),
                    "Rating": row.get("Overall rating", "N/A")
                })

            st.success(f"Top {top_k} công ty tương tự với **{selected}** ({engine})")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ----- Tab 2 -----
    with tab2:
        st.markdown(
            "Nhập mô tả công ty bằng **tiếng Anh** "
            "(model được train trên bản dịch tiếng Anh)."
        )
        new_text = st.text_area(
            "Mô tả công ty mới",
            height=120,
            placeholder=(
                "We are a fintech company building mobile banking apps, "
                "e-wallets and digital payment solutions..."
            )
        )
        top_k_new = st.slider("Số lượng gợi ý", 3, 10, 5, key="topk_new")

        if st.button("Gợi ý cho công ty mới", type="primary", key="btn_new"):
            if not new_text.strip():
                st.warning("Vui lòng nhập mô tả công ty.")
            else:
                if engine.startswith("Cosine"):
                    new_vec = vectorizer.transform([new_text.lower()])
                    scores = cosine_similarity(new_vec, tfidf_matrix).flatten()
                else:
                    tokens = new_text.lower().split()
                    bow = ge["dictionary"].doc2bow(tokens)
                    scores = np.asarray(
                        ge["similarity_index"][ge["tfidf_model"][bow]]
                    )

                top_idx = np.argsort(scores)[::-1][:top_k_new]
                rows = []
                for i in top_idx:
                    row = companies.iloc[i]
                    rows.append({
                        "Công ty": row["Company Name"],
                        "Độ tương đồng": round(float(scores[i]), 4),
                        "Lĩnh vực": row.get("Company industry", "N/A"),
                        "Loại hình": row.get("Company Type", "N/A"),
                        "Rating": row.get("Overall rating", "N/A")
                    })

                st.success(f"Kết quả gợi ý ({engine})")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ==================== 4. RECOMMEND OR NOT (YÊU CẦU 2) ====================
elif choice == "Recommend or Not":
    st.markdown('<p class="main-header">2. Recommend or Not</p>', unsafe_allow_html=True)
    st.write(
        "Dự đoán khả năng **Recommend / Not Recommend** công ty "
        "dựa trên điểm đánh giá của ứng viên"
    )

    if svm_model is None:
        st.error("Không load được model Classification (svm_tuned_model.pkl).")
        st.stop()

    available_models = ["Linear SVM (Tuned)"]
    if hybrid_model is not None:
        available_models.append("Hybrid (Numeric + Text)")

    model_choice = st.selectbox("Chọn mô hình", available_models)

    if hybrid_model is None:
        st.warning(
            "⚠️ Hybrid model không load được "
            "(thường do khác phiên bản scikit-learn). "
            "Hiện chỉ dùng được **Linear SVM**."
        )

    st.markdown("---")
    st.write("##### Nhập điểm đánh giá (1 → 5)")

    c1, c2 = st.columns(2)
    with c1:
        rating = st.radio("⭐ Rating tổng thể", [1, 2, 3, 4, 5], index=3, horizontal=True)
        salary_benefits = st.radio("💰 Salary & benefits", [1, 2, 3, 4, 5], index=2, horizontal=True)
        training_learning = st.radio("📚 Training & learning", [1, 2, 3, 4, 5], index=2, horizontal=True)
    with c2:
        management_cares = st.radio(
            "👔 Management cares about me", [1, 2, 3, 4, 5], index=2, horizontal=True
        )
        culture_fun = st.radio("🎉 Culture & fun", [1, 2, 3, 4, 5], index=2, horizontal=True)
        office_workspace = st.radio(
            "🏢 Office & workspace", [1, 2, 3, 4, 5], index=2, horizontal=True
        )

    review_length = st.number_input(
        "📝 Độ dài review (số từ)", min_value=5, max_value=1000, value=80, step=5
    )

    avg_sub = round(
        (salary_benefits + training_learning + management_cares + culture_fun + office_workspace) / 5,
        2
    )
    discrepancy = round(abs(rating - avg_sub), 2)

    m1, m2 = st.columns(2)
    m1.metric("Average Sub-Rating", avg_sub)
    m2.metric("Rating Discrepancy", discrepancy)

    review_text = ""
    if model_choice == "Hybrid (Numeric + Text)":
        review_text = st.text_area(
            "Nội dung review (Cleaned_Review_Text)",
            height=100,
            placeholder="great team, good salary, management needs improvement..."
        )

    st.markdown("---")
    submit = st.button("Submit", type="primary", use_container_width=True)

    if submit:
        features_numeric = pd.DataFrame([{
            "Rating": rating,
            "Salary & benefits": salary_benefits,
            "Training & learning": training_learning,
            "Management cares about me": management_cares,
            "Culture & fun": culture_fun,
            "Office & workspace": office_workspace,
            "Average_Sub_Rating": avg_sub,
            "Rating_Discrepancy": discrepancy,
            "Review_Length": review_length
        }])

        if model_choice == "Linear SVM (Tuned)":
            proba = svm_model.predict_proba(features_numeric)[0]
            prob_recommend = float(proba[1])
            pred = 1 if prob_recommend >= THRESHOLD else 0
            model_name = f"Linear SVM (threshold = {THRESHOLD})"
        else:
            if not review_text.strip():
                st.warning("Model Hybrid cần nội dung review.")
                st.stop()
            features_hybrid = features_numeric.copy()
            features_hybrid["Cleaned_Review_Text"] = review_text
            proba = hybrid_model.predict_proba(features_hybrid)[0]
            prob_recommend = float(proba[1])
            pred = int(hybrid_model.predict(features_hybrid)[0])
            model_name = "Hybrid (Numeric + TF-IDF)"

        # ----- Kết quả -----
        st.write("### Kết quả dự đoán")
        col_a, col_b = st.columns(2)

        with col_a:
            if pred == 1:
                st.markdown(
                    '<div class="result-box recommend">'
                    '✅ RECOMMEND<br>'
                    '<small>Nên giới thiệu công ty này</small>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-box not-recommend">'
                    '❌ NOT RECOMMEND<br>'
                    '<small>Không nên giới thiệu công ty này</small>'
                    '</div>',
                    unsafe_allow_html=True
                )

        with col_b:
            st.metric("Xác suất Recommend", f"{prob_recommend * 100:.1f}%")
            st.metric("Xác suất Not Recommend", f"{(1 - prob_recommend) * 100:.1f}%")
            st.caption(f"Model: {model_name}")

        st.progress(prob_recommend, text=f"Recommend probability: {prob_recommend:.3f}")

        # ----- Bảng tóm tắt (thay biểu đồ cột) -----
        st.write("### Tóm tắt đánh giá đã nhập")
        summary = pd.DataFrame({
            "Tiêu chí": [
                "Rating tổng thể",
                "Salary & benefits",
                "Training & learning",
                "Management cares about me",
                "Culture & fun",
                "Office & workspace",
                "Average Sub-Rating",
                "Rating Discrepancy",
                "Review Length"
            ],
            "Giá trị": [
                rating,
                salary_benefits,
                training_learning,
                management_cares,
                culture_fun,
                office_workspace,
                avg_sub,
                discrepancy,
                review_length
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if model_choice == "Linear SVM (Tuned)":
            st.caption(
                f"Model SVM dùng ngưỡng tối ưu **{THRESHOLD}**: "
                f"nếu P(Recommend) ≥ {THRESHOLD} → **Recommend**, "
                f"ngược lại → **Not Recommend**."
            )

        with st.expander("Xem input đưa vào model"):
            st.dataframe(features_numeric, use_container_width=True, hide_index=True)
            if model_choice.startswith("Hybrid"):
                st.write("Review text:", review_text)

# ==================== FOOTER ====================
st.markdown("---")
st.caption(
    "Trung Tâm Tin Học – Trường ĐH Khoa học Tự nhiên TP.HCM | "
    "Đồ án tốt nghiệp Data Science 2026"
)