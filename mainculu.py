import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt # Vẫn cần cho background_gradient

# ==============================================================================
# 1. Hàm Phân tích Điểm Bất Thường
# ==============================================================================

def analyze_outliers(df_grades, subject_cols, threshold=2.0):
    """
    Thực hiện tính toán số lượng học sinh có điểm bất thường theo Z-score 
    cho từng môn học ở mỗi lớp.
    """
    df_grades.dropna(subset=['lop'], inplace=True)
    
    for col in subject_cols:
        df_grades[col] = pd.to_numeric(df_grades[col], errors='coerce')

    unique_classes = df_grades['lop'].unique()
    outlier_counts = pd.DataFrame(index=unique_classes, columns=subject_cols)
    
    for lop in unique_classes:
        df_class = df_grades[df_grades['lop'] == lop]
        
        for subject in subject_cols:
            scores = df_class[subject].dropna()
            
            if len(scores) >= 2:
                z_scores = zscore(scores)
                num_outliers = (np.abs(z_scores) > threshold).sum()
                outlier_counts.loc[lop, subject] = num_outliers
            else:
                outlier_counts.loc[lop, subject] = 0

    outlier_counts = outlier_counts.fillna(0).astype(int)
    outlier_counts['Tổng Outliers Lớp'] = outlier_counts.sum(axis=1)
    
    total_outliers_by_subject = outlier_counts.drop('Tổng Outliers Lớp', axis=1).sum(axis=0)
    
    return outlier_counts, total_outliers_by_subject

# ==============================================================================
# 2. Thiết lập Giao diện Streamlit
# ==============================================================================

st.set_page_config(layout="wide", page_title="Phân Tích Điểm Bất Thường Z-score (Biểu đồ Thanh Chồng)")

st.title("📊 Ứng Dụng Phân Tích Điểm Bất Thường theo Z-score")
st.markdown("Sử dụng **Biểu đồ Thanh Chồng** để trực quan hóa số lượng học sinh bất thường.")
st.markdown("---")

# ----------------- Tải File -----------------
st.header("1. Tải File Dữ liệu")
st.caption("Vui lòng tải lên file **Student_GradeSummary.csv**.")
uploaded_file = st.file_uploader(
    "Chọn file CSV của bạn:", 
    type=['csv']
)

if uploaded_file is not None:
    try:
        df_grades = pd.read_csv(uploaded_file)
        st.success(f"Đã tải thành công file: **{uploaded_file.name}**")
        
        # ----------------- Thiết lập Tham số -----------------
        st.header("2. Thiết lập Tham số Phân tích")

        col1, col2 = st.columns([1, 3])
        
        with col1:
            z_score_threshold = st.slider(
                "Chọn Ngưỡng Z-score ($\left| Z \\right| >$):", 
                min_value=1.5, 
                max_value=3.5, 
                value=2.0, 
                step=0.1,
                help="Điểm số được coi là bất thường nếu độ lệch chuẩn của nó so với trung bình lớp lớn hơn ngưỡng này."
            )
        
        with col2:
            st.info(f"Phân tích sẽ tìm kiếm học sinh có điểm bất thường khi **$\left| Z \\right| > {z_score_threshold}$**.")

        # ----------------- Thực hiện Phân tích -----------------
        
        subject_cols = ['Toan', 'Van', 'Ly', 'Hoa', 'Ngoaingu', 'Su', 'Tin', 'Sinh', 'Dia']
        missing_cols = [col for col in ['lop'] + subject_cols if col not in df_grades.columns]
        
        if missing_cols:
            st.error(f"Lỗi: File CSV thiếu các cột cần thiết: **{', '.join(missing_cols)}**. Vui lòng kiểm tra lại cấu trúc file.")
        else:
            st.header("3. Kết Quả Phân Tích")
            
            outlier_counts_df_full, total_outliers_by_subject = analyze_outliers(
                df_grades.copy(), 
                subject_cols, 
                z_score_threshold
            )
            
            # Tách bảng chính (loại bỏ hàng tổng)
            outlier_counts_df = outlier_counts_df_full.iloc[:-1, :]
            
            # ----------------- Hiển thị Bảng Dữ liệu -----------------
            st.subheader("Bảng Tổng Kết Số Học Sinh Có Điểm Bất Thường")
            st.markdown(f"*(Đơn vị: Số học sinh - Ngưỡng $\left| Z \\right| > {z_score_threshold}$)*")
            
            styled_df = outlier_counts_df.style.background_gradient(
                cmap='Blues', 
                subset=subject_cols
            ).background_gradient(
                cmap='Reds', 
                subset=['Tổng Outliers Lớp']
            )

            st.dataframe(styled_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Tổng Số Điểm Bất Thường theo Môn Học")
            total_row_df = pd.DataFrame([total_outliers_by_subject]).rename(index={0: 'Tổng Outliers Môn'})
            
            st.dataframe(
                total_row_df.style.background_gradient(cmap='YlOrRd'), 
                use_container_width=True
            )
            st.markdown("---")


            # ----------------- Trực quan hóa Biểu đồ Thanh Chồng -----------------
            st.subheader("Biểu Đồ Thanh Chồng (Stacked Bar Chart)")
            st.caption("Mỗi thanh thể hiện tổng số Outlier của một lớp, phân chia theo từng môn học.")
            
            # Chuẩn bị dữ liệu cho biểu đồ (Unpivot data)
            plot_data_stacked = outlier_counts_df.drop(columns=['Tổng Outliers Lớp']).reset_index().melt(
                id_vars='index', var_name='Môn Học', value_name='Số HS Outlier'
            ).rename(columns={'index': 'Lớp'})

            # Tạo biểu đồ Thanh Chồng bằng Plotly Express
            fig_stacked_bar = px.bar(
                plot_data_stacked, 
                x="Lớp", 
                y="Số HS Outlier",
                color="Môn Học",
                title=f"Phân Bổ Điểm Bất Thường ($\left| Z \\right| > {z_score_threshold}$) Theo Lớp và Môn Học",
                labels={"Lớp": "Lớp Học", "Số HS Outlier": "Số Học Sinh Bất Thường"},
                height=600
            )
            
            # Điều chỉnh layout để tăng tính thẩm mỹ
            fig_stacked_bar.update_layout(
                xaxis_title="Lớp Học",
                yaxis_title="Tổng Số Học Sinh Bất Thường",
                legend_title="Môn Học",
                xaxis={'categoryorder': 'category ascending'} # Sắp xếp lớp theo thứ tự
            )
            
            st.plotly_chart(fig_stacked_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý file. Vui lòng kiểm tra định dạng file và tên cột: {e}")