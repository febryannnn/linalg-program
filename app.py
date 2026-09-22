import io
import contextlib

import numpy as np
import pandas as pd
import streamlit as st


# SPL dengan Gauss-Jordan

def print_iterasi_spl(iter_no, matriks):
    print(f"Iterasi {iter_no}:")
    for row in matriks:
        print("  ".join(f"{val:8.2f}" for val in row))
    print("\n")


def gauss_jordan(matriks, decimals=2):
    matriks = matriks.astype(float).copy()
    n, m = matriks.shape
    iter_no = 0

    for col in range(n):
        if abs(matriks[col, col]) < 1e-12:
            for r in range(col + 1, n):
                if abs(matriks[r, col]) > 1e-12:
                    matriks[[col, r]] = matriks[[r, col]]
                    break

        pivot = matriks[col, col]
        if abs(pivot) < 1e-12:
            continue

        matriks[col, :] = np.round(matriks[col, :] / pivot, decimals=decimals)
        iter_no += 1
        print_iterasi_spl(iter_no, matriks)

        for r in range(col + 1, n):
            if abs(matriks[r, col]) > 1e-12:
                factor = matriks[r, col]
                matriks[r, :] = np.round(matriks[r, :] - factor * matriks[col, :], decimals=decimals)
                iter_no += 1
                print_iterasi_spl(iter_no, matriks)

    for col in range(n - 1, -1, -1):
        for r in range(col - 1, -1, -1):
            if abs(matriks[r, col]) > 1e-12:
                factor = matriks[r, col]
                matriks[r, :] = np.round(matriks[r, :] - factor * matriks[col, :], decimals=decimals)
                iter_no += 1
                print_iterasi_spl(iter_no, matriks)

    return matriks


# Invers Matriks dengan Gauss-Jordan

def print_mat(mat):
    rows, cols = mat.shape
    for r in range(rows):
        print("  ".join(f"{mat[r, c]:6.2f}" for c in range(cols)))
    print()


def find_invers(matrix):
    n = len(matrix)
    A = np.array(matrix, dtype=float)
    I = np.identity(n)
    aug = np.concatenate((A, I), axis=1)
    aug = np.round(aug, 2)

    print(f"n: {n}")

    iter_no = 0
    print("Matriks Awal:")
    print_mat(aug)
    print('\n')

    for i in range(n):
        iter_no += 1
        pivot = aug[i, i]
        aug[i] = np.round(aug[i] / pivot, 2)
        print(f"Iterasi {iter_no} (baris {i+1} = baris {i+1} / {pivot}):")
        print_mat(aug)

        for j in range(i + 1, n):
            factor = aug[j, i]
            if factor != 0:
                iter_no += 1
                aug[j] = np.round(aug[j] - factor * aug[i], 2)
                print(f"Iterasi {iter_no} (baris {j+1} = baris {i+1} * {-1 * factor:.2f} + baris {j+1}:")
                print_mat(aug)

    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = aug[j, i]
            if factor != 0:
                iter_no += 1
                aug[j] = np.round(aug[j] - factor * aug[i], 2)
                print(f"Iterasi {iter_no} (baris {j+1} = baris {i+1} * {-1 * factor:.2f} + baris {j+1}:")
                print_mat(aug)

    print('\n')
    print("Hasil akhir Matriks Augmented:")
    print_mat(aug)
    inv = aug[:, n:]
    print("Hasil Invers Matriks:")
    print_mat(inv)
    return inv

def run_and_capture(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


# UI Streamlit

st.set_page_config(page_title="Program Alin", layout="wide")
st.title("Program Aljabar Linear")
st.caption(
    "Aljabar Linier 2026"
)

tab_spl, tab_inv = st.tabs(["Sistem Persamaan Linear (SPL)", "Invers Matriks"])

# TAB 1: SPL
with tab_spl:
    st.subheader("Penyelesaian SPL dengan Gauss-Jordan")
    st.write(
        "Tentukan ukuran matriks (baris x kolom), lalu isi nilainya di tabel. "
        "Baris = banyaknya persamaan/pivot, kolom = koefisien variabel ditambah "
        "kolom konstanta (dan kolom tambahan lain kalau ada)."
    )

    c1, c2 = st.columns(2)
    with c1:
        n_rows_spl = st.number_input("Jumlah baris (rows)", min_value=1, max_value=10, value=3, step=1, key="n_rows_spl")
    with c2:
        n_cols_spl = st.number_input("Jumlah kolom (columns)", min_value=2, max_value=12, value=4, step=1, key="n_cols_spl")

    if n_cols_spl < n_rows_spl:
        st.error("Jumlah kolom harus lebih besar atau sama dengan jumlah baris.")
    else:
        default_spl = np.array([
            [2, 0, 6, 40],
            [0, 4, -1, 10],
            [2, -1, 2, 12],
        ], dtype=float)

        if default_spl.shape != (n_rows_spl, n_cols_spl):
            default_spl = np.zeros((n_rows_spl, n_cols_spl))
            np.fill_diagonal(default_spl, 1)

        col_labels_spl = [f"k{i+1}" for i in range(n_cols_spl)]
        df_spl = pd.DataFrame(default_spl, columns=col_labels_spl)
        edited_spl = st.data_editor(df_spl, key="editor_spl", use_container_width=True, num_rows="fixed")

        tampilkan_iterasi_spl = st.checkbox("Tampilkan langkah iterasi", value=True, key="show_iter_spl")

        if st.button("Selesaikan SPL", type="primary", key="btn_spl"):
            matriks_input = edited_spl.to_numpy(dtype=float)

            if matriks_input.shape != (n_rows_spl, n_cols_spl):
                st.error("Ukuran matriks tidak sesuai dengan jumlah baris/kolom yang dipilih.")
            else:
                hasil, log_text = run_and_capture(gauss_jordan, matriks_input, 2)

                if tampilkan_iterasi_spl:
                    st.text("Log iterasi (identik dengan output notebook asli):")
                    st.code(log_text, language="text")

                st.text("Matriks akhir:")
                st.dataframe(pd.DataFrame(hasil, columns=col_labels_spl), use_container_width=True)

                submatrix = hasil[:, :n_rows_spl]
                if np.allclose(submatrix, np.eye(n_rows_spl), atol=1e-2):
                    st.success("Solusi ditemukan:")
                    extra = hasil[:, n_rows_spl:]
                    if extra.shape[1] == 1:
                        for i in range(n_rows_spl):
                            st.write(f"x{i+1} = {extra[i, 0]:.2f}")
                    else:
                        extra_cols = [f"hasil{i+1}" for i in range(extra.shape[1])]
                        st.dataframe(pd.DataFrame(extra, columns=extra_cols), use_container_width=True)
                else:
                    st.warning(
                        "Bagian kiri matriks belum tereduksi penuh menjadi matriks identitas. "
                        "Kemungkinan ada pivot nol yang tidak bisa ditukar, sistem punya "
                        "banyak solusi, atau tidak punya solusi sama sekali."
                    )

# TAB 2: INVERS MATRIKS
with tab_inv:
    st.subheader("Mencari Invers Matriks dengan Gauss-Jordan")
    st.write("Masukkan matriks persegi (3 x 3) yang akan dicari inversnya.")

    n_inv = 3
    # st.number_input("Ukuran matriks (n x n)", min_value=2, max_value=8, value=3, step=1, key="n_inv")

    default_inv = np.array([
        [9, -4, 7],
        [6, 8, -6],
        [-5, 3, 7],
    ], dtype=float)

    if default_inv.shape[0] != n_inv:
        default_inv = np.eye(n_inv)

    cols_inv = [f"k{i+1}" for i in range(n_inv)]
    df_inv = pd.DataFrame(default_inv, columns=cols_inv)
    edited_inv = st.data_editor(df_inv, key="editor_inv", use_container_width=True, num_rows="fixed")

    tampilkan_iterasi_inv = st.checkbox("Tampilkan langkah iterasi", value=True, key="show_iter_inv")

    if st.button("Cari Invers", type="primary", key="btn_inv"):
        matriks_a = edited_inv.to_numpy(dtype=float)

        if matriks_a.shape != (n_inv, n_inv):
            st.error("Matriks harus berukuran persegi (n x n).")
        elif np.any(np.diag(matriks_a) == 0):
            st.warning(
                "Ada elemen diagonal utama bernilai 0. Algoritma asli (seperti di "
                "Invers.ipynb) tidak melakukan penukaran baris otomatis, sehingga "
                "hasilnya bisa tidak valid. Susun ulang urutan baris matriks Anda "
                "agar diagonal utama tidak nol, lalu coba lagi."
            )
        else:
            try:
                inv, log_text = run_and_capture(find_invers, matriks_a.tolist())

                if tampilkan_iterasi_inv:
                    st.text("Log iterasi (identik dengan output notebook asli):")
                    st.code(log_text, language="text")

                st.success("Hasil invers matriks:")
                st.dataframe(pd.DataFrame(inv, columns=cols_inv), use_container_width=True)

                verifikasi = np.round(matriks_a @ inv, 2)
                st.text("Verifikasi (A x A^-1 = I):")
                st.dataframe(pd.DataFrame(verifikasi, columns=cols_inv), use_container_width=True)
            except ZeroDivisionError:
                st.error(
                    "Terjadi pembagian dengan nol saat eliminasi, matriks mungkin "
                    "singular atau butuh penukaran baris."
                )