from __future__ import annotations

import bcrypt
import streamlit as st

from app.database import SessionLocal, User, init_db


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def ensure_admin_user() -> None:
    init_db()
    session = SessionLocal()
    try:
        existing = session.query(User).filter(User.username == "admin").first()
        if not existing:
            session.add(User(username="admin", password_hash=hash_password("admin123"), role="admin"))
            session.commit()
    finally:
        session.close()


def login_form() -> bool:
    ensure_admin_user()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.markdown('<div class="ffd-auth-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="ffd-auth-card">', unsafe_allow_html=True)
    st.subheader("Secure Access")
    st.caption("Sign in to monitor transactions, predict fraud risk, and review alerts.")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True, key="login_btn"):
            session = SessionLocal()
            try:
                user = session.query(User).filter(User.username == username).first()
                if user and verify_password(password, user.password_hash):
                    st.session_state.authenticated = True
                    st.session_state.username = user.username
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            finally:
                session.close()

    with signup_tab:
        st.caption("Create a new account for transaction monitoring.")
        su_username = st.text_input("Choose a username", key="su_username")
        su_password = st.text_input("Choose a password", type="password", key="su_password")
        if st.button("Create Account", use_container_width=True, key="su_btn"):
            if not su_username or len(su_username) < 3:
                st.error("Username must be at least 3 characters.")
                return False
            if not su_password or len(su_password) < 6:
                st.error("Password must be at least 6 characters.")
                return False

            session = SessionLocal()
            try:
                existing = session.query(User).filter(User.username == su_username).first()
                if existing:
                    st.error("Username already exists. Please choose another.")
                    return False

                session.add(
                    User(
                        username=su_username,
                        password_hash=hash_password(su_password),
                        role="user",
                    )
                )
                session.commit()
                st.success("Account created. You can now login.")
            finally:
                session.close()
    st.markdown("</div></div>", unsafe_allow_html=True)
    return False

