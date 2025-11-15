import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY,
            name TEXT,
            tagline TEXT,
            about TEXT,
            github TEXT,
            twitter TEXT,
            linkedin TEXT,
            website TEXT
        )
    """)
    try:
        cur.execute("ALTER TABLE profile ADD COLUMN avatar_path TEXT")
    except Exception:
        pass
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            image_path TEXT,
            link TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experience (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            company TEXT,
            start_date TEXT,
            end_date TEXT,
            description TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY,
            knowledge_pdf_path TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            exploring INTEGER DEFAULT 0
        )
    """)
    cur.execute("SELECT COUNT(*) AS c FROM profile")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO profile (id, name, tagline, about, github, twitter, linkedin, website) VALUES (1, '', '', '', '', '', '', '')")
    cur.execute("SELECT COUNT(*) AS c FROM settings")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO settings (id, knowledge_pdf_path) VALUES (1, '')")
    conn.commit()
    conn.close()

class RAGEngine:
    def __init__(self):
        self.vectorizer = None
        self.chunks = []
        self.matrix = None
        self.ready = False

    def build_from_pdf(self, path):
        self.chunks = []
        text = ""
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                content = page.extract_text() or ""
                text += content + "\n\n"
        except Exception:
            self.ready = False
            return False
        units = [u.strip() for u in text.split("\n\n") if u.strip()]
        buf = ""
        for u in units:
            if len(buf) + len(u) < 800:
                buf = (buf + "\n" + u).strip()
            else:
                if buf:
                    self.chunks.append(buf)
                buf = u
        if buf:
            self.chunks.append(buf)
        if not self.chunks:
            self.ready = False
            return False
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self.ready = True
        return True

    def answer(self, question, k=3):
        if not self.ready:
            return "Knowledge base is not ready. Upload a PDF in admin."
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        idxs = sims.argsort()[::-1][:k]
        parts = [self.chunks[i] for i in idxs if sims[i] > 0]
        if not parts:
            return "I could not find relevant information in the document."
        joined = "\n\n".join(parts)
        return joined[:1500]

rag = RAGEngine()

def require_login():
    return session.get("logged_in") is True

@app.route("/")
def index():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM profile WHERE id=1")
    profile = cur.fetchone()
    cur.execute("SELECT * FROM projects ORDER BY datetime(created_at) DESC")
    projects = cur.fetchall()
    cur.execute("SELECT * FROM experience ORDER BY id DESC")
    experience = cur.fetchall()
    cur.execute("SELECT * FROM skills ORDER BY exploring ASC, name ASC")
    skills = cur.fetchall()
    cur.execute("SELECT knowledge_pdf_path FROM settings WHERE id=1")
    knowledge_pdf_path = cur.fetchone()[0]
    conn.close()
    return render_template("index.html", profile=profile, projects=projects, experience=experience, skills=skills, has_kb=bool(knowledge_pdf_path))

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html", error="Invalid password")
    if require_login():
        return redirect(url_for("admin_dashboard"))
    return render_template("admin_login.html", error=None)

@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))

@app.route("/admin")
def admin_dashboard():
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM profile WHERE id=1")
    profile = cur.fetchone()
    cur.execute("SELECT * FROM projects ORDER BY datetime(created_at) DESC")
    projects = cur.fetchall()
    cur.execute("SELECT * FROM experience ORDER BY id DESC")
    experience = cur.fetchall()
    cur.execute("SELECT knowledge_pdf_path FROM settings WHERE id=1")
    knowledge_pdf_path = cur.fetchone()[0]
    conn.close()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM skills ORDER BY exploring ASC, name ASC")
    skills = cur.fetchall()
    conn.close()
    return render_template("admin_dashboard.html", profile=profile, projects=projects, experience=experience, knowledge_pdf_path=knowledge_pdf_path, skills=skills)

@app.route("/admin/profile", methods=["POST"])
def admin_profile():
    if not require_login():
        return redirect(url_for("admin_login"))
    name = request.form.get("name", "")
    tagline = request.form.get("tagline", "")
    about = request.form.get("about", "")
    github = request.form.get("github", "")
    twitter = request.form.get("twitter", "")
    linkedin = request.form.get("linkedin", "")
    website = request.form.get("website", "")
    avatar = request.files.get("avatar")
    avatar_path = None
    if avatar and avatar.filename:
        filename = secure_filename(avatar.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        avatar.save(save_path)
        avatar_path = "uploads/" + filename.replace("\\", "/")
    conn = get_db()
    cur = conn.cursor()
    if avatar_path:
        cur.execute("UPDATE profile SET name=?, tagline=?, about=?, github=?, twitter=?, linkedin=?, website=?, avatar_path=? WHERE id=1", (name, tagline, about, github, twitter, linkedin, website, avatar_path))
    else:
        cur.execute("UPDATE profile SET name=?, tagline=?, about=?, github=?, twitter=?, linkedin=?, website=? WHERE id=1", (name, tagline, about, github, twitter, linkedin, website))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/projects/add", methods=["POST"])
def admin_projects_add():
    if not require_login():
        return redirect(url_for("admin_login"))
    title = request.form.get("title", "")
    description = request.form.get("description", "")
    link = request.form.get("link", "")
    image = request.files.get("image")
    image_path = ""
    if image and image.filename:
        filename = secure_filename(image.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(save_path)
        image_path = os.path.join("uploads", filename)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO projects (title, description, image_path, link, created_at) VALUES (?, ?, ?, ?, ?)", (title, description, image_path, link, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/projects/delete/<int:pid>", methods=["POST"])
def admin_projects_delete(pid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM projects WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/experience/add", methods=["POST"])
def admin_experience_add():
    if not require_login():
        return redirect(url_for("admin_login"))
    role = request.form.get("role", "")
    company = request.form.get("company", "")
    start_date = request.form.get("start_date", "")
    end_date = request.form.get("end_date", "")
    description = request.form.get("description", "")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO experience (role, company, start_date, end_date, description) VALUES (?, ?, ?, ?, ?)", (role, company, start_date, end_date, description))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/experience/delete/<int:eid>", methods=["POST"])
def admin_experience_delete(eid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM experience WHERE id=?", (eid,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/skills/add", methods=["POST"])
def admin_skills_add():
    if not require_login():
        return redirect(url_for("admin_login"))
    name = request.form.get("name", "").strip()
    exploring = 1 if request.form.get("exploring") == "on" else 0
    if name:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO skills (name, exploring) VALUES (?, ?)", (name, exploring))
        conn.commit()
        conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/skills/delete/<int:sid>", methods=["POST"])
def admin_skills_delete(sid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM skills WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/knowledge", methods=["POST"])
def admin_knowledge():
    if not require_login():
        return redirect(url_for("admin_login"))
    pdf = request.files.get("pdf")
    if pdf and pdf.filename:
        filename = secure_filename(pdf.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        pdf.save(save_path)
        rel_path = os.path.join("static", "uploads", filename)
        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE settings SET knowledge_pdf_path=? WHERE id=1", (rel_path,))
        conn.commit()
        conn.close()
        rag.build_from_pdf(os.path.join(BASE_DIR, rel_path))
    return redirect(url_for("admin_dashboard"))

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a question."})
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT knowledge_pdf_path FROM settings WHERE id=1")
    knowledge_pdf_path = cur.fetchone()[0]
    conn.close()
    if knowledge_pdf_path and not rag.ready:
        rag.build_from_pdf(os.path.join(BASE_DIR, knowledge_pdf_path))
    answer = rag.answer(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    init_db()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT knowledge_pdf_path FROM settings WHERE id=1")
    p = cur.fetchone()[0]
    conn.close()
    if p:
        rag.build_from_pdf(os.path.join(BASE_DIR, p))
    app.run(host="127.0.0.1", port=5000, debug=True)