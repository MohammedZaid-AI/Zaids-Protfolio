import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
# Ensure token is set
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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
            website TEXT,
            avatar_path TEXT,
            avatar_back_path TEXT
        )
    """)
    try:
        cur.execute("ALTER TABLE profile ADD COLUMN avatar_path TEXT")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE profile ADD COLUMN avatar_back_path TEXT")
    except Exception:
        pass
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            image_path TEXT,
            link TEXT,
            github_link TEXT,
            created_at TEXT
        )
    """)
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN github_link TEXT")
    except Exception:
        pass
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experience (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            company TEXT,
            start_date TEXT,
            end_date TEXT,
            description TEXT,
            logo_path TEXT
        )
    """)
    try:
        cur.execute("ALTER TABLE experience ADD COLUMN logo_path TEXT")
    except Exception:
        pass
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
        cur.execute("INSERT INTO profile (id, name, tagline, about, github, twitter, linkedin, website, avatar_path, avatar_back_path) VALUES (1, '', '', '', '', '', '', '', '', '')")
    cur.execute("SELECT COUNT(*) AS c FROM settings")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO settings (id, knowledge_pdf_path) VALUES (1, '')")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            message TEXT,
            timestamp TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS research_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            publication TEXT,
            date TEXT,
            link TEXT,
            description TEXT
        )
    """)
    conn.commit()
    conn.close()

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
    cur.execute("SELECT * FROM research_papers ORDER BY date DESC")
    research_papers = cur.fetchall()
    conn.close()
    return render_template("index.html", profile=profile, projects=projects, experience=experience, skills=skills, research_papers=research_papers)

@app.route("/contact", methods=["POST"])
def contact():
    name = request.form.get("name", "")
    email = request.form.get("email", "")
    message = request.form.get("message", "")
    if name and message:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO messages (name, email, message, timestamp) VALUES (?, ?, ?, ?)", (name, email, message, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    return redirect(url_for("index"))

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
    cur.execute("SELECT * FROM skills ORDER BY exploring ASC, name ASC")
    skills = cur.fetchall()
    cur.execute("SELECT * FROM messages ORDER BY datetime(timestamp) DESC")
    messages = cur.fetchall()
    cur.execute("SELECT * FROM research_papers ORDER BY date DESC")
    research_papers = cur.fetchall()
    conn.close()
    return render_template("admin_dashboard.html", profile=profile, projects=projects, experience=experience, skills=skills, messages=messages, research_papers=research_papers)

@app.route("/admin/messages/delete/<int:mid>", methods=["POST"])
def admin_messages_delete(mid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE id=?", (mid,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))



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
    
    conn = get_db()
    cur = conn.cursor()

    cur.execute("UPDATE profile SET name=?, tagline=?, about=?, github=?, twitter=?, linkedin=?, website=? WHERE id=1", (name, tagline, about, github, twitter, linkedin, website))
    conn.commit()

    if "avatar" in request.files:
        f = request.files["avatar"]
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = "uploads/" + filename
            f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            cur.execute("UPDATE profile SET avatar_path=? WHERE id=1", (path,))
            conn.commit()

    if "avatar_back" in request.files:
        f = request.files["avatar_back"]
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = "uploads/" + filename
            f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            cur.execute("UPDATE profile SET avatar_back_path=? WHERE id=1", (path,))
            conn.commit()

    conn.close()
    return redirect(url_for("admin_dashboard"))

    return redirect(url_for("admin_dashboard"))



@app.route("/admin/projects/add", methods=["POST"])
def admin_projects_add():
    if not require_login():
        return redirect(url_for("admin_login"))
    title = request.form.get("title", "")
    description = request.form.get("description", "")
    link = request.form.get("link", "")
    github_link = request.form.get("github_link", "")
    image = request.files.get("image")
    image_path = ""
    if image and image.filename:
        filename = secure_filename(image.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(save_path)
        image_path = "uploads/" + filename
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO projects (title, description, image_path, link, github_link, created_at) VALUES (?, ?, ?, ?, ?, ?)", (title, description, image_path, link, github_link, datetime.utcnow().isoformat()))
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
    logo = request.files.get("logo")
    logo_path = ""
    if logo and logo.filename:
        filename = secure_filename(logo.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        logo.save(save_path)
        logo_path = "uploads/" + filename
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO experience (role, company, start_date, end_date, description, logo_path) VALUES (?, ?, ?, ?, ?, ?)", (role, company, start_date, end_date, description, logo_path))
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

@app.route("/admin/projects/edit/<int:pid>", methods=["GET", "POST"])
def admin_projects_edit(pid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    if request.method == "POST":
        title = request.form.get("title", "")
        description = request.form.get("description", "")
        link = request.form.get("link", "")
        github_link = request.form.get("github_link", "")
        image = request.files.get("image")
        
        if image and image.filename:
            filename = secure_filename(image.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(save_path)
            image_path = "uploads/" + filename
            cur.execute("UPDATE projects SET title=?, description=?, link=?, github_link=?, image_path=? WHERE id=?", (title, description, link, github_link, image_path, pid))
        else:
            cur.execute("UPDATE projects SET title=?, description=?, link=?, github_link=? WHERE id=?", (title, description, link, github_link, pid))
        conn.commit()
        conn.close()
        return redirect(url_for("admin_dashboard"))
    
    cur.execute("SELECT * FROM projects WHERE id=?", (pid,))
    project = cur.fetchone()
    conn.close()
    return render_template("edit_project.html", project=project)

@app.route("/admin/experience/edit/<int:eid>", methods=["GET", "POST"])
def admin_experience_edit(eid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    if request.method == "POST":
        role = request.form.get("role", "")
        company = request.form.get("company", "")
        start_date = request.form.get("start_date", "")
        end_date = request.form.get("end_date", "")
        description = request.form.get("description", "")
        logo = request.files.get("logo")
        
        if logo and logo.filename:
            filename = secure_filename(logo.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            logo.save(save_path)
            logo_path = "uploads/" + filename
            cur.execute("UPDATE experience SET role=?, company=?, start_date=?, end_date=?, description=?, logo_path=? WHERE id=?", (role, company, start_date, end_date, description, logo_path, eid))
        else:
            cur.execute("UPDATE experience SET role=?, company=?, start_date=?, end_date=?, description=? WHERE id=?", (role, company, start_date, end_date, description, eid))
        conn.commit()
        conn.close()
        return redirect(url_for("admin_dashboard"))
        
    cur.execute("SELECT * FROM experience WHERE id=?", (eid,))
    experience = cur.fetchone()
    conn.close()
    return render_template("edit_experience.html", experience=experience)

@app.route("/admin/skills/edit/<int:sid>", methods=["GET", "POST"])
def admin_skills_edit(sid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        exploring = 1 if request.form.get("exploring") == "on" else 0
        if name:
            cur.execute("UPDATE skills SET name=?, exploring=? WHERE id=?", (name, exploring, sid))
            conn.commit()
        conn.close()
        return redirect(url_for("admin_dashboard"))
        
    cur.execute("SELECT * FROM skills WHERE id=?", (sid,))
    skill = cur.fetchone()
    conn.close()
    return render_template("edit_skill.html", skill=skill)



@app.route("/admin/research/add", methods=["POST"])
def admin_research_add():
    if not require_login():
        return redirect(url_for("admin_login"))
    title = request.form.get("title", "")
    publication = request.form.get("publication", "")
    date = request.form.get("date", "")
    link = request.form.get("link", "")
    description = request.form.get("description", "")
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO research_papers (title, publication, date, link, description) VALUES (?, ?, ?, ?, ?)", (title, publication, date, link, description))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/research/delete/<int:rid>", methods=["POST"])
def admin_research_delete(rid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM research_papers WHERE id=?", (rid,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/research/edit/<int:rid>", methods=["GET", "POST"])
def admin_research_edit(rid):
    if not require_login():
        return redirect(url_for("admin_login"))
    conn = get_db()
    cur = conn.cursor()
    
    if request.method == "POST":
        title = request.form.get("title", "")
        publication = request.form.get("publication", "")
        date = request.form.get("date", "")
        link = request.form.get("link", "")
        description = request.form.get("description", "")
        
        cur.execute("UPDATE research_papers SET title=?, publication=?, date=?, link=?, description=? WHERE id=?", (title, publication, date, link, description, rid))
        conn.commit()
        conn.close()
        return redirect(url_for("admin_dashboard"))
        
    cur.execute("SELECT * FROM research_papers WHERE id=?", (rid,))
    research_paper = cur.fetchone()
    conn.close()
    return render_template("edit_research.html", research=research_paper)

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)