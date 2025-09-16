import streamlit as st
import sqlite3
import hashlib
import io
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

try:
    from pypdf import PdfReader  # modern fork of PyPDF2
except Exception as e:
    st.error("Failed to import pypdf. Please add 'pypdf' to your environment (pip install pypdf).")
    raise

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="PDF → SQLite (Laser Microstructure & Multiscale Modeling Papers)",
    layout="wide",
)

st.title("PDF → SQLite builder for laser microstructure & multiscale modeling papers")
st.write(
    "Upload multiple PDF files. This app will extract full text and lightweight metadata, and write two SQLite databases: "
    "`lasermicrostructure_metadata.db` and `lasermicrostructure_knowledgeuniverse.db`."
)

# -----------------------------
# Keyword universe for tagging (very lightweight)
# -----------------------------
KEY_TOPICS = {
    "laser": [
        r"laser", r"beam", r"welding", r"cladding", r"additive", r"3D printing",
        r"selective laser", r"SLM", r"SLS", r"direct energy deposition", r"DED"
    ],
    "microstructure": [
        r"microstructure", r"grain", r"phase", r"texture", r"precipitate",
        r"defect", r"dislocation", r"twin", r"morphology"
    ],
    "multicomponent alloys": [
        r"multicomponent", r"high entropy", r"HEA", r"CCA", r"complex concentrated",
        r"alloy design", r"compositionally complex"
    ],
    "heat source models": [
        r"heat source", r"thermal model", r"Gaussian", r"Rosenthal", r"Goldak",
        r"moving heat source", r"thermal efficiency"
    ],
    "thermal history": [
        r"thermal history", r"cooling rate", r"thermal cycle", r"temperature gradient",
        r"thermal gradient", r"heating rate"
    ],
    "multiscale models": [
        r"multiscale", r"multiscale modeling", r"scale bridging", r"hierarchical",
        r"multi-scale"
    ],
    "artificial intelligence": [
        r"artificial intelligence", r"AI", r"neural network", r"deep learning",
        r"expert system", r"knowledge representation"
    ],
    "machine learning": [
        r"machine learning", r"ML", r"supervised", r"unsupervised", r"reinforcement",
        r"regression", r"classification", r"clustering"
    ],
    "featurization": [
        r"featurization", r"feature engineering", r"descriptor", r"fingerprint",
        r"representation learning"
    ],
    "molecular dynamics": [
        r"molecular dynamics", r"MD simulation", r"atomistic", r"interatomic potential",
        r"empirical potential"
    ],
    "dft": [
        r"density functional", r"DFT", r"first principles", r"ab initio",
        r"quantum mechanics", r"electronic structure"
    ],
    "phase field method": [
        r"phase field", r"phase-field", r"order parameter", r"Ginzburg-Landau",
        r"Cahn-Hilliard", r"Allen-Cahn"
    ],
    "micrographs": [
        r"micrograph", r"SEM", r"TEM", r"EBSD", r"optical microscopy", r"OM",
        r"scanning electron", r"transmission electron", r"electron backscatter"
    ],
    "XRD": [
        r"X-ray diffraction", r"XRD", r"diffraction pattern", r"Bragg's law",
        r"crystallography", r"Rietveld"
    ],
    "properties": [
        r"mechanical properties", r"tensile", r"hardness", r"yield strength",
        r"ductility", r"toughness", r"fatigue", r"creep", r"fracture"
    ],
}

KEY_TOPIC_PATTERNS = {k: [re.compile(pat, re.IGNORECASE) for pat in pats] for k, pats in KEY_TOPICS.items()}

# -----------------------------
# Helpers
# -----------------------------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def parse_pdf_date(raw: str) -> str:
    """Parse PDF date like D:YYYYMMDDHHmmSS and return ISO 8601; fallback to raw."""
    if not raw:
        return ""
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", str(raw))
    if not m:
        return str(raw)
    parts = [m.group(i) or "01" for i in range(1, 7)]
    try:
        dt = datetime(
            int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        )
        return dt.isoformat()
    except Exception:
        return str(raw)


def extract_metadata(reader: PdfReader, file_name: str, file_bytes: bytes, full_text: str) -> Dict[str, Any]:
    info = reader.metadata or {}
    # pypdf exposes keys without slash, e.g., info.title, info.author
    meta = {
        "filename": file_name,
        "title": getattr(info, "title", None) or "",
        "authors": getattr(info, "author", None) or "",
        "subject": getattr(info, "subject", None) or "",
        "keywords": getattr(info, "keywords", None) or "",
        "producer": getattr(info, "producer", None) or "",
        "creation_date": parse_pdf_date(getattr(info, "creation_date", "")),
        "mod_date": parse_pdf_date(getattr(info, "mod_date", "")),
        "pages": len(reader.pages),
        "size_bytes": len(file_bytes),
        "checksum_sha256": sha256_bytes(file_bytes),
        "upload_time": datetime.utcnow().isoformat(timespec="seconds"),
        # Attempt a year from subject/title if present
        "year_guess": "",
        "doi_guess": "",
        "abstract": "",
    }
    year_match = re.search(r"(19|20)\d{2}", " ".join([meta["title"], meta["subject"]]))
    if year_match:
        meta["year_guess"] = year_match.group(0)

    # DOI heuristic: look anywhere in text
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", full_text, flags=re.IGNORECASE)
    if doi_match:
        meta["doi_guess"] = doi_match.group(0)

    # Abstract extraction heuristic
    abstract_match = re.search(r"(?i)abstract[:\s]*([\s\S]{1,2000}?)(?=\n\n|\n\s*(?:1\.|introduction|keywords))", full_text)
    if abstract_match:
        meta["abstract"] = abstract_match.group(1).strip()
    else:
        # Fallback: first 500 characters if abstract not found
        meta["abstract"] = full_text[:500].strip()

    return meta


def count_topic_hits(text: str) -> Dict[str, int]:
    counts = {}
    for k, pats in KEY_TOPIC_PATTERNS.items():
        c = 0
        for pat in pats:
            c += len(pat.findall(text))
        counts[k] = int(c)
    return counts


# -----------------------------
# Database setup
# -----------------------------

META_DB = "lasermicrostructure_metadata.db"
UNIV_DB = "lasermicrostructure_knowledgeuniverse.db"


def init_meta_db(path: str = META_DB) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            title TEXT,
            authors TEXT,
            subject TEXT,
            keywords TEXT,
            abstract TEXT,
            producer TEXT,
            creation_date TEXT,
            mod_date TEXT,
            year_guess TEXT,
            doi_guess TEXT,
            pages INTEGER,
            size_bytes INTEGER,
            checksum_sha256 TEXT UNIQUE,
            upload_time TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS topic_counts (
            paper_id INTEGER,
            topic TEXT,
            count INTEGER,
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        );
        """
    )
    con.commit()
    return con


def init_universe_db(path: str = UNIV_DB) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            paper_checksum TEXT,
            filename TEXT,
            page_num INTEGER,
            text TEXT
        );
        """
    )
    # Try to add FTS5 for fast search
    try:
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS page_fts USING fts5(
                text,
                filename UNINDEXED,
                page_num UNINDEXED,
                paper_checksum UNINDEXED,
                content='pages', content_rowid='id'
            );
            """
        )
        # Triggers to sync FTS with pages
        cur.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
              INSERT INTO page_fts(rowid, text, filename, page_num, paper_checksum)
              VALUES (new.id, new.text, new.filename, new.page_num, new.paper_checksum);
            END;
            CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
              INSERT INTO page_fts(page_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
              INSERT INTO page_fts(page_fts, rowid, text) VALUES('delete', old.id, old.text);
              INSERT INTO page_fts(rowid, text, filename, page_num, paper_checksum)
              VALUES (new.id, new.text, new.filename, new.page_num, new.paper_checksum);
            END;
            """
        )
    except sqlite3.OperationalError:
        # FTS5 not available; continue without it
        pass
    con.commit()
    return con


# -----------------------------
# Core processing
# -----------------------------

def extract_fulltext(reader: PdfReader) -> Tuple[List[str], str]:
    pages_text: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        # Normalize whitespace
        txt = re.sub(r"\s+", " ", txt).strip()
        pages_text.append(txt)
    full_text = "\n".join(pages_text)
    return pages_text, full_text


def insert_meta(con: sqlite3.Connection, meta: Dict[str, Any], topic_counts: Dict[str, int]) -> Tuple[int, bool]:
    cur = con.cursor()
    # Upsert by checksum to avoid duplicates
    cur.execute(
        """
        INSERT OR IGNORE INTO papers (
            filename, title, authors, subject, keywords, abstract, producer, creation_date, mod_date,
            year_guess, doi_guess, pages, size_bytes, checksum_sha256, upload_time
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            meta["filename"], meta["title"], meta["authors"], meta["subject"], meta["keywords"],
            meta["abstract"], meta["producer"], meta["creation_date"], meta["mod_date"], 
            meta["year_guess"], meta.get("doi_guess", ""), meta["pages"], meta["size_bytes"], 
            meta["checksum_sha256"], meta["upload_time"],
        ),
    )
    # Get id
    cur.execute("SELECT id FROM papers WHERE checksum_sha256 = ?", (meta["checksum_sha256"],))
    row = cur.fetchone()
    paper_id = int(row[0]) if row else -1

    # Detect duplicate
    duplicate = cur.rowcount == 0

    # Clear existing topic counts then insert
    cur.execute("DELETE FROM topic_counts WHERE paper_id = ?", (paper_id,))
    cur.executemany(
        "INSERT INTO topic_counts (paper_id, topic, count) VALUES (?,?,?)",
        [(paper_id, k, int(v)) for k, v in topic_counts.items()],
    )

    con.commit()
    return paper_id, duplicate


def insert_pages(con: sqlite3.Connection, filename: str, checksum: str, pages_text: List[str]):
    cur = con.cursor()
    # Remove any existing pages for this checksum (rebuild)
    cur.execute("DELETE FROM pages WHERE paper_checksum = ?", (checksum,))
    cur.executemany(
        "INSERT INTO pages (paper_checksum, filename, page_num, text) VALUES (?,?,?,?)",
        [(checksum, filename, i + 1, t) for i, t in enumerate(pages_text)],
    )
    con.commit()


# -----------------------------
# UI: File upload + processing
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Drop in research papers related to laser processing, microstructure, multiscale modeling, and related topics.",
)

colA, colB = st.columns([1, 1])
with colA:
    meta_db_name = st.text_input("Metadata DB filename", META_DB)
with colB:
    univ_db_name = st.text_input("Universe DB filename", UNIV_DB)

if st.button("Build SQLite Databases", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        st.stop()

    meta_con = init_meta_db(meta_db_name)
    univ_con = init_universe_db(univ_db_name)

    progress = st.progress(0)
    log = st.empty()

    for idx, uf in enumerate(uploaded_files, start=1):
        try:
            file_bytes = uf.read()
            checksum = sha256_bytes(file_bytes)
            reader = PdfReader(io.BytesIO(file_bytes))

            pages_text, full_text = extract_fulltext(reader)
            meta = extract_metadata(reader, uf.name, file_bytes, full_text)

            topic_counts = count_topic_hits(full_text)

            paper_id, duplicate = insert_meta(meta_con, meta, topic_counts)
            insert_pages(univ_con, uf.name, checksum, pages_text)

            if duplicate:
                st.warning(f"Duplicate detected: {uf.name} (checksum {checksum[:10]}…) already exists in DB.")
            else:
                log.info(
                    f"Processed {uf.name} → paper_id={paper_id}, pages={len(pages_text)}, checksum={checksum[:10]}…"
                )
        except Exception as e:
            st.error(f"Failed to process {uf.name}: {e}")
        finally:
            progress.progress(min(idx / max(len(uploaded_files), 1), 1.0))

    # Close connections to ensure files are flushed to disk before download
    meta_con.close()
    univ_con.close()

    st.success("All done! Databases created.")

    # Offer downloads
    def file_exists(path: str) -> bool:
        try:
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception:
            return False

    dl_cols = st.columns(2)
    with dl_cols[0]:
        if file_exists(meta_db_name):
            with open(meta_db_name, "rb") as f:
                st.download_button(
                    label=f"Download {meta_db_name}", file_name=meta_db_name, data=f.read(), mime="application/x-sqlite3"
                )
        else:
            st.warning("Metadata DB not found.")
    with dl_cols[1]:
        if file_exists(univ_db_name):
            with open(univ_db_name, "rb") as f:
                st.download_button(
                    label=f"Download {univ_db_name}", file_name=univ_db_name, data=f.read(), mime="application/x-sqlite3"
                )
        else:
            st.warning("Universe DB not found.")

    # Quick preview: show a small table from metadata
    try:
        con = sqlite3.connect(meta_db_name)
        df = None
        import pandas as pd  # optional
        df = pd.read_sql_query(
            "SELECT id, filename, title, authors, year_guess, pages, size_bytes, checksum_sha256 FROM papers ORDER BY id DESC LIMIT 100",
            con,
        )
        st.dataframe(df, use_container_width=True)
        con.close()
    except Exception:
        st.info("(Optional) Install pandas to preview metadata table.")

st.markdown(
    """
---
**Notes**
- Duplicate PDFs are detected via SHA-256 checksum. A warning will appear if a file already exists in DB (metadata preserved; pages rebuilt).
- If your SQLite build supports FTS5, `lasermicrostructure_knowledgeuniverse.db` includes a full-text search virtual table `page_fts` synchronized with `pages`.
- Topic counters are simple regex hits across the whole document for categories: laser, microstructure, multicomponent alloys, heat source models, etc.
- You can rename the DB outputs above if you need variants.
    """
)
