"""
HSCM Config — Paths, auto-locate, and auto-download
Never loads full files. Sets pointers only.
"""

import os
import urllib.request
import hashlib
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
CHECKPOINT_DIR= BASE_DIR / "checkpoints"
INDEX_DIR     = BASE_DIR / "index"
LOG_DIR       = BASE_DIR / "logs"
BUILD_DIR     = BASE_DIR / "build"

for d in [DATA_DIR, CHECKPOINT_DIR, INDEX_DIR, LOG_DIR, BUILD_DIR]:
    d.mkdir(exist_ok=True)

# =============================================================================
# DICTIONARY SOURCES — ordered by preference
# System will use first one it finds or can download
# =============================================================================

CMU_DICT_URL  = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
CMU_DICT_NAME = "cmudict-0.7b"
CMU_DICT_MD5  = None  # will verify size instead

# Search locations for user-supplied dictionary
# Checks uploads, home, common locations — in order
DICT_SEARCH_PATHS = [
    Path("/mnt/user-data/uploads"),          # claude.ai upload dir
    Path.home(),
    Path.home() / "Downloads",
    Path.home() / "Documents",
    BASE_DIR / "data",
]

DICT_EXTENSIONS = [".dict", ".txt", ".wordset", ".json", ".csv"]
DICT_NAME_PATTERNS = [
    "cmudict", "wordset", "dictionary", "words", "lexicon", "vocab"
]

# =============================================================================
# DICTIONARY MANAGER — pointer only, no full loads
# =============================================================================

class DictPointer:
    """
    Holds a file pointer to a dictionary.
    Reads one line at a time. Never loads full file into memory.
    """
    def __init__(self, path: Path, format: str, line_count: int = None):
        self.path       = path
        self.format     = format       # "cmu", "wordlist", "wordset_json"
        self.line_count = line_count
        self._offset_index = {}        # {word: byte_offset} — built lazily

    def __repr__(self):
        return f"DictPointer({self.path.name}, format={self.format}, lines={self.line_count})"

    def iter_words(self, batch_size: int = 1000):
        """Stream words in batches. Yields (word, phonemes_or_None) tuples."""
        with open(self.path, 'rb') as f:
            batch = []
            for line in f:
                try:
                    line = line.decode('latin-1').strip()
                except:
                    continue
                if not line or line.startswith(';;;'):
                    continue
                parsed = self._parse_line(line)
                if parsed:
                    batch.append(parsed)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch

    def _parse_line(self, line: str):
        if self.format == "cmu":
            # CMU format: WORD  P1 P2 P3 ...
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0].lower().rstrip('(12)')  # strip variant markers
                phonemes = parts[1:]
                return (word, phonemes)
        elif self.format == "wordlist":
            word = line.strip().lower()
            if word.isalpha():
                return (word, None)
        return None

    def seek_word(self, word: str):
        """Return phonemes for a specific word using byte offset index."""
        if not self._offset_index:
            self._build_offset_index()
        offset = self._offset_index.get(word.lower())
        if offset is None:
            return None
        with open(self.path, 'rb') as f:
            f.seek(offset)
            line = f.readline().decode('latin-1').strip()
            parsed = self._parse_line(line)
            return parsed[1] if parsed else None

    def _build_offset_index(self):
        """Build word → byte_offset index. One pass, then O(1) lookups."""
        print(f"  Building offset index for {self.path.name}...")
        with open(self.path, 'rb') as f:
            offset = 0
            for line in f:
                try:
                    decoded = line.decode('latin-1').strip()
                except:
                    offset += len(line)
                    continue
                if decoded and not decoded.startswith(';;;'):
                    parsed = self._parse_line(decoded)
                    if parsed:
                        self._offset_index[parsed[0]] = offset
                offset += len(line)
        print(f"  Index built: {len(self._offset_index)} entries")


def _count_lines(path: Path) -> int:
    """Fast line count using read in chunks."""
    count = 0
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            count += chunk.count(b'\n')
    return count


def _detect_format(path: Path) -> str:
    """Detect dictionary format from content sample."""
    with open(path, 'rb') as f:
        sample = f.read(4096).decode('latin-1', errors='replace')

    if ';;;' in sample or (
        len(sample.split('\n')[1].split()) > 3 and
        any(p in sample for p in ['AH', 'EH', 'IH', 'OW', 'AE'])
    ):
        return "cmu"
    if sample.strip().startswith('{') or sample.strip().startswith('['):
        return "wordset_json"
    return "wordlist"


def locate_or_download_dictionary() -> DictPointer:
    """
    1. Check user upload locations
    2. Check data dir for previously downloaded copy
    3. Auto-download CMU dict
    Returns a DictPointer — never loads full content.
    """
    # Step 1: Search for user-supplied dictionary
    print("  Scanning for dictionary files...")
    for search_dir in DICT_SEARCH_PATHS:
        if not search_dir.exists():
            continue
        for fpath in search_dir.iterdir():
            if not fpath.is_file():
                continue
            name_lower = fpath.name.lower()
            ext_match  = any(fpath.suffix == ext for ext in DICT_EXTENSIONS)
            name_match = any(pat in name_lower for pat in DICT_NAME_PATTERNS)
            if ext_match or name_match:
                fmt = _detect_format(fpath)
                lc  = _count_lines(fpath)
                print(f"  Found: {fpath} ({fmt}, {lc} lines)")
                return DictPointer(fpath, fmt, lc)

    # Step 2: Check data dir for cached CMU dict
    cached = DATA_DIR / CMU_DICT_NAME
    if cached.exists():
        lc = _count_lines(cached)
        print(f"  Using cached CMU dict ({lc} lines)")
        return DictPointer(cached, "cmu", lc)

    # Step 3: Download CMU dict
    print(f"  Downloading CMU Pronouncing Dictionary...")
    print(f"  Source: {CMU_DICT_URL}")
    try:
        urllib.request.urlretrieve(CMU_DICT_URL, cached)
        lc = _count_lines(cached)
        print(f"  Downloaded: {lc} lines → {cached}")
        return DictPointer(cached, "cmu", lc)
    except Exception as e:
        raise RuntimeError(
            f"Could not locate or download a dictionary.\n"
            f"Place your dictionary file in {DATA_DIR} and re-run.\n"
            f"Download error: {e}"
        )
