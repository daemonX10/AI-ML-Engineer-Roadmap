"""
Generate PROGRESS.md dashboard by scanning checkboxes in folders 01-11.
Re-run anytime to refresh the dashboard.

Usage:
    python track_progress.py
"""

import os
import re
from collections import OrderedDict
from datetime import datetime
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, 'PROGRESS.md')

QUESTION_RE = re.compile(r'^(#{2,3})\s+(Question\s+\d+.*)', re.IGNORECASE)
DONE_RE = re.compile(r'^- \[x\] Done', re.IGNORECASE)
BOLD_TEXT_RE = re.compile(r'^\*\*(.+?)\*\*\s*$')

FOLDER_PREFIXES = tuple(f"{i:02d}_" for i in range(1, 12))

SECTION_META = {
    '01_foundations_mathematics':       ('🧮', 'Foundations & Mathematics'),
    '02_programming_tools':             ('🐍', 'Programming Tools'),
    '03_data_science':                  ('📊', 'Data Science'),
    '04_machine_learning':              ('🤖', 'Machine Learning'),
    '06_algorithms_optimization':       ('⚡', 'Algorithms & Optimization'),
    '07_computer_vision':               ('👁️', 'Computer Vision'),
    '08_natural_language_processing':   ('💬', 'Natural Language Processing'),
    '09_large_language_models_genai':   ('🧠', 'LLMs & GenAI'),
    '10_explainable_ai':                ('🔍', 'Explainable AI'),
    '11_model_evaluation_metrics':      ('📏', 'Model Evaluation & Metrics'),
}

FILE_ICONS = {
    'theory_questions':         '📖',
    'coding_questions':         '💻',
    'general_questions':        '📋',
    'scenario_based_questions': '🎯',
    'explainable_ai_questions': '🔍',
    '00_core_questions':        '⭐',
}


def pct(done: int, total: int) -> float:
    return (done / total * 100) if total > 0 else 0.0


def pct_str(done: int, total: int) -> str:
    return f"{pct(done, total):.0f}%"


def text_bar(done: int, total: int, width: int = 20) -> str:
    if total == 0:
        return '`' + '░' * width + '`'
    filled = round(width * done / total)
    bar = '█' * filled + '░' * (width - filled)
    return f'`{bar}`'


def encode_path(rel_path: str) -> str:
    """URL-encode each path segment (handles spaces, special chars) for markdown links."""
    parts = rel_path.split('/')
    return '/'.join(quote(p, safe='') for p in parts)


def heading_to_anchor(heading_text: str) -> str:
    """Convert a markdown heading to a GitHub-compatible anchor.
    e.g. 'Question 1: What is PCA?' -> 'question-1-what-is-pca'
    """
    text = heading_text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)   # remove non-word chars except hyphens
    text = re.sub(r'\s+', '-', text)        # spaces to hyphens
    text = re.sub(r'-+', '-', text)         # collapse multiple hyphens
    return text.strip('-')


def scan_file_detailed(filepath: str) -> list:
    """Returns list of (heading_text, is_done, question_title) for each question."""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        m = QUESTION_RE.match(line.rstrip())
        if m:
            heading_text = m.group(2).strip()  # e.g. "Question 1: What is..." or "Question 1"
            is_done = (i + 1 < len(lines) and DONE_RE.match(lines[i + 1].strip()))

            # Extract a short title for display
            # Case 1: Title in heading "Question 1: What is PCA?"
            colon_idx = heading_text.find(':')
            if colon_idx != -1:
                q_num = heading_text[:colon_idx].strip()
                q_title = heading_text[colon_idx + 1:].strip()
            else:
                q_num = heading_text.strip()
                q_title = ''

            # Case 2: No title in heading — look for bold text on next lines
            if not q_title:
                for j in range(i + 1, min(i + 5, len(lines))):
                    bm = BOLD_TEXT_RE.match(lines[j].strip())
                    if bm:
                        q_title = bm.group(1).strip()
                        break

            # Truncate long titles
            if len(q_title) > 80:
                q_title = q_title[:77] + '...'

            questions.append((heading_text, is_done, q_num, q_title))

    return questions


def file_icon(fname: str) -> str:
    stem = fname.replace('.md', '')
    return FILE_ICONS.get(stem, '📄')


def friendly_file_name(fname: str) -> str:
    return fname.replace('.md', '').replace('_', ' ').title()


def group_files_by_subtopic(file_map: OrderedDict) -> OrderedDict:
    groups = OrderedDict()
    for rel_path, questions in file_map.items():
        parts = rel_path.split('/')
        if len(parts) >= 3:
            subtopic = '/'.join(parts[1:-1])
        else:
            subtopic = '(root)'
        if subtopic not in groups:
            groups[subtopic] = []
        groups[subtopic].append((rel_path, questions))
    return groups


def main():
    # section_key -> OrderedDict{ rel_path: [question_list] }
    sections = OrderedDict()

    for root, dirs, files in os.walk(BASE_DIR):
        rel = os.path.relpath(root, BASE_DIR)
        if rel == '.':
            dirs[:] = sorted(d for d in dirs if d.startswith(FOLDER_PREFIXES))
            continue
        for fname in sorted(files):
            if not fname.endswith('.md'):
                continue
            filepath = os.path.join(root, fname)
            questions = scan_file_detailed(filepath)
            if not questions:
                continue
            top_folder = rel.split(os.sep)[0]
            rel_path = os.path.relpath(filepath, BASE_DIR).replace('\\', '/')
            if top_folder not in sections:
                sections[top_folder] = OrderedDict()
            sections[top_folder][rel_path] = questions

    grand_done = sum(
        sum(1 for _, done, _, _ in qs if done)
        for sec in sections.values() for qs in sec.values()
    )
    grand_total = sum(
        len(qs) for sec in sections.values() for qs in sec.values()
    )
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    L = []

    # ── HEADER ──
    L.append('<div align="center">\n')
    L.append('# 📊 ML Interview Prep — Study Progress\n')
    L.append(f'**Last updated:** {now}\n')
    L.append('</div>\n')
    L.append('')

    # ── OVERALL HERO ──
    p = pct(grand_done, grand_total)
    L.append('---\n')
    L.append('')
    L.append('<div align="center">\n')
    L.append(f'### 🎯 Overall Progress: **{grand_done}** / **{grand_total}** questions completed\n')
    L.append('')
    L.append(f'{text_bar(grand_done, grand_total, 40)} **{p:.1f}%**\n')
    L.append('</div>\n')
    L.append('')

    # ── QUICK STATS ──
    total_files = sum(len(fm) for fm in sections.values())
    completed_files = sum(
        1 for sec in sections.values()
        for qs in sec.values()
        if all(done for _, done, _, _ in qs)
    )
    L.append('| 📁 Sections | 📄 Files Tracked | ✅ Files Complete | 🔥 Questions Left |')
    L.append('|:-----------:|:----------------:|:----------------:|:-----------------:|')
    L.append(f'| **{len(sections)}** | **{total_files}** | **{completed_files}/{total_files}** | **{grand_total - grand_done}** |')
    L.append('')

    # ── SECTION OVERVIEW TABLE ──
    L.append('---\n')
    L.append('## 📋 Section Overview\n')
    L.append('')
    L.append('| | Section | Progress | Done | Total | % |')
    L.append('|:---:|:--------|:---------|-----:|------:|---:|')

    for sec_key, file_map in sections.items():
        emoji, name = SECTION_META.get(sec_key, ('📂', sec_key))
        sec_done = sum(sum(1 for _, d, _, _ in qs if d) for qs in file_map.values())
        sec_total = sum(len(qs) for qs in file_map.values())
        p = pct(sec_done, sec_total)
        bar = text_bar(sec_done, sec_total, 15)
        check = '✅' if sec_done == sec_total else ''
        L.append(f'| {emoji} | [**{name}**](#{sec_key}) {check} | {bar} | {sec_done} | {sec_total} | **{p:.0f}%** |')

    L.append('')

    # ── DETAILED SECTIONS ──
    L.append('---\n')
    L.append('')

    for sec_key, file_map in sections.items():
        emoji, name = SECTION_META.get(sec_key, ('📂', sec_key))
        sec_done = sum(sum(1 for _, d, _, _ in qs if d) for qs in file_map.values())
        sec_total = sum(len(qs) for qs in file_map.values())
        p = pct(sec_done, sec_total)

        L.append(f'<h2 id="{sec_key}">{emoji} {name}</h2>\n')
        L.append('')
        L.append(f'> **{sec_done}/{sec_total}** completed ({p:.0f}%) {text_bar(sec_done, sec_total, 25)}\n')
        L.append('')

        groups = group_files_by_subtopic(file_map)

        for subtopic, file_list in groups.items():
            sub_done = sum(sum(1 for _, d, _, _ in qs if d) for _, qs in file_list)
            sub_total = sum(len(qs) for _, qs in file_list)
            sub_check = ' ✅' if sub_done == sub_total else ''

            display_sub = subtopic.split('/')[-1] if '/' in subtopic else subtopic
            display_sub = display_sub.replace('_', ' ')
            if display_sub != '(root)':
                display_sub = display_sub.title()

            L.append(f'<details{"" if sub_done < sub_total else " open"}>')
            L.append(f'<summary><strong>{display_sub}</strong> — {sub_done}/{sub_total} ({pct(sub_done, sub_total):.0f}%){sub_check}</summary>\n')
            L.append('')

            for rel_path, questions in file_list:
                fname = rel_path.split('/')[-1]
                icon = file_icon(fname)
                display = friendly_file_name(fname)
                file_done = sum(1 for _, d, _, _ in questions if d)
                file_total = len(questions)
                fp = pct(file_done, file_total)
                bar = text_bar(file_done, file_total, 10)
                encoded = encode_path(rel_path)
                file_check = '✅' if file_done == file_total else '⬜'

                L.append(f'> {file_check} {icon} **[{display}]({encoded})** — {file_done}/{file_total} ({fp:.0f}%) {bar}\n')
                L.append('')

                # Question-level listing
                for heading_text, is_done, q_num, q_title in questions:
                    anchor = heading_to_anchor(heading_text)
                    q_link = f'{encoded}#{anchor}'
                    check = '✅' if is_done else '⬜'
                    if q_title:
                        L.append(f'- {check} [{q_num}]({q_link}): {q_title}')
                    else:
                        L.append(f'- {check} [{q_num}]({q_link})')

                L.append('')

            L.append('</details>\n')
            L.append('')

        L.append('---\n')
        L.append('')

    # ── FOOTER ──
    L.append('<div align="center">\n')
    L.append('')
    L.append('*Generated by `track_progress.py` — run `python track_progress.py` to refresh*\n')
    L.append('')
    L.append('</div>\n')

    content = '\n'.join(L)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"📊 Dashboard generated: PROGRESS.md")
    print(f"   Overall: {grand_done}/{grand_total} ({pct_str(grand_done, grand_total)})")
    print(f"   Sections: {len(sections)}")
    print(f"   Files tracked: {total_files}")
    print(f"   Individual questions listed with clickable links")


if __name__ == '__main__':
    main()
