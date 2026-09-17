"""
LanceDB 浏览工具 —— 执行即可获取所有内容

运行方式：
    cd backend
    .venv\Scripts\python.exe tests\browse_lancedb.py

可选参数：
    --table  表名，默认 documents
    --source 按文件名过滤

结果同时输出到控制台和 tests/output/ 目录下的文本文件。
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lancedb
from config.settings import settings

_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def browse(table_name="documents", source_file=None):
    lines = []

    def emit(msg):
        print(msg)
        lines.append(msg)

    db = lancedb.connect(settings.lancedb_uri)

    tables = db.table_names()
    emit(f"Tables: {tables}")
    emit("-" * 80)

    if table_name not in tables:
        emit(f"Table '{table_name}' not found. Available: {tables}")
        return lines

    tbl = db.open_table(table_name)
    df = tbl.to_pandas()

    if source_file:
        df = df[df["source_file"] == source_file]

    total = len(df)
    emit(f"Table: {table_name}  |  Total rows: {total}")
    emit("=" * 80)

    if total == 0:
        emit("(empty table)")
        return lines

    for i, (_, row) in enumerate(df.iterrows()):
        idx = i + 1
        meta = {}
        try:
            meta = json.loads(row.get("metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass

        vec = row.get("vector", None)
        if vec is not None and hasattr(vec, "__len__"):
            vec_preview = f"[{', '.join(f'{v:.6f}' for v in vec[:5])} ...] dim={len(vec)}"
        else:
            vec_preview = str(vec)

        emit(f"--- Row #{idx} ---")
        emit(f"  id          : {row.get('id', '')}")
        emit(f"  source_file : {row.get('source_file', '')}")
        emit(f"  doc_type    : {row.get('doc_type', '')}")
        emit(f"  page_number : {row.get('page_number', -1)}")
        emit(f"  chunk_index : {row.get('chunk_index', -1)}")
        emit(f"  file_hash   : {row.get('file_hash', '')}")
        emit(f"  file_size   : {row.get('file_size', 0)}")
        emit(f"  text        : {row.get('text', '')}")
        emit(f"  vector      : {vec_preview}")
        emit(f"  metadata    : {json.dumps(meta, ensure_ascii=False)}")
        emit(f"  created_at  : {row.get('created_at', '')}")
        emit(f"  updated_at  : {row.get('updated_at', '')}")
        emit("")

    emit(f"Total: {total} rows")
    return lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Browse LanceDB - show all content")
    parser.add_argument("--table", default="documents", help="Table name (default: documents)")
    parser.add_argument("--source", default=None, help="Filter by source file name")
    args = parser.parse_args()

    lines = browse(
        table_name=args.table,
        source_file=args.source,
    )

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_suffix = f"_{args.source}" if args.source else ""
    out_path = os.path.join(_OUTPUT_DIR, f"lancedb_{args.table}{source_suffix}_{timestamp}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nOutput saved to: {out_path}")