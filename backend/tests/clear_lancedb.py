"""
清空 LanceDB 所有数据

执行方式：
    cd backend
    python tests/clear_lancedb.py

功能：
    删除指定的 LanceDB 表，释放所有磁盘空间。
    执行后需要重新上传文档才能恢复索引。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lancedb
from config.settings import settings


def main():
    db_path = settings.lancedb_uri
    table_name = settings.lancedb_table_name

    print(f"LanceDB 路径: {db_path}")
    print(f"目标表名: {table_name}")

    db = lancedb.connect(db_path)

    tables = db.table_names()
    print(f"现有表: {tables}")

    if table_name not in tables:
        print(f"\n表 '{table_name}' 不存在，无需清空。")
        return

    tbl = db.open_table(table_name)
    row_count = tbl.count_rows()
    print(f"表 '{table_name}' 当前有 {row_count} 行数据。")

    db.drop_table(table_name)
    print(f"\n表 '{table_name}' 已删除。")

    remaining = db.table_names()
    print(f"剩余表: {remaining}")
    print("清空完成。")


if __name__ == "__main__":
    main()