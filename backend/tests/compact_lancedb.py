"""
LanceDB 数据碎片整理

执行方式：
    cd backend
    python tests/output/compact_lancedb.py

功能：
    调用 LanceDB 的 optimize() 方法，执行：
    1. Compaction — 合并小文件为大文件，提升读取性能
    2. Prune — 清理 delete 产生的旧版本数据，释放磁盘空间
    3. Index  — 优化向量索引，将新增数据纳入已有索引

    建议在频繁增删数据后定期执行，或当数据量超过 10 万行时执行。
"""

import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lancedb
from config.settings import settings


def main():
    db_path = settings.lancedb_uri
    table_name = settings.lancedb_table_name

    print(f"LanceDB 路径: {db_path}")
    print(f"目标表名: {table_name}")

    db = lancedb.connect(db_path)

    if table_name not in db.table_names():
        print(f"\n表 '{table_name}' 不存在。")
        return

    tbl = db.open_table(table_name)
    row_count = tbl.count_rows()
    print(f"表 '{table_name}' 当前有 {row_count} 行数据。")

    print("\n开始整理...")
    print("  1. Compaction — 合并小文件")
    print("  2. Prune — 清理旧版本数据")
    print("  3. Index — 优化向量索引")

    try:
        stats = tbl.optimize(
            cleanup_older_than=timedelta(0),
            delete_unverified=False,
        )
        print(f"\n整理完成。")
        print(f"  统计信息: {stats}")
    except Exception as e:
        print(f"\n整理失败: {e}")
        sys.exit(1)

    print("碎片整理完成。")


if __name__ == "__main__":
    main()